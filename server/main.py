from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
from pydantic import BaseModel
from pathlib import Path
from typing import List
import os
import tempfile
import subprocess
import base64
import asyncio
import shutil
import logging
import uvicorn

load_dotenv()

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable is required")

ALLOWED_ORIGINS = os.getenv(
    "ALLOWED_ORIGINS", "http://localhost:3000").split(",")
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB
SUPPORTED_FORMATS = {".mp4", ".mov", ".avi", ".mkv", ".wmv", ".flv", ".webm"}
KEYFRAME_INTERVAL = 5  # seconds

app = FastAPI(title="Video to Text Summarization", version="1.0.0")

try:
    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY)
    logger.info("OpenAI client initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize OpenAI client: {e}")
    raise RuntimeError("Could not initialize OpenAI client")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    success: bool
    response: str


class TranscribeResponse(BaseModel):
    success: bool
    text: str


class VideoProcessor:
    def __init__(self):
        self.temp_dir = tempfile.mkdtemp()

    async def run_ffmpeg_command(self, cmd: List[str], error_msg: str):
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, check=True)
            return result.stdout
        except subprocess.CalledProcessError as e:
            logger.error(f"FFmpeg error: {e.stderr}")
            raise HTTPException(
                status_code=500, detail=f"{error_msg}: {e.stderr}")

    async def extract_audio(self, video_path: str) -> str:
        # First, check if the video has an audio stream
        probe_cmd = ["ffprobe", "-v", "quiet", "-select_streams", "a",
                     "-show_entries", "stream=index", "-of", "csv=p=0", video_path]

        try:
            result = subprocess.run(
                probe_cmd, capture_output=True, text=True, check=True)
            if not result.stdout.strip():
                logger.info(
                    f"No audio stream found in video: {os.path.basename(video_path)}")
                return None
        except subprocess.CalledProcessError:
            logger.warning(
                f"Could not probe audio streams in video: {os.path.basename(video_path)}")
            return None

        # Proceed with audio extraction if audio stream exists
        audio_path = os.path.join(self.temp_dir, "audio.wav")
        cmd = ["ffmpeg", "-i", video_path, "-vn", "-acodec",
               "pcm_s16le", "-ar", "16000", "-ac", "1", "-y", audio_path]
        await self.run_ffmpeg_command(cmd, "Audio extraction failed")
        return audio_path

    async def extract_keyframes(self, video_path: str) -> List[str]:
        # First, check if the video has a video stream
        probe_cmd = ["ffprobe", "-v", "quiet", "-select_streams", "v",
                     "-show_entries", "stream=index", "-of", "csv=p=0", video_path]

        try:
            result = subprocess.run(
                probe_cmd, capture_output=True, text=True, check=True)
            if not result.stdout.strip():
                logger.info(
                    f"No video stream found in video: {os.path.basename(video_path)}")
                return []
        except subprocess.CalledProcessError:
            logger.warning(
                f"Could not probe video streams in video: {os.path.basename(video_path)}")
            return []

        frames_dir = os.path.join(self.temp_dir, "frames")
        os.makedirs(frames_dir, exist_ok=True)

        cmd = ["ffmpeg", "-i", video_path, "-vf", f"fps=1/{KEYFRAME_INTERVAL}", "-q:v", "3", "-y",
               os.path.join(frames_dir, "frame_%03d.jpg")]

        try:
            await self.run_ffmpeg_command(cmd, "Keyframe extraction failed")
        except HTTPException as e:
            logger.warning(
                f"Keyframe extraction failed for {os.path.basename(video_path)}: {e.detail}")
            return []

        frame_files = [f for f in os.listdir(frames_dir) if f.endswith('.jpg')]
        if not frame_files:
            logger.warning(
                f"No keyframes extracted from video: {os.path.basename(video_path)}")
            return []

        return sorted([os.path.join(frames_dir, f) for f in frame_files])

    async def process_video(self, video_path: str, filename: str) -> tuple:
        audio_path, frame_paths = await asyncio.gather(
            self.extract_audio(video_path),
            self.extract_keyframes(video_path)
        )

        # Handle case where there's no audio or no video
        transcript = None
        visual_descriptions = []

        if audio_path:
            transcript = await transcribe_audio(audio_path)
            logger.info(f"Audio transcription completed for {filename}")
        else:
            logger.info(
                f"Skipping audio transcription - no audio track found in {filename}")

        if frame_paths:
            visual_descriptions = await self.analyze_frames(frame_paths)
            logger.info(
                f"Visual analysis completed for {filename} - {len(frame_paths)} frames analyzed")
        else:
            logger.info(
                f"Skipping visual analysis - no video frames found in {filename}")

        # Ensure we have at least one type of content to analyze
        if not transcript and not visual_descriptions:
            raise HTTPException(
                status_code=400,
                detail="Video file contains neither audio nor visual content that can be processed"
            )

        summary = await generate_summary(transcript, visual_descriptions, filename)
        return summary, transcript, visual_descriptions, len(frame_paths)

    async def analyze_frames(self, frame_paths: List[str]) -> List[str]:
        if not frame_paths:
            return []

        semaphore = asyncio.Semaphore(5)

        async def analyze_frame(frame_path: str, index: int):
            async with semaphore:
                timestamp = index * KEYFRAME_INTERVAL
                return await analyze_images(frame_path, timestamp)

        tasks = [analyze_frame(frame_path, i)
                 for i, frame_path in enumerate(frame_paths)]
        descriptions = await asyncio.gather(*tasks, return_exceptions=True)

        return [desc if isinstance(desc, str) else f"[{i*KEYFRAME_INTERVAL}s]: Analysis failed"
                for i, desc in enumerate(descriptions)]

    def cleanup(self):
        try:
            shutil.rmtree(self.temp_dir)
        except Exception as e:
            logger.error(f"Error cleaning up temp files: {str(e)}")


async def call_openai_api(model: str, messages: list, max_tokens: int = 500, temperature: float = 0.3) -> str:
    try:
        response = client.chat.completions.create(
            model=model, messages=messages, max_tokens=max_tokens, temperature=temperature
        )
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"OpenAI API error: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"OpenAI API error: {str(e)}")


async def transcribe_audio(audio_path: str) -> str:
    try:
        with open(audio_path, "rb") as audio_file:
            transcript = client.audio.transcriptions.create(
                model="gpt-4o-mini-transcribe", file=audio_file, response_format="text"
            )
            return transcript
    except Exception as e:
        logger.error(f"Error transcribing audio: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Audio transcription failed: {str(e)}")


async def analyze_images(image_path: str, timestamp: int) -> str:
    try:
        with open(image_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode('utf-8')

        messages = [{
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": f"Analyze this video frame at {timestamp}s. Describe: 1) Main subjects/people and their actions, 2) Key objects and environment, 3) Text/graphics visible, 4) Overall scene context. Be specific and concise."
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                }
            ]
        }]

        return await call_openai_api("gpt-4.1", messages, max_tokens=200)
    except Exception as e:
        logger.error(f"Error analyzing frame at {timestamp}s: {str(e)}")
        return f"[{timestamp}s]: Analysis failed"


async def generate_summary(transcript: str, visual_descriptions: List[str], filename: str) -> str:
    try:
        has_audio = transcript is not None and transcript.strip()
        has_visual = visual_descriptions and len(visual_descriptions) > 0

        audio_section = ""
        visual_section = ""

        if has_audio:
            audio_section = f"""
AUDIO TRANSCRIPT:
{transcript}
"""

        if has_visual:
            visual_content = "\n".join(visual_descriptions)
            visual_section = f"""
VISUAL ANALYSIS (timestamped):
{visual_content}
"""

        if has_audio and has_visual:
            content_description = "video content with both audio and visual elements"
        elif has_audio:
            content_description = "audio content from this video file"
        else:
            content_description = "visual content from this video file"

        prompt = f"""Analyze this {content_description} and create a comprehensive yet concise summary.

VIDEO FILE: {filename}
{audio_section}{visual_section}
Write a detailed summary that clearly explains what happens in the video from beginning to end. The summary should be written in paragraph form using clear, professional language that flows naturally and cohesively.

Cover the following aspects in your narrative (only include sections that are relevant based on available content):
- What the video is about and who is involved
- The main topics, discussions, or demonstrations presented
- Key visual elements, scenes, or actions shown throughout
- Important conclusions, data, or insights shared
- The overall purpose and value of the content

Requirements:
- Use complete sentences that flow smoothly from one idea to the next
- Highlight the most important and unique moments in the video
- Be specific and factual; avoid vague or generic descriptions
- Describe visual elements such as charts, graphics, demonstrations, or locations when relevant
- Keep the summary between 300 to 400 words
- If only audio is available, focus on the spoken content, topics discussed, and key insights
- If only visual content is available, focus on what is shown, demonstrated, or displayed
- Omit any of the above elements if they are not present in the video

Focus on telling the story of the video in a way that is informative, accurate, and easy to understand for someone who has not seen it."""

        messages = [
            {
                "role": "system",
                "content": "You are an expert video content analyst specializing in creating executive-level summaries. Your summaries help busy professionals quickly understand video content without watching. Focus on actionable insights, key data points, and business value. Be precise, structured, and eliminate fluff. Adapt your analysis based on the available content - whether it's audio-only, visual-only, or combined content."
            },
            {"role": "user", "content": prompt}
        ]

        return await call_openai_api("gpt-4.1", messages, max_tokens=800, temperature=0.2)
    except Exception as e:
        logger.error(f"Error generating summary: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Summary generation failed: {str(e)}")


async def generate_chat_response(message: str) -> str:
    messages = [
        {
            "role": "system",
            "content": "You are a helpful AI assistant. Provide detailed, clear and informative responses."
        },
        {"role": "user", "content": message}
    ]
    return await call_openai_api("gpt-4.1-mini", messages, max_tokens=500, temperature=0.7)


@app.get("/")
async def root():
    return {"message": "Video to Text Summarization", "status": "healthy"}


@app.post("/api/upload")
async def upload_video(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    file_extension = Path(file.filename).suffix.lower()
    if file_extension not in SUPPORTED_FORMATS:
        raise HTTPException(
            status_code=400, detail=f"Unsupported format. Supported: {', '.join(SUPPORTED_FORMATS)}")

    if file.size and file.size > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=400, detail=f"File too large. Max: {MAX_FILE_SIZE // (1024*1024)}MB")

    processor = VideoProcessor()
    try:
        video_path = os.path.join(processor.temp_dir, f"video{file_extension}")
        with open(video_path, 'wb') as f:
            f.write(await file.read())

        logger.info(f"Processing video: {file.filename}")

        summary, transcript, visual_descriptions, frames_count = await processor.process_video(video_path, file.filename)

        logger.info(f"Successfully processed video: {file.filename}")

        response_data = {
            "success": True,
            "summary": summary,
            "metadata": {
                "filename": file.filename,
                "transcript_length": len(transcript) if transcript else 0,
                "has_audio": transcript is not None,
                "has_visual": frames_count > 0,
                "frames_analyzed": frames_count,
                "keyframe_interval": KEYFRAME_INTERVAL
            }
        }

        logger.info(f"Response data: {response_data}")
        return JSONResponse(content=response_data)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error processing video: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Processing failed: {str(e)}")
    finally:
        processor.cleanup()


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        if not request.message.strip():
            raise HTTPException(
                status_code=400, detail="Message cannot be empty")

        response = await generate_chat_response(request.message.strip())
        return ChatResponse(success=True, response=response)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Chat error: {str(e)}")
        raise HTTPException(
            status_code=500, detail="Failed to generate response")


@app.post("/api/transcribe", response_model=TranscribeResponse)
async def transcribe_audio_endpoint(audio_file: UploadFile = File(...)):
    try:
        if not audio_file.filename or not any(audio_file.filename.lower().endswith(ext) for ext in ['.webm', '.mp3', '.wav', '.m4a', '.ogg']):
            raise HTTPException(
                status_code=400, detail="Unsupported audio format")

        with tempfile.NamedTemporaryFile(delete=False, suffix='.webm') as tmp_file:
            tmp_file.write(await audio_file.read())
            tmp_file_path = tmp_file.name

        try:
            transcript = await transcribe_audio(tmp_file_path)
            return TranscribeResponse(success=True, text=transcript)
        finally:
            if os.path.exists(tmp_file_path):
                os.unlink(tmp_file_path)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Transcription error: {str(e)}")
        raise HTTPException(
            status_code=500, detail="Failed to transcribe audio")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(
        os.getenv("PORT", 8000)), log_level="info")
