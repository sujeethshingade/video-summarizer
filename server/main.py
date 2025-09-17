from fastapi import FastAPI, File, UploadFile, HTTPException, Form, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
from pydantic import BaseModel
from pathlib import Path
from typing import List, Optional, Tuple, Dict
from openai import OpenAI
from motor.motor_asyncio import AsyncIOMotorClient
from datetime import datetime, timezone
import os
import uuid
import tempfile
import subprocess
import base64
import asyncio
import shutil
import logging
import uvicorn
import json

load_dotenv()

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

START_TIME = datetime.now(timezone.utc)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable is required")

MONGODB_URL = os.getenv("MONGODB_URL")
if not MONGODB_URL:
    raise ValueError("MONGODB_URL environment variable is required")

MONGODB_DB_NAME = os.getenv("MONGODB_DB_NAME")

ALLOWED_ORIGINS = os.getenv(
    "ALLOWED_ORIGINS", "http://localhost:3000").split(",")
MAX_FILE_SIZE = 1024 * 1024 * 1024  # 1GB
SUPPORTED_FORMATS = {".mp4", ".mov", ".avi", ".mkv", ".wmv", ".flv", ".webm"}
DEFAULT_KEYFRAME_INTERVAL = 30  # seconds (used as fallback)

app = FastAPI(title="Video Summarizer", version="1.0.0")

mongodb_client = AsyncIOMotorClient(MONGODB_URL)
db = mongodb_client[MONGODB_DB_NAME]
summaries_collection = db.summaries

jobs: Dict[str, Dict] = {}
job_queue: asyncio.Queue = asyncio.Queue()
queue_worker_started = False

try:
    client = OpenAI(api_key=OPENAI_API_KEY)
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


class TranscribeResponse(BaseModel):
    success: bool
    text: str


class SummaryDocument(BaseModel):
    filename: str
    summary_data: dict
    processed_at: datetime
    metadata: Optional[dict] = None


async def save_summary_to_db(filename: str, summary_json: str, prompt_used: str, metadata: dict = None) -> str:
    try:
        summary_data = json.loads(summary_json)
        
        document = {
            "filename": filename,
            "summary_data": summary_data,
            "prompt_used": prompt_used,
            "processed_at": datetime.now(timezone.utc),
            "metadata": metadata or {}
        }
        
        result = await summaries_collection.insert_one(document)
        logger.info(f"Saved summary to database with ID: {result.inserted_id}")
        return str(result.inserted_id)
        
    except Exception as e:
        logger.error(f"Error saving summary to database: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Database save failed: {str(e)}")


async def get_all_summaries() -> List[dict]:
    try:
        cursor = summaries_collection.find().sort("processed_at", -1)
        summaries = []
        async for document in cursor:
            document["_id"] = str(document["_id"])
            summaries.append(document)
        return summaries
    except Exception as e:
        logger.error(f"Error retrieving summaries from database: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Database retrieval failed: {str(e)}")


class VideoProcessor:
    def __init__(self):
        self.temp_dir = tempfile.mkdtemp()

    async def get_video_duration(self, video_path: str) -> float:
        try:
            cmd = ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
                   "-of", "default=noprint_wrappers=1:nokey=1", video_path]
            result = subprocess.run(
                cmd, capture_output=True, text=True, check=True)
            duration = float(result.stdout.strip())
            return duration
        except (subprocess.CalledProcessError, ValueError) as e:
            logger.warning(
                f"Could not get video duration for {os.path.basename(video_path)}: {e}")
            return 0.0

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

        audio_path = os.path.join(self.temp_dir, "audio.wav")
        cmd = ["ffmpeg", "-i", video_path, "-vn", "-acodec",
               "pcm_s16le", "-ar", "16000", "-ac", "1", "-y", audio_path]
        await self.run_ffmpeg_command(cmd, "Audio extraction failed")
        return audio_path

    async def extract_keyframes(self, video_path: str, keyframe_interval: int) -> List[str]:
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

        cmd = [
            "ffmpeg", "-i", video_path,
            "-vf", f"fps=1/{keyframe_interval}",
            "-q:v", "3", "-y",
            os.path.join(frames_dir, "frame_%03d.jpg")
        ]

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

    async def process_video(self, video_path: str, filename: str, custom_prompt: Optional[str] = None) -> Tuple[str, Optional[str], List[str], int, float, int, str]:
        video_duration = await self.get_video_duration(video_path)

        # Choose dynamic keyframe interval based on duration
        keyframe_interval = choose_keyframe_interval(video_duration)

        audio_path, frame_paths = await asyncio.gather(
            self.extract_audio(video_path),
            self.extract_keyframes(video_path, keyframe_interval)
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
            visual_descriptions = await self.analyze_frames(frame_paths, keyframe_interval)
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

        summary, prompt_used = await generate_summary(
            transcript, visual_descriptions, filename, video_duration, custom_prompt
        )
        return summary, transcript, visual_descriptions, len(frame_paths), video_duration, keyframe_interval, prompt_used

    async def analyze_frames(self, frame_paths: List[str], keyframe_interval: int = DEFAULT_KEYFRAME_INTERVAL) -> List[str]:
        if not frame_paths:
            return []

        semaphore = asyncio.Semaphore(60)

        async def analyze_frame(frame_path: str, index: int):
            async with semaphore:
                timestamp = index * keyframe_interval
                minutes = timestamp // 60
                seconds = timestamp % 60
                timestamp_str = f"{int(minutes):02d}:{int(seconds):02d}"
                return await analyze_images(frame_path, timestamp, timestamp_str)

        tasks = [analyze_frame(fp, i) for i, fp in enumerate(frame_paths)]
        descriptions = await asyncio.gather(*tasks, return_exceptions=True)

        normalized = []
        for i, desc in enumerate(descriptions):
            if isinstance(desc, str):
                normalized.append(desc)
            else:
                normalized.append(f"[{i*keyframe_interval}s]: Analysis failed")
        return normalized

    def cleanup(self):
        try:
            shutil.rmtree(self.temp_dir)
        except Exception as e:
            logger.error(f"Error cleaning up temp files: {str(e)}")


async def call_openai_api(model: str, messages: list) -> str:
    try:
        response = client.chat.completions.create(
            model=model, messages=messages
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


async def analyze_images(image_path: str, timestamp: int, timestamp_str: str = None) -> str:
    try:
        with open(image_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode('utf-8')

        if timestamp_str is None:
            minutes = timestamp // 60
            seconds = timestamp % 60
            timestamp_str = f"{int(minutes):02d}:{int(seconds):02d}"

        messages = [{
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": f"Analyze this video frame at {timestamp_str} ({timestamp}s). Describe: 1) Main subjects/people and their actions, 2) Key objects and environment, 3) Text/graphics visible, 4) Overall scene context. Be specific and concise about what tasks or activities are being performed."
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                }
            ]
        }]

        result = await call_openai_api("gpt-5-mini", messages)
        return f"[{timestamp_str}]: {result}"
    except Exception as e:
        logger.error(f"Error analyzing frame at {timestamp_str}: {str(e)}")
        return f"[{timestamp_str}]: Analysis failed"


async def generate_summary(
    transcript: str,
    visual_descriptions: List[str],
    filename: str,
    video_duration: float = 0.0,
    custom_prompt: Optional[str] = None
) -> Tuple[str, str]:
    try:
        has_audio = transcript is not None and str(transcript).strip()
        has_visual = bool(visual_descriptions)

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

        content_description = (
            "video content with both audio and visual elements" if (has_audio and has_visual)
            else ("audio content from this video file" if has_audio else "visual content from this video file")
        )

        duration_minutes = int(video_duration // 60)
        duration_seconds = int(video_duration % 60)
        duration_str = f"{duration_minutes}m {duration_seconds}s" if duration_minutes > 0 else f"{duration_seconds}s"

        default_instruction = f"""You are an AI assistant analyzing employee work session transcripts. Analyze this {content_description} and your goal is to:
1. Summarize the key tasks the employee performed.
2. Categorize each task into one of the following: 
   - Repetitive
   - Analytical
   - Communication
   - Decision-Making
   - Knowledge Work
3. Identify tools and systems used.
4. Estimate time spent for each task based on the video duration and content analysis.
5. Suggest whether the task has High, Medium, or Low potential for AI support.

VIDEO FILE: {filename}
VIDEO DURATION: {duration_str} (total duration)
{audio_section}{visual_section}

IMPORTANT: When estimating time for tasks, consider that:
- The total video duration is {duration_str}
- Multiple tasks can occur simultaneously or sequentially within this timeframe
- Individual task durations should be realistic relative to the total video length

Provide output in valid JSON format with the following structure:
{{
  "summary": "Brief detailed summary of the work session within 5 lines.",
  "tasks": [
    {{
      "task": "Description of the task",
      "category": "One of: Repetitive, Analytical, Communication, Decision-Making, Knowledge Work",
      "tools": ["e.g. \"Tool1\", \"Tool2\", \"Tool3\""],
      "timeEstimate": "Estimated duration (e.g., '15s', '30s', '45s') based on video analysis or 'Unknown' if not applicable",
      "aiOpportunity": "High, Medium, or Low"
    }}
  ]
}}

Return only valid JSON, no additional text or formatting."""

        user_prompt_display = custom_prompt.strip() if (custom_prompt and custom_prompt.strip()) else default_instruction
        composed_prompt_for_model = (
            (custom_prompt.strip() if custom_prompt else "")
            + f"\n\nVIDEO FILE: {filename}\nVIDEO DURATION: {duration_str} (total duration)\n"
            + audio_section
            + visual_section
        ) if (custom_prompt and custom_prompt.strip()) else default_instruction

        messages = [
            {
                "role": "system",
                "content": "You are an AI assistant analyzing employee work session transcripts. Your summaries help busy professionals quickly understand video content without watching. Adapt your analysis based on the available content - whether it's audio-only, visual-only, or combined content. Always return valid JSON format as requested. When estimating task durations, be realistic based on the total video duration and the context of the tasks observed. Use timestamps from visual analysis and audio cues to determine how long each task actually took."
            },
            {"role": "user", "content": composed_prompt_for_model}
        ]

        response_text = await call_openai_api("gpt-5-mini", messages)

        try:
            json.loads(response_text)
            return response_text, user_prompt_display
        except json.JSONDecodeError:
            logger.warning("OpenAI response was not valid JSON, creating fallback structure")
            fallback_response = {
                "summary": response_text,
                "tasks": [{
                    "task": "Unable to parse specific tasks from response",
                    "category": "Knowledge Work",
                    "tools": ["Unknown"],
                    "timeEstimate": duration_str if video_duration > 0 else "Unknown",
                    "aiOpportunity": "Medium"
                }]
            }
            return json.dumps(fallback_response, indent=2), user_prompt_display
    except Exception as e:
        logger.error(f"Error generating summary: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Summary generation failed: {str(e)}")


def choose_keyframe_interval(video_duration: float) -> int:
    try:
        if video_duration < 3 * 60:  # Less than 3 minutes
            return 10
        elif video_duration < 10 * 60:  # Less than 10 minutes
            return 20
        else:  # 10 minutes or more
            return 30
    except Exception:
        return DEFAULT_KEYFRAME_INTERVAL


@app.get("/")
async def root():
    return {
        "message": "Video Summarizer",
        "status": "healthy"
    }


@app.get("/health")
async def health():
    now = datetime.now(timezone.utc)
    uptime_seconds = int((now - START_TIME).total_seconds())
    openai_ok = True if client else False
    mongo_ok = False
    try:
        await mongodb_client.admin.command('ping')
        mongo_ok = True
    except Exception:
        mongo_ok = False

    return {
        "status": "ok" if (openai_ok and mongo_ok) else "degraded",
        "uptime_seconds": uptime_seconds,
        "openai": openai_ok,
        "mongodb": mongo_ok,
        "timestamp": now.isoformat()
    }


@app.get("/api/summaries")
async def get_summaries():
    try:
        summaries = await get_all_summaries()
        return {"success": True, "summaries": summaries}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in get_summaries endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve summaries")


@app.post("/api/upload")
async def upload_video(
    file: UploadFile = File(...),
    prompt: Optional[str] = Form(None)
):
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

        summary, transcript, visual_descriptions, frames_count, video_duration, keyframe_interval, prompt_used = await processor.process_video(
            video_path, file.filename, prompt
        )

        logger.info(f"Successfully processed video: {file.filename}")

        metadata = {
            "filename": file.filename,
            "transcript_length": len(transcript) if transcript else 0,
            "has_audio": transcript is not None,
            "has_visual": frames_count > 0,
            "frames_analyzed": frames_count,
            "keyframe_interval": keyframe_interval,
            "video_duration_seconds": video_duration,
            "used_custom_prompt": bool(prompt and str(prompt).strip())
        }

        document_id = await save_summary_to_db(file.filename, summary, prompt_used, metadata)

        response_data = {
            "success": True,
            "summary": summary,
            "document_id": document_id,
            "metadata": metadata,
            "prompt": prompt_used
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


async def process_job(job_id: str):
    job = jobs.get(job_id)
    if not job:
        return
    if job.get("status") != "queued":
        return
    job["status"] = "processing"
    file_obj: UploadFile = job["file"]
    prompt = job.get("prompt")
    processor = VideoProcessor()
    try:
        file_extension = Path(file_obj.filename).suffix.lower()
        video_path = os.path.join(processor.temp_dir, f"video{file_extension}")
        with open(video_path, 'wb') as f:
            f.write(job["file_bytes"])  

        summary, transcript, visual_descriptions, frames_count, video_duration, keyframe_interval, prompt_used = await processor.process_video(
            video_path, file_obj.filename, prompt
        )

        metadata = {
            "filename": file_obj.filename,
            "transcript_length": len(transcript) if transcript else 0,
            "has_audio": transcript is not None,
            "has_visual": frames_count > 0,
            "frames_analyzed": frames_count,
            "keyframe_interval": keyframe_interval,
            "video_duration_seconds": video_duration,
            "used_custom_prompt": bool(prompt and str(prompt).strip())
        }

        document_id = await save_summary_to_db(file_obj.filename, summary, prompt_used, metadata)

        job.update({
            "status": "completed",
            "summary": summary,
            "metadata": metadata,
            "prompt_used": prompt_used,
            "document_id": document_id
        })
    except Exception as e:
        job.update({
            "status": "error",
            "error": str(e)
        })
    finally:
        processor.cleanup()


async def queue_worker():
    global queue_worker_started
    if queue_worker_started:
        return
    queue_worker_started = True
    while True:
        job_id = await job_queue.get()
        try:
            await process_job(job_id)
        except Exception as e:
            if job_id in jobs:
                jobs[job_id].update({"status": "error", "error": str(e)})
        finally:
            job_queue.task_done()


@app.post("/api/upload-multiple")
async def upload_multiple_videos(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
    prompt: Optional[str] = Form(None)
):
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")

    created_jobs = []
    for f in files:
        if not f.filename:
            continue
        ext = Path(f.filename).suffix.lower()
        if ext not in SUPPORTED_FORMATS:
            continue  
        file_bytes = await f.read()
        if len(file_bytes) > MAX_FILE_SIZE:
            continue
        job_id = str(uuid.uuid4())
        jobs[job_id] = {
            "id": job_id,
            "filename": f.filename,
            "status": "queued",
            "prompt": prompt,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "file": f,
            "file_bytes": file_bytes
        }
        await job_queue.put(job_id)
        created_jobs.append(job_id)

    background_tasks.add_task(queue_worker)

    if not created_jobs:
        raise HTTPException(status_code=400, detail="No valid files queued for processing")

    return {"success": True, "job_ids": created_jobs}


@app.get("/api/jobs")
async def list_jobs():
    return {"jobs": [
        {k: v for k, v in job.items() if k in {"id", "filename", "status", "error", "document_id"}}
        for job in jobs.values()
    ]}


@app.get("/api/job/{job_id}")
async def get_job(job_id: str):
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    sanitized = {k: v for k, v in job.items() if k not in {"file", "file_bytes"}}
    return sanitized


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
