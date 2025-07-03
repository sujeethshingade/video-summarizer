"""
Video to Text Summarization API
A complete FastAPI server that processes video files to generate comprehensive summaries.
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import os
import tempfile
import subprocess
import base64
import asyncio
from pathlib import Path
from typing import List, Optional
import aiofiles
import shutil
import logging
from dotenv import load_dotenv
from pydantic import BaseModel

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration


class Config:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    ALLOWED_ORIGINS = os.getenv(
        "ALLOWED_ORIGINS", "http://localhost:3000").split(",")
    MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB
    SUPPORTED_FORMATS = {".mp4", ".mov",
                         ".avi", ".mkv", ".wmv", ".flv", ".webm"}
    KEYFRAME_INTERVAL = 5  # seconds

    @classmethod
    def validate(cls):
        if not cls.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY environment variable is required")


# Validate configuration
Config.validate()

# Initialize FastAPI app
app = FastAPI(title="Video to Text Summarization API", version="1.0.0")

# Initialize OpenAI client with error handling
try:
    from openai import OpenAI
    client = OpenAI(api_key=Config.OPENAI_API_KEY)
except Exception as e:
    logger.error(f"Failed to initialize OpenAI client: {e}")
    # Try with older OpenAI client initialization
    try:
        import openai
        openai.api_key = Config.OPENAI_API_KEY
        client = None  # We'll use the module-level functions
        logger.info("Using legacy OpenAI client")
    except Exception as e2:
        logger.error(f"Failed to initialize legacy OpenAI client: {e2}")
        raise RuntimeError("Could not initialize OpenAI client")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=Config.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Helper functions for OpenAI API compatibility


async def transcribe_audio_file(audio_path: str) -> str:
    """Transcribe audio using OpenAI Whisper API with compatibility handling"""
    try:
        if client:  # New OpenAI client
            with open(audio_path, "rb") as audio_file:
                transcript = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    response_format="text"
                )
                return transcript
        else:  # Legacy OpenAI client
            import openai
            with open(audio_path, "rb") as audio_file:
                transcript = openai.Audio.transcribe(
                    model="whisper-1",
                    file=audio_file,
                    response_format="text"
                )
                return transcript["text"]
    except Exception as e:
        logger.error(f"Error transcribing audio: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Audio transcription failed: {str(e)}")


async def analyze_image_with_gpt4v(image_path: str, frame_number: int) -> str:
    """Analyze image using GPT-4o Vision with compatibility handling"""
    try:
        with open(image_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode('utf-8')

        if client:  # New OpenAI client
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": f"Describe what's happening in this video frame (Frame {frame_number}). Focus on key visual elements, actions, people, objects, and scene context. Be concise but comprehensive."
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=300
            )
            return response.choices[0].message.content
        else:  # Legacy OpenAI client
            import openai
            response = openai.ChatCompletion.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": f"Describe what's happening in this video frame (Frame {frame_number}). Focus on key visual elements, actions, people, objects, and scene context. Be concise but comprehensive."
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=300
            )
            return response.choices[0].message.content
    except Exception as e:
        logger.error(f"Error analyzing frame {frame_number}: {str(e)}")
        return f"Frame {frame_number} ({(frame_number-1)*Config.KEYFRAME_INTERVAL}s): [Analysis failed]"


async def generate_summary_with_gpt4(transcript: str, visual_descriptions: List[str]) -> str:
    """Generate summary using GPT-4o with compatibility handling"""
    try:
        visual_content = "\n".join(visual_descriptions)

        prompt = f"""
You are an expert video summarizer. Based on the audio transcript and visual scene descriptions below, create a comprehensive, concise summary of the video content.

AUDIO TRANSCRIPT:
{transcript}

VISUAL SCENE DESCRIPTIONS:
{visual_content}

Please provide a well-structured summary that:
1. Captures the main topic and key points from the spoken content
2. Integrates important visual elements and scenes
3. Maintains chronological flow when relevant
4. Is concise but complete (aim for 3-5 paragraphs)
5. Highlights the most significant moments and information

Format the summary in a clear, readable manner with proper paragraphs.
"""

        if client:  # New OpenAI client
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a professional video content summarizer. Create clear, comprehensive summaries that combine both audio and visual information effectively."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=800,
                temperature=0.3
            )
            return response.choices[0].message.content
        else:  # Legacy OpenAI client
            import openai
            response = openai.ChatCompletion.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a professional video content summarizer. Create clear, comprehensive summaries that combine both audio and visual information effectively."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=800,
                temperature=0.3
            )
            return response.choices[0].message.content
    except Exception as e:
        logger.error(f"Error generating summary: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Summary generation failed: {str(e)}")


class VideoProcessor:
    def __init__(self):
        self.temp_dir = tempfile.mkdtemp()

    async def extract_audio(self, video_path: str) -> str:
        """Extract audio from video using ffmpeg"""
        audio_path = os.path.join(self.temp_dir, "audio.wav")

        try:
            # Use ffmpeg to extract audio
            cmd = [
                "ffmpeg", "-i", video_path,
                "-vn", "-acodec", "pcm_s16le",
                "-ar", "44100", "-ac", "2",
                "-y", audio_path
            ]

            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode != 0:
                raise Exception(
                    f"FFmpeg audio extraction failed: {result.stderr}")

            return audio_path
        except Exception as e:
            logger.error(f"Error extracting audio: {str(e)}")
            raise HTTPException(
                status_code=500, detail=f"Audio extraction failed: {str(e)}")

    async def extract_keyframes(self, video_path: str, interval: int = None) -> List[str]:
        """Extract keyframes from video every N seconds"""
        if interval is None:
            interval = Config.KEYFRAME_INTERVAL

        frames_dir = os.path.join(self.temp_dir, "frames")
        os.makedirs(frames_dir, exist_ok=True)

        try:
            # Extract keyframes every N seconds
            cmd = [
                "ffmpeg", "-i", video_path,
                "-vf", f"fps=1/{interval}",
                "-q:v", "2",
                "-y", os.path.join(frames_dir, "frame_%03d.jpg")
            ]

            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode != 0:
                raise Exception(
                    f"FFmpeg keyframe extraction failed: {result.stderr}")

            # Get all extracted frames
            frame_files = sorted([
                os.path.join(frames_dir, f) for f in os.listdir(frames_dir)
                if f.endswith('.jpg')
            ])

            return frame_files
        except Exception as e:
            logger.error(f"Error extracting keyframes: {str(e)}")
            raise HTTPException(
                status_code=500, detail=f"Keyframe extraction failed: {str(e)}")

    async def transcribe_audio(self, audio_path: str) -> str:
        """Transcribe audio using OpenAI Whisper API"""
        return await transcribe_audio_file(audio_path)

    async def analyze_frames(self, frame_paths: List[str]) -> List[str]:
        """Analyze frames using GPT-4o Vision"""
        descriptions = []

        for i, frame_path in enumerate(frame_paths):
            try:
                description = await analyze_image_with_gpt4v(frame_path, i + 1)
                descriptions.append(
                    f"Frame {i+1} ({i*Config.KEYFRAME_INTERVAL}s): {description}")
            except Exception as e:
                logger.error(f"Error analyzing frame {i+1}: {str(e)}")
                descriptions.append(
                    f"Frame {i+1} ({i*Config.KEYFRAME_INTERVAL}s): [Analysis failed]")

        return descriptions

    async def generate_summary(self, transcript: str, visual_descriptions: List[str]) -> str:
        """Generate final summary using GPT-4o"""
        return await generate_summary_with_gpt4(transcript, visual_descriptions)

    def cleanup(self):
        """Clean up temporary files"""
        try:
            shutil.rmtree(self.temp_dir)
        except Exception as e:
            logger.error(f"Error cleaning up temp files: {str(e)}")

# API Routes


@app.post("/api/upload")
async def upload_video(file: UploadFile = File(...)):
    """Upload and process video file"""

    # Validate file format
    file_extension = Path(file.filename).suffix.lower()
    if file_extension not in Config.SUPPORTED_FORMATS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format. Supported formats: {', '.join(Config.SUPPORTED_FORMATS)}"
        )

    # Validate file size
    if file.size > Config.MAX_FILE_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"File too large. Maximum size is {Config.MAX_FILE_SIZE // (1024*1024)}MB."
        )

    processor = VideoProcessor()

    try:
        # Save uploaded file
        video_path = os.path.join(processor.temp_dir, f"video{file_extension}")

        # Write file content
        content = await file.read()
        with open(video_path, 'wb') as f:
            f.write(content)

        logger.info(f"Processing video: {file.filename}")

        # Process video in parallel where possible
        audio_path = await processor.extract_audio(video_path)
        frame_paths = await processor.extract_keyframes(video_path)

        # Transcribe audio and analyze frames
        transcript_task = processor.transcribe_audio(audio_path)
        frames_task = processor.analyze_frames(frame_paths)

        transcript, visual_descriptions = await asyncio.gather(
            transcript_task, frames_task
        )

        # Generate final summary
        summary = await processor.generate_summary(transcript, visual_descriptions)

        logger.info(f"Successfully processed video: {file.filename}")

        return JSONResponse(content={
            "success": True,
            "summary": summary,
            "metadata": {
                "filename": file.filename,
                "transcript_length": len(transcript),
                "frames_analyzed": len(frame_paths),
                "processing_time": "completed"
            }
        })

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error processing video: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Processing failed: {str(e)}")
    finally:
        processor.cleanup()


@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "message": "Video to Text API is running"}


@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "Video to Text Summarization API", "version": "1.0.0"}

# Test API endpoints


@app.get("/api/test")
async def test_openai():
    """Test OpenAI connection"""
    try:
        if client:
            # Test with new client
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "user", "content": "Hello, this is a test."}],
                max_tokens=10
            )
            return {"status": "success", "client": "new", "response": response.choices[0].message.content}
        else:
            # Test with legacy client
            import openai
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "user", "content": "Hello, this is a test."}],
                max_tokens=10
            )
            return {"status": "success", "client": "legacy", "response": response.choices[0].message.content}
    except Exception as e:
        return {"status": "error", "message": str(e)}


# Add request/response models
class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    success: bool
    response: str


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Chat endpoint for AI conversation"""
    try:
        message = request.message.strip()
        if not message:
            raise HTTPException(
                status_code=400, detail="Message cannot be empty")

        # Generate AI response
        response = await generate_chat_response(message)

        return ChatResponse(success=True, response=response)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Chat error: {str(e)}")
        raise HTTPException(
            status_code=500, detail="Failed to generate response")


async def generate_chat_response(message: str) -> str:
    """Generate AI response for chat"""
    try:
        # Use OpenAI to generate response
        if client:  # New OpenAI client
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a helpful AI assistant specialized in video processing, transcription, and general assistance. You are knowledgeable about video formats, audio processing, and can help with various tasks related to multimedia content analysis."},
                    {"role": "user", "content": message}
                ],
                max_tokens=500,
                temperature=0.7
            )
            return response.choices[0].message.content
        else:  # Legacy OpenAI client
            import openai
            response = openai.ChatCompletion.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a helpful AI assistant specialized in video processing, transcription, and general assistance. You are knowledgeable about video formats, audio processing, and can help with various tasks related to multimedia content analysis."},
                    {"role": "user", "content": message}
                ],
                max_tokens=500,
                temperature=0.7
            )
            return response.choices[0].message.content

    except Exception as e:
        logger.error(f"OpenAI API error: {str(e)}")
        # Return fallback response
        return generate_fallback_response(message)


def generate_fallback_response(message: str) -> str:
    """Generate fallback response when OpenAI is unavailable"""
    message_lower = message.lower()

    # Greeting responses
    if any(word in message_lower for word in ['hello', 'hi', 'hey', 'greetings']):
        return "Hello! I'm your AI assistant. I can help you with video transcription, analysis, and answer questions about your content. How can I assist you today?"

    # Video-related responses
    if any(word in message_lower for word in ['video', 'transcrib', 'upload', 'process']):
        return "I can help you with video analysis and transcription! You can upload a video file, and I'll provide you with a comprehensive summary including both the spoken content and visual scenes. What specific aspect of video processing would you like to know more about?"

    # Help responses
    if any(word in message_lower for word in ['help', 'what can you do', 'capabilities']):
        return "I can assist you with several tasks:\n\n• Video-to-text transcription using advanced AI models\n• Visual scene analysis and description\n• Content summarization and insights\n• General conversation and questions\n\nWould you like me to help you with any of these?"

    # Technical questions
    if any(word in message_lower for word in ['how does', 'how do you', 'explain']):
        return "I use advanced AI models for video analysis. The process involves extracting audio for transcription using models like Whisper, and analyzing video frames using computer vision. The combination provides comprehensive insights into both audio and visual content. What specific technical aspect would you like me to explain further?"

    # General responses
    responses = [
        "That's an interesting point! I'm here to help with video processing and analysis, but I can also discuss various topics. How can I assist you further?",
        "I appreciate your message. While I specialize in video-to-text conversion and analysis, I'm happy to help with other questions too. What would you like to explore?",
        "Thanks for sharing that with me. I'm designed to help with video content analysis, but I can also provide general assistance. What specific help do you need?",
        "I understand what you're saying. My main expertise is in video transcription and analysis, but I can discuss various topics. How can I be most helpful to you?",
        "That's a great question! I'm equipped to handle video processing tasks and provide helpful responses to various inquiries. What would you like to know more about?"
    ]

    import random
    return random.choice(responses)


# Audio transcription endpoint for voice messages

class TranscribeResponse(BaseModel):
    success: bool
    text: str


@app.post("/api/transcribe", response_model=TranscribeResponse)
async def transcribe_audio(audio: UploadFile = File(...)):
    """Transcribe audio file using OpenAI Whisper"""
    try:
        # Validate file type
        if not audio.filename or not any(audio.filename.lower().endswith(ext) for ext in ['.webm', '.mp3', '.wav', '.m4a', '.ogg']):
            raise HTTPException(
                status_code=400, detail="Unsupported audio format")

        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.webm') as tmp_file:
            content = await audio.read()
            tmp_file.write(content)
            tmp_file_path = tmp_file.name

        try:
            # Transcribe audio
            transcript = await transcribe_audio_file(tmp_file_path)

            return TranscribeResponse(success=True, text=transcript)

        finally:
            # Clean up temporary file
            if os.path.exists(tmp_file_path):
                os.unlink(tmp_file_path)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Transcription error: {str(e)}")
        raise HTTPException(
            status_code=500, detail="Failed to transcribe audio")


if __name__ == "__main__":
    import uvicorn

    # Check if OpenAI API key is set
    if not Config.OPENAI_API_KEY:
        print("ERROR: OPENAI_API_KEY environment variable is not set!")
        print("Please set your OpenAI API key in the .env file")
        exit(1)

    print("🚀 Starting Video to Text Summarization Server")
    print(f"📁 Supported formats: {', '.join(Config.SUPPORTED_FORMATS)}")
    print(f"📊 Max file size: {Config.MAX_FILE_SIZE // (1024*1024)}MB")
    print(f"🔧 Keyframe interval: {Config.KEYFRAME_INTERVAL}s")
    print(f"🌐 CORS origins: {Config.ALLOWED_ORIGINS}")
    print("💡 Make sure FFmpeg is installed and accessible")
    print("🔗 Server will be available at: http://localhost:8000")
    print("📖 API docs will be available at: http://localhost:8000/docs")
    print("🏥 Health check: http://localhost:8000/api/health")
    print("🧪 Test OpenAI: http://localhost:8000/api/test")

    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
