# Video Summarization

An application that processes video files to analyze employee work sessions, providing structured summaries with task categorization, time estimation, and AI automation opportunity assessment.

## Installation

```bash
# Navigate to client directory
cd client

# Install dependencies
npm install

# Start frontend server
npm run dev

# Frontend will start on http://localhost:3000
```

```bash
# Navigate to server directory
cd server

# Create and activate virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install Python dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env

# Start backend server
python main.py

# Server will start on http://localhost:8000
# API documentation available at http://localhost:8000/docs
```

## How It Works?

### Phase 1: Upload & Validation

1. **File Upload**: User uploads video via drag-and-drop or file selection in chat interface
2. **Format Validation**: Server validates file format against supported types
3. **Size Check**: Ensures file size is within configured limits (default: 100MB)
4. **Temporary Storage**: File is stored temporarily for processing

### Phase 2: Media Extraction

```python
# Audio extraction using FFmpeg
ffmpeg -i input_video.mp4 -vn -acodec pcm_s16le -ar 16000 -ac 1 output_audio.wav

# Keyframe extraction (every 5 seconds)
ffmpeg -i input_video.mp4 -vf "fps=1/5" -q:v 3 frame_%03d.jpg
```

### Phase 3: Parallel Processing

#### Audio Transcription with Whisper Models

```python
async def transcribe_audio(audio_path: str) -> str:
    """
    Transcribes audio using OpenAI Whisper Models
    - Supports multiple languages
    - Handles various audio qualities
    - Returns timestamped transcription
    """
    with open(audio_path, "rb") as audio_file:
        transcript = client.audio.transcriptions.create(
            model="gpt-4o-mini-transcribe",
            file=audio_file,
            response_format="text"
        )
    return transcript
```

#### Visual Analysis with GPT-4 Vision

```python
async def analyze_images(image_path: str, timestamp: int, timestamp_str: str = None) -> str:
    """
    Analyzes video frames using GPT-4 Vision
    - Identifies subjects, objects, and scenes
    - Extracts visible text and graphics
    - Focuses on work activities and task identification
    """
    with open(image_path, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode('utf-8')
    
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
    
    return await call_openai_api("gpt-4.1", messages, max_tokens=200)
```

### Phase 4: Video Analysis & Summary Generation

```python
async def generate_summary(transcript: str, visual_descriptions: List[str], filename: str, video_duration: float = 0.0) -> str:

    prompt = f"""You are an AI assistant analyzing employee work session transcripts. Analyze this {content_description} and your goal is to:
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

Provide output in valid JSON format with the following structure:
{{
  "summary": "Brief detailed summary of the work session within 5 lines.",
  "tasks": [
    {{
      "task": "Description of the task",
      "category": "One of: Repetitive, Analytical, Communication, Decision-Making, Knowledge Work",
      "tools": ["Tool1", "Tool2", "Tool3"],
      "timeEstimate": "Estimated duration based on video analysis",
      "aiOpportunity": "High, Medium, or Low"
    }}
  ]
}}"""

    return await call_openai_api("gpt-4.1", messages, max_tokens=800, temperature=0.2)
```

### Configurable Options

- **Max file size**: 1GB (adjustable via MAX_FILE_SIZE)
- **Supported formats**: MP4, MOV, AVI, WMV, FLV, WebM, MKV
- **Keyframe extraction**: Every 5 seconds (configurable via KEYFRAME_INTERVAL)
- **Output format**: Structured JSON with fallback to plain text
- **Audio handling**: Automatically detects and handles videos without audio tracks
- **Visual handling**: Automatically detects and handles videos without visual content
- **Content requirements**: Processes videos with audio-only, visual-only, or combined content
- **Task analysis**: Specialized prompting for work session evaluation

### References

- [OpenAI GPT-4V System Card](https://openai.com/index/gpt-4v-system-card/)
- [OpenAI Images & Vision API Documentation](https://platform.openai.com/docs/guides/images-vision)
