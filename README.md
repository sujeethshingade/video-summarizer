# Video to Text Summarization

An application that processes video files to generate comprehensive summaries by combining both spoken content and visual scene.

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

1. **File Upload**: User selects video file through the web interface
2. **Format Validation**: Server validates file format against supported types
3. **Size Check**: Ensures file size is within configured limits (default: 100MB)
4. **Temporary Storage**: File is stored temporarily for processing

### Phase 2: Media Extraction

```python
# Audio extraction using FFmpeg
ffmpeg -i input_video.mp4 -vn -acodec pcm_s16le -ar 44100 -ac 2 output_audio.wav

# Keyframe extraction (every 5 seconds)
ffmpeg -i input_video.mp4 -vf "select='eq(pict_type,PICT_TYPE_I)*not(mod(n,150))" frame_%03d.jpg
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
        response = await openai.Audio.transcribe(
            model="gpt-4o-mini-transcribe",
            file=audio_file,
            response_format="text"
        )
    return response
```

#### Visual Analysis with GPT-4 Vision

```python
async def analyze_images(image_path: str, timestamp: int) -> str:
    """
    Analyzes video frames using GPT-4 Vision
    - Identifies subjects, objects, and scenes
    - Extracts visible text and graphics
    - Provides contextual descriptions
    """
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
```

### Phase 4: Summary Generation

```python
async def generate_summary(transcript: str, visual_descriptions: List[str], filename: str) -> str:
    """
    Combines audio transcript and visual analysis into comprehensive summary
    Uses GPT-4 with specialized prompting for video content analysis
    """
    visual_content = "\n".join(visual_descriptions)
    
    prompt = f"""Analyze this video content and create a comprehensive yet concise summary.

VIDEO FILE: {filename}

AUDIO TRANSCRIPT:
{transcript}

VISUAL ANALYSIS (timestamped):
{visual_content}

Write a detailed summary that clearly explains what happens in the video from beginning to end.
Cover: main topics, key visual elements, important conclusions, and overall purpose.
Requirements: 300-400 words, professional language, specific and factual."""

    messages = [
        {
            "role": "system",
            "content": "You are an expert video content analyst specializing in creating executive-level summaries. Focus on actionable insights, key data points, and business value."
        },
        {"role": "user", "content": prompt}
    ]
    
    return await call_openai_api("gpt-4.1", messages, max_tokens=800, temperature=0.2)
```

### Configurable Options

- **Max file size**: 100MB (adjustable via MAX_FILE_SIZE)
- **Supported formats**: MP4, MOV, AVI, MKV, WMV, FLV, WebM
- **Keyframe extraction**: Every 5 seconds (configurable)
- **Frame analysis limit**: 50 frames maximum (prevents excessive API usage)
- **Summary length**: 300-400 words (optimized for readability)

### References

- [OpenAI GPT-4V System Card](https://openai.com/index/gpt-4v-system-card/)
- [OpenAI Images & Vision API Documentation](https://platform.openai.com/docs/guides/images-vision)
  