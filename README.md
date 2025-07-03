# Video to Text Summarization Application

A full-stack application that processes video files to generate comprehensive summaries combining both spoken content and visual scene descriptions using AI.

## Features

- **Video Upload**: Support for MP4, MOV, AVI, MKV, and other popular video formats
- **Audio Transcription**: Extract and transcribe audio using OpenAI Whisper API
- **Visual Analysis**: Extract keyframes and analyze visual content using GPT-4o Vision
- **AI Summarization**: Generate comprehensive summaries combining audio and visual content
- **Chat Interface**: Modern chat-like UI for uploading files and viewing results
- **Real-time Processing**: Live updates and loading states during processing
- **Responsive Design**: Mobile-friendly interface with Tailwind CSS

## Tech Stack

### Frontend (Next.js)
- **Framework**: Next.js 15.3.4 with React 19
- **Styling**: Tailwind CSS 4
- **Icons**: Lucide React
- **TypeScript**: Full type safety

### Backend (FastAPI)
- **Framework**: FastAPI with Python 3.8+
- **AI Services**: OpenAI Whisper + GPT-4o Vision
- **Video Processing**: FFmpeg
- **File Handling**: Aiofiles, Python-multipart
- **Environment**: Python-dotenv

## Project Structure

```
video-to-text/
├── client/                 # Next.js frontend
│   ├── src/
│   │   └── app/
│   │       ├── page.tsx    # Main chat interface
│   │       ├── layout.tsx  # App layout
│   │       └── globals.css # Global styles
│   ├── public/            # Static assets
│   ├── package.json
│   └── next.config.ts
├── server/                # FastAPI backend
│   ├── main.py           # Main API server
│   ├── config.py         # Configuration settings
│   ├── run.py            # Server runner
│   ├── requirements.txt  # Python dependencies
│   └── .env             # Environment variables
└── README.md
```

## Setup Instructions

### Prerequisites

- Node.js 18+ and npm
- Python 3.8+
- FFmpeg (for video processing)
- OpenAI API key

### 1. Install FFmpeg

**Windows:**
```bash
# Using chocolatey
choco install ffmpeg

# Or download from https://ffmpeg.org/download.html
```

**macOS:**
```bash
brew install ffmpeg
```

**Linux:**
```bash
sudo apt update
sudo apt install ffmpeg
```

### 2. Backend Setup

```bash
# Navigate to server directory
cd server

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env and add your OpenAI API key:
# OPENAI_API_KEY=your_openai_api_key_here
```

### 3. Frontend Setup

```bash
# Navigate to client directory
cd client

# Install dependencies
npm install

# Or use yarn
yarn install
```

### 4. Running the Application

**Start the Backend:**
```bash
cd server
python run.py
# Server will start on http://localhost:8000
```

**Start the Frontend:**
```bash
cd client
npm run dev
# Frontend will start on http://localhost:3000
```

## Usage

1. **Open the Application**: Navigate to `http://localhost:3000`
2. **Upload Video**: Drag and drop or click to select a video file
3. **Processing**: Watch the real-time processing status
4. **View Summary**: Review the AI-generated summary combining audio and visual content
5. **Metadata**: See processing statistics and details

## API Endpoints

### `POST /api/upload`
Upload and process a video file.

**Request:**
- `file`: Video file (multipart/form-data)

**Response:**
```json
{
  "success": true,
  "summary": "Generated summary text...",
  "metadata": {
    "filename": "video.mp4",
    "transcript_length": 1500,
    "frames_analyzed": 12,
    "processing_time": "completed"
  }
}
```

### `GET /api/health`
Health check endpoint.

### `GET /`
API information.

## Configuration

### Environment Variables

**Server (.env):**
```env
OPENAI_API_KEY=your_openai_api_key_here
ALLOWED_ORIGINS=http://localhost:3000
PORT=8000
DEBUG=false
```

### Configuration Options

- **Max file size**: 100MB (configurable)
- **Supported formats**: MP4, MOV, AVI, MKV, WMV, FLV, WebM
- **Keyframe interval**: 5 seconds (configurable)
- **Max frames analyzed**: 50 (configurable)

## Video Processing Pipeline

1. **Upload Validation**: Check file format and size
2. **Audio Extraction**: Use FFmpeg to extract audio track
3. **Keyframe Extraction**: Extract frames every 5 seconds using FFmpeg
4. **Parallel Processing**:
   - **Audio Transcription**: OpenAI Whisper API
   - **Visual Analysis**: GPT-4o Vision API for each frame
5. **Summary Generation**: GPT-4o combines audio and visual content
6. **Response**: Return comprehensive summary with metadata

## Future Enhancements

- **User Authentication**: Add user accounts and history
- **Follow-up Questions**: Allow users to ask questions about the video
- **Timestamp Navigation**: Click on summary sections to jump to video timestamps
- **Downloadable Reports**: Export summaries as PDF or Word documents
- **Batch Processing**: Process multiple videos simultaneously
- **Advanced Analytics**: Video content categorization and insights
- **Custom Prompts**: Allow users to customize summarization prompts

## Troubleshooting

### Common Issues

1. **FFmpeg not found**: Install FFmpeg and ensure it's in your PATH
2. **OpenAI API errors**: Check your API key and account credits
3. **File upload fails**: Verify file size and format restrictions
4. **CORS errors**: Check ALLOWED_ORIGINS setting in .env

### Error Handling

- File validation for format and size
- Comprehensive error messages
- Graceful fallbacks for processing failures
- Detailed logging for debugging

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License.

## Support

For issues and questions:
- Check the troubleshooting section
- Review error logs
- Open an issue on GitHub

## Acknowledgments

- OpenAI for Whisper and GPT-4o APIs
- FFmpeg for video processing capabilities
- Next.js and FastAPI communities
- Tailwind CSS for styling framework
