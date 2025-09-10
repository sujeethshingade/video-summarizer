# Video Summarizer

An application that processes video files to analyze employee work sessions and returns summaries and analytics.

## How it works?

1. Upload: user uploads a video via the frontend (drag-and-drop or file select). A `prompt` field may be provided.
2. Processing: server validates the file, extracts audio (if present) and keyframes using `ffmpeg`/`ffprobe`.
3. Keyframe selection: server chooses a keyframe interval based on video duration (e.g. 10s / 20s / 30s) and extracts frames at that rate.
4. Analysis: audio is transcribed (OpenAI transcription models) and frames are analyzed via vision-capable models. A composed internal prompt (including audio/visual context) is sent to the model.
5. Summary database: results are saved to MongoDB along with `prompt_used` and `metadata` to enable the summaries page to display the prompt and other details later.

### Installation

```bash
# Navigate to client
cd client

# Install dependencies
npm install

# Start frontend
npm run dev

# Frontend available at http://localhost:3000
```

```bash
# Navigate to server
cd server

# Create & activate venv (macOS / Linux)
python -m venv venv
source venv/bin/activate

# Install server deps
pip install -r requirements.txt

# Copy environment file and set credentials
cp .env.example .env
# Edit .env to set OPENAI_API_KEY, MONGODB_URL, MONGODB_DB_NAME, etc.

# (Optional) If you need to merge WebM fragments:
python merge_webm.py

# Start backend (the project contains a FastAPI app)
python main.py

# Backend available at http://localhost:8000 (API docs at /docs)
```

```bash
# Build both services
docker compose build

# Run both services
docker compose up -d

# Check logs
docker compose logs -f

# Stop services
docker compose down
```

Note: ensure `ffmpeg` and `ffprobe` are installed. On macOS, you can install via Homebrew:

```bash
brew install ffmpeg
```
