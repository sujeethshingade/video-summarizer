# Deployment Guide

## Local Development

### Prerequisites
- Python 3.8+
- Node.js 18+
- FFmpeg
- OpenAI API Key

### Quick Setup

1. **Clone and Setup**
   ```bash
   cd video-to-text
   .\setup.bat  # Windows
   # or
   ./setup.sh   # Linux/Mac
   ```

2. **Configure Environment**
   ```bash
   # Edit server/.env
   OPENAI_API_KEY=your_api_key_here
   ```

3. **Start Application**
   ```bash
   .\start.bat  # Windows
   # or
   ./start.sh   # Linux/Mac
   ```

## Docker Deployment

### Prerequisites
- Docker
- Docker Compose

### Setup

1. **Environment Variables**
   ```bash
   # Create .env file in root directory
   OPENAI_API_KEY=your_api_key_here
   ```

2. **Build and Run**
   ```bash
   docker-compose up --build
   ```

3. **Access Application**
   - Frontend: http://localhost:3000
   - Backend: http://localhost:8000

## Production Deployment

### Using Docker

1. **Build for Production**
   ```bash
   docker-compose -f docker-compose.prod.yml up --build -d
   ```

2. **Configure Reverse Proxy**
   ```nginx
   server {
       listen 80;
       server_name your-domain.com;
       
       location / {
           proxy_pass http://localhost:3000;
           proxy_set_header Host $host;
           proxy_set_header X-Real-IP $remote_addr;
       }
       
       location /api/ {
           proxy_pass http://localhost:8000;
           proxy_set_header Host $host;
           proxy_set_header X-Real-IP $remote_addr;
       }
   }
   ```

### Using Cloud Platforms

#### Vercel (Frontend)
1. Connect your GitHub repository
2. Set environment variables:
   - `NEXT_PUBLIC_API_URL=https://your-backend-url.com`
3. Deploy

#### Railway/Heroku (Backend)
1. Connect your GitHub repository
2. Set environment variables:
   - `OPENAI_API_KEY=your_api_key`
   - `PORT=8000`
3. Add buildpack for FFmpeg
4. Deploy

### Environment Variables

#### Backend (.env)
```env
OPENAI_API_KEY=your_openai_api_key
ALLOWED_ORIGINS=https://your-frontend-domain.com
PORT=8000
DEBUG=false
```

#### Frontend (.env.local)
```env
NEXT_PUBLIC_API_URL=https://your-backend-domain.com
```

## Monitoring and Maintenance

### Health Checks
- Backend: `GET /api/health`
- Test script: `python server/test_api.py`

### Logs
- Backend: Check server logs for processing errors
- Frontend: Check browser console for client errors

### Performance Optimization
- Use CDN for static assets
- Implement caching for API responses
- Monitor OpenAI API usage and costs
- Set up error tracking (Sentry, etc.)

## Security Considerations

1. **API Key Security**
   - Never commit API keys to version control
   - Use environment variables
   - Rotate keys regularly

2. **File Upload Security**
   - Validate file types and sizes
   - Scan for malware
   - Implement rate limiting

3. **CORS Configuration**
   - Restrict origins in production
   - Use HTTPS in production

## Troubleshooting

### Common Issues

1. **FFmpeg not found**
   - Install FFmpeg
   - Add to system PATH
   - Use Docker image with FFmpeg

2. **OpenAI API errors**
   - Check API key validity
   - Verify account credits
   - Monitor rate limits

3. **File upload failures**
   - Check file size limits
   - Verify supported formats
   - Check server disk space

4. **CORS errors**
   - Update ALLOWED_ORIGINS
   - Check protocol (http/https)
   - Verify domain spelling

### Performance Issues

1. **Large file processing**
   - Implement streaming upload
   - Add progress indicators
   - Use background job queues

2. **High API costs**
   - Cache results
   - Implement usage limits
   - Optimize prompts

## Scaling

### Horizontal Scaling
- Use load balancers
- Implement session management
- Use distributed file storage

### Vertical Scaling
- Increase server resources
- Optimize video processing
- Use faster storage

### Background Processing
- Implement job queues (Celery, RQ)
- Use message brokers (Redis, RabbitMQ)
- Add progress tracking
