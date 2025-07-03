'use client';

import { useState, useRef } from 'react';
import { Upload, FileVideo, Loader2, MessageCircle, Bot, User } from 'lucide-react';

interface Message {
  id: string;
  type: 'user' | 'bot';
  content: string;
  timestamp: Date;
  filename?: string;
}

interface ProcessingResponse {
  success: boolean;
  summary: string;
  metadata: {
    filename: string;
    transcript_length: number;
    frames_analyzed: number;
    processing_time: string;
  };
}

export default function Home() {
  const [messages, setMessages] = useState<Message[]>([
    {
      id: '1',
      type: 'bot',
      content: 'Hello! I\'m your video summarization assistant. Upload a video file (.mp4, .mov, .avi, etc.) and I\'ll provide you with a comprehensive summary including both the spoken content and visual scenes.',
      timestamp: new Date(),
    }
  ]);
  const [isProcessing, setIsProcessing] = useState(false);
  const [dragActive, setDragActive] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileUpload = async (file: File) => {
    if (!file) return;

    // Validate file type
    const supportedTypes = ['video/mp4', 'video/quicktime', 'video/x-msvideo', 'video/x-matroska'];
    if (!supportedTypes.includes(file.type)) {
      alert('Please upload a supported video file (.mp4, .mov, .avi, .mkv)');
      return;
    }

    // Validate file size (100MB limit)
    const maxSize = 100 * 1024 * 1024;
    if (file.size > maxSize) {
      alert('File is too large. Maximum size is 100MB.');
      return;
    }

    // Add user message
    const userMessage: Message = {
      id: Date.now().toString(),
      type: 'user',
      content: `Uploaded video: ${file.name}`,
      timestamp: new Date(),
      filename: file.name,
    };

    setMessages(prev => [...prev, userMessage]);
    setIsProcessing(true);

    try {
      const formData = new FormData();
      formData.append('file', file);

      const response = await fetch('http://localhost:8000/api/upload', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || 'Upload failed');
      }

      const data: ProcessingResponse = await response.json();

      // Add bot response
      const botMessage: Message = {
        id: (Date.now() + 1).toString(),
        type: 'bot',
        content: data.summary,
        timestamp: new Date(),
      };

      setMessages(prev => [...prev, botMessage]);

      // Add metadata message
      const metadataMessage: Message = {
        id: (Date.now() + 2).toString(),
        type: 'bot',
        content: `**Processing Complete!**\n\n• Video: ${data.metadata.filename}\n• Transcript Length: ${data.metadata.transcript_length} characters\n• Frames Analyzed: ${data.metadata.frames_analyzed}\n• Status: ${data.metadata.processing_time}`,
        timestamp: new Date(),
      };

      setMessages(prev => [...prev, metadataMessage]);

    } catch (error) {
      console.error('Error uploading file:', error);
      const errorMessage: Message = {
        id: (Date.now() + 1).toString(),
        type: 'bot',
        content: `Sorry, there was an error processing your video: ${error instanceof Error ? error.message : 'Unknown error'}. Please try again.`,
        timestamp: new Date(),
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsProcessing(false);
    }
  };

  const handleDrag = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true);
    } else if (e.type === 'dragleave') {
      setDragActive(false);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);

    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFileUpload(e.dataTransfer.files[0]);
    }
  };

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      handleFileUpload(e.target.files[0]);
    }
  };

  const formatMessage = (content: string) => {
    // Simple markdown-like formatting
    return content
      .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
      .replace(/\*(.*?)\*/g, '<em>$1</em>')
      .replace(/\n/g, '<br />');
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 dark:from-gray-900 dark:to-gray-800">
      <div className="container mx-auto px-4 py-8 max-w-4xl">
        {/* Header */}
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold text-gray-900 dark:text-white mb-2">
            Video to Text Summarizer
          </h1>
          <p className="text-gray-600 dark:text-gray-300">
            Upload a video and get an AI-powered summary of both audio and visual content
          </p>
        </div>

        {/* Chat Interface */}
        <div className="bg-white dark:bg-gray-800 rounded-lg shadow-lg overflow-hidden">
          {/* Messages */}
          <div className="h-96 overflow-y-auto p-6 space-y-4">
            {messages.map((message) => (
              <div
                key={message.id}
                className={`flex ${message.type === 'user' ? 'justify-end' : 'justify-start'}`}
              >
                <div
                  className={`flex items-start space-x-3 max-w-3xl ${message.type === 'user' ? 'flex-row-reverse space-x-reverse' : ''
                    }`}
                >
                  {/* Avatar */}
                  <div
                    className={`w-8 h-8 rounded-full flex items-center justify-center ${message.type === 'user'
                        ? 'bg-blue-500 text-white'
                        : 'bg-gray-200 dark:bg-gray-700 text-gray-600 dark:text-gray-300'
                      }`}
                  >
                    {message.type === 'user' ? <User size={16} /> : <Bot size={16} />}
                  </div>

                  {/* Message */}
                  <div
                    className={`rounded-lg px-4 py-3 ${message.type === 'user'
                        ? 'bg-blue-500 text-white'
                        : 'bg-gray-100 dark:bg-gray-700 text-gray-900 dark:text-gray-100'
                      }`}
                  >
                    {message.filename && (
                      <div className="flex items-center space-x-2 mb-2">
                        <FileVideo size={16} />
                        <span className="text-sm font-medium">{message.filename}</span>
                      </div>
                    )}
                    <div
                      className="text-sm whitespace-pre-wrap"
                      dangerouslySetInnerHTML={{
                        __html: formatMessage(message.content),
                      }}
                    />
                    <div className="text-xs mt-2 opacity-70">
                      {message.timestamp.toLocaleTimeString()}
                    </div>
                  </div>
                </div>
              </div>
            ))}

            {/* Processing indicator */}
            {isProcessing && (
              <div className="flex justify-start">
                <div className="flex items-start space-x-3">
                  <div className="w-8 h-8 rounded-full bg-gray-200 dark:bg-gray-700 flex items-center justify-center">
                    <Bot size={16} className="text-gray-600 dark:text-gray-300" />
                  </div>
                  <div className="bg-gray-100 dark:bg-gray-700 rounded-lg px-4 py-3">
                    <div className="flex items-center space-x-2 text-gray-900 dark:text-gray-100">
                      <Loader2 size={16} className="animate-spin" />
                      <span className="text-sm">Processing your video...</span>
                    </div>
                  </div>
                </div>
              </div>
            )}
          </div>

          {/* Upload Area */}
          <div className="border-t border-gray-200 dark:border-gray-700 p-6">
            <div
              className={`border-2 border-dashed rounded-lg p-8 text-center transition-colors ${dragActive
                  ? 'border-blue-500 bg-blue-50 dark:bg-blue-900/20'
                  : 'border-gray-300 dark:border-gray-600 hover:border-gray-400 dark:hover:border-gray-500'
                }`}
              onDragEnter={handleDrag}
              onDragLeave={handleDrag}
              onDragOver={handleDrag}
              onDrop={handleDrop}
            >
              <Upload className="mx-auto mb-4 text-gray-400" size={48} />
              <p className="text-lg font-medium text-gray-900 dark:text-white mb-2">
                Drop your video here or click to browse
              </p>
              <p className="text-sm text-gray-500 dark:text-gray-400 mb-4">
                Supports MP4, MOV, AVI, MKV • Max file size: 100MB
              </p>
              <button
                onClick={() => fileInputRef.current?.click()}
                disabled={isProcessing}
                className="bg-blue-500 text-white px-6 py-3 rounded-lg font-medium hover:bg-blue-600 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
              >
                {isProcessing ? 'Processing...' : 'Select Video File'}
              </button>
              <input
                ref={fileInputRef}
                type="file"
                accept="video/*"
                onChange={handleFileSelect}
                className="hidden"
                disabled={isProcessing}
              />
            </div>
          </div>
        </div>

        {/* Footer */}
        <div className="text-center mt-8 text-sm text-gray-500 dark:text-gray-400">
          <p>
            Powered by OpenAI Whisper and GPT-4o Vision • Built with Next.js and FastAPI
          </p>
        </div>
      </div>
    </div>
  );
}
