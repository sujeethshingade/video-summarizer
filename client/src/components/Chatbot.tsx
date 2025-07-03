"use client"

import { useState, useRef, useEffect } from "react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Mic, MicOff, Send, Paperclip, FileVideo, Loader2 } from "lucide-react"
import { cn } from "@/lib/utils"

interface Message {
    id: string
    text: string
    isUser: boolean
    isProcessing?: boolean
    filename?: string
}

export function Chatbot() {
    const [messages, setMessages] = useState<Message[]>([])
    const [isRecording, setIsRecording] = useState(false)
    const [isTranscribing, setIsTranscribing] = useState(false)
    const [isProcessingVideo, setIsProcessingVideo] = useState(false)
    const [message, setMessage] = useState("")
    const messagesEndRef = useRef<HTMLDivElement>(null)
    const mediaRecorderRef = useRef<MediaRecorder | null>(null)
    const chunksRef = useRef<Blob[]>([])
    const fileInputRef = useRef<HTMLInputElement>(null)

    useEffect(() => {
        messagesEndRef.current?.scrollIntoView({ behavior: "smooth" })
    }, [messages])

    const addMessage = (text: string, isUser: boolean, filename?: string, isProcessing?: boolean) => {
        setMessages(prev => [...prev, {
            id: Date.now().toString() + isUser,
            text,
            isUser,
            filename,
            isProcessing
        }])
    }

    const updateMessage = (id: string, updates: Partial<Message>) => {
        setMessages(prev => prev.map(msg =>
            msg.id === id ? { ...msg, ...updates } : msg
        ))
    }

    const handleFileUpload = async (file: File) => {
        if (!file) return

        // Validate file type
        const supportedTypes = ['video/mp4', 'video/quicktime', 'video/x-msvideo', 'video/x-matroska', 'video/webm']
        if (!supportedTypes.includes(file.type)) {
            addMessage('Please upload a supported video file (.mp4, .mov, .avi, .mkv, .webm)', false)
            return
        }

        // Validate file size (100MB limit)
        const maxSize = 100 * 1024 * 1024
        if (file.size > maxSize) {
            addMessage('File is too large. Maximum size is 100MB.', false)
            return
        }

        // Add user message showing file upload
        addMessage(`Uploaded video: ${file.name}`, true, file.name)

        // Add processing message
        const processingMessageId = Date.now().toString() + '_processing'
        addMessage('Processing your video...', false, undefined, true)
        setIsProcessingVideo(true)

        try {
            const formData = new FormData()
            formData.append('file', file)

            const response = await fetch('http://localhost:8000/api/upload', {
                method: 'POST',
                body: formData,
            })

            if (!response.ok) {
                const error = await response.json()
                throw new Error(error.detail || 'Upload failed')
            }

            const data = await response.json()

            // Remove processing message and add results
            setMessages(prev => prev.filter(msg => !msg.isProcessing))

            // Add summary message
            addMessage(data.summary, false)

            // Add metadata message if available
            if (data.metadata) {
                const metadataText = `**Processing Complete!**\n\n• Video: ${data.metadata.filename}\n• Transcript Length: ${data.metadata.transcript_length} characters\n• Frames Analyzed: ${data.metadata.frames_analyzed}\n• Processing Time: ${data.metadata.processing_time}`
                addMessage(metadataText, false)
            }

        } catch (error) {
            console.error('Error uploading file:', error)
            // Remove processing message
            setMessages(prev => prev.filter(msg => !msg.isProcessing))
            addMessage(`Sorry, there was an error processing your video: ${error instanceof Error ? error.message : 'Unknown error'}. Please try again.`, false)
        } finally {
            setIsProcessingVideo(false)
        }
    }

    const handleSendMessage = async (text: string) => {
        addMessage(text, true)

        try {
            // Call FastAPI backend for AI response
            const response = await fetch('http://localhost:8000/api/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    message: text
                })
            })

            if (!response.ok) {
                throw new Error(`HTTP ${response.status}`)
            }

            const data = await response.json()

            if (data.success && data.response) {
                addMessage(data.response, false)
            } else {
                // Fallback response
                addMessage(generateFallbackResponse(text), false)
            }
        } catch (error) {
            console.error('Chat error:', error)
            // Fallback response on error
            addMessage(generateFallbackResponse(text), false)
        }
    }

    const generateFallbackResponse = (userMessage: string) => {
        const responses = [
            "I understand you're asking about that. As an AI assistant, I can help you with various tasks including video transcription and analysis.",
            "That's an interesting question! I'm here to help with video processing, text analysis, and general assistance.",
            "I appreciate your message. While I'm designed to help with video-to-text conversion, I can also assist with other queries.",
            "Thanks for reaching out! I'm equipped to handle video analysis, transcription, and provide helpful responses to your questions.",
            "I see what you're asking about. I'm an AI assistant focused on helping with video content analysis and general support."
        ]

        // Simple keyword-based responses
        const lowerMessage = userMessage.toLowerCase()

        if (lowerMessage.includes('video') || lowerMessage.includes('transcrib') || lowerMessage.includes('upload')) {
            return "I can help you with video analysis and transcription! You can ask me specific questions about video processing, or use the file upload button to upload a video file for analysis."
        }

        if (lowerMessage.includes('hello') || lowerMessage.includes('hi') || lowerMessage.includes('hey')) {
            return "Hello! I'm your AI assistant. I can help you with video transcription, analysis, and answer questions about your content. How can I assist you today?"
        }

        if (lowerMessage.includes('help') || lowerMessage.includes('what') || lowerMessage.includes('how')) {
            return "I'm here to help! I can assist with video-to-text conversion, content analysis, and general questions. Ask me anything you'd like to know!"
        }

        // Return a random response
        return responses[Math.floor(Math.random() * responses.length)]
    }

    const transcribeAudio = async (audioBlob: Blob) => {
        setIsTranscribing(true)
        try {
            const formData = new FormData()
            formData.append("audio", audioBlob, "recording.webm")

            const response = await fetch('http://localhost:8000/api/transcribe', {
                method: "POST",
                body: formData
            })

            if (!response.ok) throw new Error(`HTTP ${response.status}`)
            const data = await response.json()
            if (!data.success) throw new Error(data.error || "Transcription failed")
            return data.text?.trim() || ""
        } catch (error) {
            console.error("Transcription error:", error)
            return ""
        } finally {
            setIsTranscribing(false)
        }
    }

    const sendTextMessage = () => {
        const trimmed = message.trim()
        if (trimmed && !isTranscribing) {
            handleSendMessage(trimmed)
            setMessage("")
        }
    }

    const toggleRecording = async () => {
        if (isTranscribing) return

        if (isRecording) {
            mediaRecorderRef.current?.stop()
            setIsRecording(false)
            return
        }

        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
            const mediaRecorder = new MediaRecorder(stream)
            mediaRecorderRef.current = mediaRecorder
            chunksRef.current = []

            mediaRecorder.ondataavailable = (event) => {
                if (event.data.size > 0) chunksRef.current.push(event.data)
            }

            mediaRecorder.onstop = async () => {
                const audioBlob = new Blob(chunksRef.current, { type: "audio/webm" })
                const text = await transcribeAudio(audioBlob)
                if (text) {
                    setMessage(prev => prev ? `${prev} ${text}` : text)
                }
                stream.getTracks().forEach(track => track.stop())
            }

            mediaRecorder.start()
            setIsRecording(true)
        } catch (error) {
            console.error("Recording error:", error)
            alert("Could not access microphone. Please check permissions.")
        }
    }

    const placeholder = isRecording ? "Recording... Click on mic to stop" :
        isTranscribing ? "Processing audio..." :
            "Ask anything..."

    return (
        <div className="flex flex-col h-full bg-background">
            <div className="flex-1 overflow-y-auto p-4 space-y-4">
                {messages.map((msg) => (
                    <div key={msg.id} className={cn("flex w-full", msg.isUser ? "justify-end" : "justify-start")}>
                        <div className={cn(
                            "max-w-[80%] rounded-lg px-4 py-3 text-sm shadow-sm",
                            msg.isUser ? "bg-blue-500 text-white" : "bg-gray-100 text-gray-900 border border-gray-200"
                        )}>
                            {msg.filename && (
                                <div className="flex items-center space-x-2 mb-2">
                                    <FileVideo className="h-4 w-4" />
                                    <span className="text-xs font-medium">{msg.filename}</span>
                                </div>
                            )}
                            {msg.isProcessing && (
                                <div className="flex items-center space-x-2 mb-2">
                                    <Loader2 className="h-4 w-4 animate-spin" />
                                    <span className="text-xs">Processing...</span>
                                </div>
                            )}
                            <p className="whitespace-pre-wrap leading-relaxed">{msg.text}</p>
                        </div>
                    </div>
                ))}
                <div ref={messagesEndRef} />
            </div>

            <div className="flex-shrink-0 relative flex items-center gap-2 p-4 border-t bg-background">
                <Input
                    value={message}
                    onChange={(e: React.ChangeEvent<HTMLInputElement>) => setMessage(e.target.value)}
                    onKeyDown={(e: React.KeyboardEvent<HTMLInputElement>) => e.key === "Enter" && !e.shiftKey && (e.preventDefault(), sendTextMessage())}
                    placeholder={placeholder}
                    disabled={isTranscribing}
                    className="flex-1"
                />

                <Button
                    size="icon"
                    variant="outline"
                    onClick={() => fileInputRef.current?.click()}
                    disabled={isTranscribing || isProcessingVideo}
                    className="transition-all duration-200"
                >
                    <Paperclip className="h-4 w-4" />
                </Button>

                <Button
                    size="icon"
                    variant="outline"
                    onClick={toggleRecording}
                    disabled={isTranscribing}
                    className={cn(
                        "transition-all duration-200 text-gray-900",
                        isRecording && "bg-red-500 text-white hover:bg-red-600 hover:text-white border-red-500 animate-pulse",
                        isTranscribing && "opacity-50"
                    )}
                >
                    {isRecording ? <MicOff className="h-4 w-4" /> : <Mic className="h-4 w-4" />}
                </Button>

                <Button
                    size="icon"
                    onClick={sendTextMessage}
                    disabled={!message.trim() || isTranscribing}
                >
                    <Send className="h-4 w-4" />
                </Button>

                <input
                    ref={fileInputRef}
                    type="file"
                    accept="video/*"
                    onChange={(e) => e.target.files?.[0] && handleFileUpload(e.target.files[0])}
                    className="hidden"
                    disabled={isProcessingVideo}
                />

                {isRecording && (
                    <div className="absolute -top-10 left-1/2 transform -translate-x-1/2 bg-red-500 text-white px-3 py-1 rounded-full text-sm">
                        <div className="flex items-center gap-2">
                            <div className="w-2 h-2 bg-white rounded-full animate-pulse" />
                            Recording...
                        </div>
                    </div>
                )}
            </div>
        </div>
    )
}
