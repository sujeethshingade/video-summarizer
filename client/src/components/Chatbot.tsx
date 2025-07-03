"use client"

import { useState, useRef, useEffect } from "react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Avatar, AvatarFallback } from "@/components/ui/avatar"
import { Send, Paperclip, FileVideo, Loader2, Bot, User } from "lucide-react"
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
    const [isProcessingVideo, setIsProcessingVideo] = useState(false)
    const [message, setMessage] = useState("")
    const messagesEndRef = useRef<HTMLDivElement>(null)
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

    const handleFileUpload = async (file: File) => {
        if (!file) return

        const supportedTypes = ['video/mp4', 'video/quicktime', 'video/x-msvideo', 'video/x-matroska', 'video/webm']
        if (!supportedTypes.includes(file.type)) {
            addMessage('Please upload a supported video file (.mp4, .mov, .avi, .mkv, .webm)', false)
            return
        }

        if (file.size > 100 * 1024 * 1024) {
            addMessage('File is too large. Maximum size is 100MB.', false)
            return
        }

        addMessage(`Uploaded video: ${file.name}`, true, file.name)
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
            setMessages(prev => prev.filter(msg => !msg.isProcessing))
            addMessage(data.summary, false)

            if (data.metadata) {
                const metadataText = `**Processing Complete!**\n\n• Video: ${data.metadata.filename}\n• Transcript Length: ${data.metadata.transcript_length} characters\n• Frames Analyzed: ${data.metadata.frames_analyzed}\n• Processing Time: ${data.metadata.processing_time}`
                addMessage(metadataText, false)
            }
        } catch (error) {
            console.error('Error uploading file:', error)
            setMessages(prev => prev.filter(msg => !msg.isProcessing))
            addMessage(`Sorry, there was an error processing your video: ${error instanceof Error ? error.message : 'Unknown error'}.`, false)
        } finally {
            setIsProcessingVideo(false)
        }
    }

    const handleSendMessage = async (text: string) => {
        addMessage(text, true)

        try {
            const response = await fetch('http://localhost:8000/api/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ message: text })
            })

            if (!response.ok) throw new Error(`HTTP ${response.status}`)
            const data = await response.json()

            if (data.success && data.response) {
                addMessage(data.response, false)
            } else {
                addMessage("I'm here to help with video analysis and transcription. Upload a video or ask me a question!", false)
            }
        } catch (error) {
            console.error('Chat error:', error)
            addMessage("I can help you analyze videos and answer questions. How can I assist you today?", false)
        }
    }

    const sendTextMessage = () => {
        const trimmed = message.trim()
        if (trimmed) {
            handleSendMessage(trimmed)
            setMessage("")
        }
    }

    return (
        <div className="flex flex-col h-full bg-gray-200">
            <div className="flex-1 overflow-y-auto">
                <div className="max-w-4xl mx-auto px-4 py-6 space-y-6">
                    {messages.map((msg) => (
                        <div key={msg.id} className={cn("flex", msg.isUser ? "justify-end" : "justify-start")}>
                            {!msg.isUser && (
                                <Avatar className="w-10 h-10 mt-1">
                                    <AvatarFallback className="text-gray-900">
                                        <Bot className="h-6 w-6" />
                                    </AvatarFallback>
                                </Avatar>
                            )}
                            <div className={cn(
                                "flex flex-col max-w-[80%]",
                                msg.isUser ? "items-end" : "items-start"
                            )}>
                                <div className={cn(
                                    "rounded-2xl px-4 py-3 text-sm shadow-sm",
                                    msg.isUser
                                        ? "bg-white border border-gray-400 text-gray-900"
                                        : "bg-white border border-gray-400 text-gray-900"
                                )}>
                                    {msg.filename && (
                                        <div className="flex items-center space-x-2 mb-2 pb-2 border-b border-gray-200">
                                            <FileVideo className="h-5 w-5" />
                                            <span className="text-xs font-medium">{msg.filename}</span>
                                        </div>
                                    )}
                                    {msg.isProcessing && (
                                        <div className="flex items-center space-x-2 mb-2">
                                            <Loader2 className="h-5 w-5 animate-spin" />
                                            <span className="text-xs">Processing...</span>
                                        </div>
                                    )}
                                    <p className="whitespace-pre-wrap leading-relaxed">{msg.text}</p>
                                </div>
                            </div>
                            {msg.isUser && (
                                <Avatar className="w-10 h-10 mt-1">
                                    <AvatarFallback className="text-gray-900">
                                        <User className="h-6 w-6" />
                                    </AvatarFallback>
                                </Avatar>
                            )}
                        </div>
                    ))
                    }
                    <div ref={messagesEndRef} />
                </div>
            </div>

            <div className="flex-shrink-0 bg-gray-200 border-t border-gray-400">
                <div className="max-w-5xl mx-auto px-4 py-4">
                    <div className="relative flex items-center">
                        <Input
                            value={message}
                            onChange={(e) => setMessage(e.target.value)}
                            onKeyDown={(e) => e.key === "Enter" && !e.shiftKey && (e.preventDefault(), sendTextMessage())}
                            className="h-10 border-gray-400 focus:border-gray-900 focus:ring-gray-900 pr-24"
                        />
                        <div className="absolute right-2 top-1/2 -translate-y-1/2 flex items-center gap-2">
                            <Button
                                type="button"
                                variant="ghost"
                                onClick={() => fileInputRef.current?.click()}
                                disabled={isProcessingVideo}
                                className="hover:bg-gray-100 focus:outline-none rounded-full text-gray-900"
                                tabIndex={-1}
                            >
                                <Paperclip className="h-5 w-5" />
                            </Button>
                            <Button
                                type="button"
                                variant="ghost"
                                onClick={sendTextMessage}
                                disabled={!message.trim()}
                                className="hover:bg-gray-100 focus:outline-none rounded-full text-gray-900e"
                            >
                                <Send className="h-5 w-5" />
                            </Button>
                        </div>
                    </div>
                </div>
            </div>

            <input
                ref={fileInputRef}
                type="file"
                accept="video/*"
                onChange={(e) => e.target.files?.[0] && handleFileUpload(e.target.files[0])}
                className="hidden"
                disabled={isProcessingVideo}
            />
        </div>
    )
}