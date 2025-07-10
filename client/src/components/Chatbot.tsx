"use client"

import { useState, useRef, useEffect } from "react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Avatar, AvatarFallback } from "@/components/ui/avatar"
import { Send, Paperclip, Bot, User } from "lucide-react"
import { cn } from "@/lib/utils"
import { Message } from "@/lib/types"
import { Summary } from "./Summary"

export function Chatbot() {
    const [messages, setMessages] = useState<Message[]>([])
    const [isProcessingVideo, setIsProcessingVideo] = useState(false)
    const [message, setMessage] = useState("")
    const messagesEndRef = useRef<HTMLDivElement>(null)
    const fileInputRef = useRef<HTMLInputElement>(null)

    const apiUrl = process.env.NEXT_PUBLIC_API_URL || (
        process.env.NODE_ENV === 'production'
            ? process.env.NEXT_PUBLIC_API_URL
            : 'http://localhost:8000'
    )

    useEffect(() => {
        messagesEndRef.current?.scrollIntoView({ behavior: "smooth" })
    }, [messages])

    const addMessage = (text: string, isUser: boolean, filename?: string, isProcessing?: boolean, isVideoSummary?: boolean) => {
        setMessages(prev => [...prev, {
            id: `${Date.now()}-${isUser}`,
            text,
            isUser,
            filename,
            isProcessing,
            isVideoSummary
        }])
    }

    const handleFileUpload = async (file: File) => {
        if (!file) return

        addMessage(`${file.name}`, true, file.name)
        addMessage('Processing...', false, undefined, true)
        setIsProcessingVideo(true)

        try {
            const formData = new FormData()
            formData.append('file', file)

            const response = await fetch(`${apiUrl}/api/upload`, {
                method: 'POST',
                body: formData,
            })

            const data = await response.json()

            setMessages(prev => prev.filter(msg => !msg.isProcessing))

            if (response.ok && data.summary) {
                addMessage(data.summary, false, undefined, false, true)
            } else {
                addMessage('Error processing video. Please try again.', false)
            }
        } catch (error) {
            setMessages(prev => prev.filter(msg => !msg.isProcessing))
            addMessage('Upload failed. Please try again.', false)
        } finally {
            setIsProcessingVideo(false)
        }
    }

    const handleSendMessage = async (text: string) => {
        addMessage(text, true)

        try {
            const response = await fetch(`${apiUrl}/api/chat`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ message: text })
            })

            const data = await response.json()

            if (response.ok && data.response) {
                addMessage(data.response, false)
            }
            else {
                addMessage("Error processing message. Please try again.", false)
            }
        } catch (error) {
            addMessage("Error sending message. Please try again.", false)
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
                <div className="max-w-6xl mx-auto px-4 py-6 space-y-6">
                    {messages.map((msg) => {
                        const isUser = msg.isUser;
                        return (
                            <div key={msg.id} className={cn("flex", isUser ? "justify-end" : "justify-start")}>
                                {!isUser && (
                                    <Avatar className="w-10 h-10 mt-1">
                                        <AvatarFallback className="text-gray-900">
                                            <Bot className="h-6 w-6" />
                                        </AvatarFallback>
                                    </Avatar>
                                )}
                                <div className={cn("flex flex-col max-w-[80%]", isUser ? "items-end" : "items-start")}>
                                    <div className="rounded-2xl px-4 py-3 text-sm shadow-sm bg-white border border-gray-400 text-gray-900">
                                        <div className="leading-relaxed">
                                            {msg.isVideoSummary ? (
                                                <Summary content={msg.text} />
                                            ) : (
                                                <div className="whitespace-pre-wrap">{msg.text}</div>
                                            )}
                                        </div>
                                    </div>
                                </div>
                                {isUser && (
                                    <Avatar className="w-10 h-10 mt-1">
                                        <AvatarFallback className="text-gray-900">
                                            <User className="h-6 w-6" />
                                        </AvatarFallback>
                                    </Avatar>
                                )}
                            </div>
                        );
                    })}
                    <div ref={messagesEndRef} />
                </div>
            </div>

            <div className="flex-shrink-0 bg-gray-200 border-t border-gray-400">
                <div className="max-w-6xl mx-auto px-4 py-3">
                    <div className="flex items-center gap-2">
                        <Input
                            value={message}
                            placeholder="Ask anything..."
                            disabled={isProcessingVideo}
                            onChange={(e) => setMessage(e.target.value)}
                            onKeyDown={(e) => e.key === "Enter" && !e.shiftKey && (e.preventDefault(), sendTextMessage())}
                            className="h-10 border-gray-400 focus:border-gray-900 focus:ring-gray-900"
                        />
                        <Button
                            type="button"
                            variant="ghost"
                            size="default"
                            onClick={() => fileInputRef.current?.click()}
                            disabled={isProcessingVideo}
                            className="hover:bg-white cursor-pointer focus:outline-none rounded-2xl border border-gray-400 text-gray-900"
                            tabIndex={-1}
                        >
                            <Paperclip className="h-5 w-5" />
                        </Button>
                        <Button
                            type="button"
                            variant="ghost"
                            size="default"
                            onClick={sendTextMessage}
                            disabled={!message.trim()}
                            className="hover:bg-white cursor-pointer focus:outline-none rounded-2xl border border-gray-400 text-gray-900"
                        >
                            <Send className="h-5 w-5" />
                        </Button>
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