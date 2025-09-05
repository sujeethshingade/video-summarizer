"use client"

import { useState, useEffect } from "react"
import { Summary } from "@/components/Summary"
import { Navbar } from "@/components/Navbar"
import { Button } from "@/components/ui/button"
import { RefreshCw, Calendar, FileVideo, ChevronDown, ChevronUp } from "lucide-react"
import { SummaryDocument } from "@/lib/types"

export default function OutputPage() {
    const [summaries, setSummaries] = useState<SummaryDocument[]>([])
    const [loading, setLoading] = useState(true)
    const [error, setError] = useState<string | null>(null)
    const [expandedSummaryId, setExpandedSummaryId] = useState<string | null>(null)

    const defaultPrompt = `You are an AI assistant analyzing employee work session transcripts. Analyze this video content and your goal is to:
1. Summarize the key tasks the employee performed.
2. Categorize each task into the following: Repetitive, Analytical, Communication, Decision Making, Knowledge Work.
3. Identify tools and systems used.
4. Estimate time spent for each task based on the video duration and content analysis.
5. Suggest whether the task has High, Medium, or Low potential for AI support.`

    const apiUrl = process.env.NEXT_PUBLIC_API_URL || (
        process.env.NODE_ENV === 'production'
            ? process.env.NEXT_PUBLIC_API_URL
            : 'http://localhost:8000'
    )

    const fetchSummaries = async () => {
        setLoading(true)
        setError(null)
        try {
            const response = await fetch(`${apiUrl}/api/summaries`)
            const data = await response.json()

            if (response.ok && data.success) {
                setSummaries(data.summaries)
            } else {
                setError('Failed to fetch summaries from database')
            }
        } catch (err) {
            setError('Error connecting to server')
            console.error('Error fetching summaries:', err)
        } finally {
            setLoading(false)
        }
    }

    useEffect(() => {
        fetchSummaries()
    }, [])

    const formatDate = (dateString: string) => {
        return new Date(dateString).toLocaleString()
    }

    const toggleSummaryDetails = (id: string) => {
        setExpandedSummaryId(expandedSummaryId === id ? null : id)
    }

    if (loading) {
        return (
            <div className="h-screen flex flex-col overflow-hidden">
                <Navbar />
                <div className="flex-1 flex items-center justify-center bg-gray-200">
                    <div className="text-center">
                        <RefreshCw className="h-8 w-8 animate-spin mx-auto mb-4 text-gray-600" />
                        <p className="text-gray-600">Loading summaries...</p>
                    </div>
                </div>
            </div>
        )
    }

    if (error) {
        return (
            <div className="h-screen flex flex-col overflow-hidden">
                <Navbar />
                <div className="flex-1 flex items-center justify-center bg-gray-200">
                    <div className="text-center">
                        <p className="text-gray-600 mb-4">{error}</p>
                        <Button onClick={fetchSummaries} className="bg-blue-600 hover:bg-blue-700">
                            <RefreshCw className="h-4 w-4 mr-2" />
                            Retry
                        </Button>
                    </div>
                </div>
            </div>
        )
    }

    return (
        <div className="h-screen bg-gray-200 flex flex-col overflow-hidden">
            <Navbar />
            <div className="flex-1 overflow-auto bg-gray-200">
                <div className="max-w-6xl mx-auto px-4 py-8">
                    <div className="flex items-center justify-between mb-8">
                        <div>
                            <h1 className="text-lg font-bold text-gray-900 mb-2">Video Analysis Result</h1>
                        </div>
                        <Button onClick={fetchSummaries} variant="outline" className="bg-white rounded-lg">
                            <RefreshCw className="h-4 w-4 mr-2" />
                            Refresh
                        </Button>
                    </div>

                    {summaries.length === 0 ? (
                        <div className="text-center py-12">
                            <h3 className="text-lg font-medium text-gray-900">No Summaries found</h3>
                            <p className="text-gray-600">Process some videos to see their analysis here</p>
                        </div>
                    ) : (
                        <div className="space-y-4">
                            {summaries.map((summary) => (
                                <div key={summary._id} className="bg-white rounded-lg shadow-sm border border-gray-200">
                                    <div
                                        className="px-6 py-4 flex items-center justify-between cursor-pointer"
                                        onClick={() => toggleSummaryDetails(summary._id)}
                                    >
                                        <div className="flex items-center space-x-3">
                                            <FileVideo className="h-5 w-5 text-gray-500" />
                                            <h3 className="text-md font-semibold text-gray-900">
                                                {summary.filename}
                                            </h3>
                                        </div>
                                        <div className="flex items-center space-x-4 text-sm text-gray-500">
                                            <div className="flex items-center space-x-1">
                                                <Calendar className="h-4 w-4" />
                                                <span>{formatDate(summary.processed_at)}</span>
                                            </div>
                                            {expandedSummaryId === summary._id ? (
                                                <ChevronUp className="h-5 w-5 text-gray-500" />
                                            ) : (
                                                <ChevronDown className="h-5 w-5 text-gray-500" />
                                            )}
                                        </div>
                                    </div>
                                    {expandedSummaryId === summary._id && (
                                        <div className="p-6 border-t border-gray-200 space-y-6">

                                            {summary.metadata && (
                                                <div className="flex flex-wrap gap-2 text-xs">
                                                    <span className="bg-orange-100 text-orange-800 px-3 py-1 rounded-full font-medium">
                                                        Document ID: {summary._id}
                                                    </span>
                                                    {summary.metadata.has_audio && (
                                                        <span className="bg-blue-100 text-blue-800 px-3 py-1 rounded-full font-medium">
                                                            Audio: {summary.metadata.transcript_length || 0} chars
                                                        </span>
                                                    )}
                                                    {summary.metadata.has_visual && (
                                                        <span className="bg-green-100 text-green-800 px-3 py-1 rounded-full font-medium">
                                                            Visual: {summary.metadata.frames_analyzed || 0} frames
                                                        </span>
                                                    )}
                                                    {summary.metadata.keyframe_interval && (
                                                        <span className="bg-purple-100 text-purple-800 px-3 py-1 rounded-full font-medium">
                                                            Keyframe: {summary.metadata.keyframe_interval}s
                                                        </span>
                                                    )}
                                                    {summary.metadata.video_duration_seconds && (
                                                        <span className="bg-yellow-100 text-yellow-800 px-3 py-1 rounded-full font-medium">
                                                            Duration: {Math.floor(summary.metadata.video_duration_seconds / 60)}m {Math.floor(summary.metadata.video_duration_seconds % 60)}s
                                                        </span>
                                                    )}
                                                </div>
                                            )}

                                            {(summary.prompt_used || summary.metadata?.used_custom_prompt === false) && (
                                                <div className="bg-gray-50 border border-gray-200 rounded-lg p-4">
                                                    <h4 className="font-semibold text-gray-900 mb-3 flex items-center gap-2">
                                                        Prompt Used
                                                    </h4>
                                                    <div className="bg-white border border-gray-200 rounded-md p-3 max-h-40 overflow-y-auto">
                                                        <pre className="whitespace-pre-wrap text-sm text-gray-700">
                                                            {summary.metadata?.used_custom_prompt === false
                                                                ? defaultPrompt
                                                                : (summary.prompt_used || defaultPrompt)}
                                                        </pre>
                                                    </div>
                                                </div>
                                            )}

                                            <div>
                                                <Summary content={JSON.stringify(summary.summary_data)} />
                                            </div>
                                        </div>
                                    )}
                                </div>
                            ))}
                        </div>
                    )}
                </div>
            </div>
        </div>
    )
}
