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
                            <h1 className="text-xl font-bold text-gray-900 mb-2">Video Analysis Output</h1>
                        </div>
                        <Button onClick={fetchSummaries} variant="outline" className="bg-white">
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
                                        <div className="p-6 border-t border-gray-200">
                                            {summary.metadata && (
                                                <div className="mb-4 flex flex-wrap gap-4 text-xs text-gray-600">
                                                    <span className="bg-gray-100 text-gray-800 px-2 py-1 rounded-full">
                                                        Document ID: {summary._id}
                                                    </span>
                                                    {summary.metadata.has_audio && (
                                                        <span className="bg-gray-100 text-gray-800 px-2 py-1 rounded-full">
                                                            Audio: {summary.metadata.transcript_length || 0} chars analysed
                                                        </span>
                                                    )}
                                                    {summary.metadata.has_visual && (
                                                        <span className="bg-gray-100 text-gray-800 px-2 py-1 rounded-full">
                                                            Visual: {summary.metadata.frames_analyzed || 0} frames analysed
                                                        </span>
                                                    )}
                                                </div>
                                            )}
                                            <Summary content={JSON.stringify(summary.summary_data)} />
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
