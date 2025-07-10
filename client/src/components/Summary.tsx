"use client"

import {Task, SummaryData, SummaryDisplayProps} from "@/lib/types"

export function Summary({ content }: SummaryDisplayProps) {
    let summaryData: SummaryData | null = null

    try {
        summaryData = JSON.parse(content)
    } catch (error) {
        return (
            <div className="whitespace-pre-wrap leading-relaxed">
                {content}
            </div>
        )
    }

    if (summaryData && summaryData.summary) {
        return (
            <div className="space-y-4">
                <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
                    <h3 className="font-semibold text-blue-900 mb-2">Summary</h3>
                    <p className="text-blue-800 leading-relaxed">{summaryData.summary}</p>
                </div>

                {summaryData.tasks && summaryData.tasks.length > 0 && (
                    <div className="bg-gray-50 border border-gray-200 rounded-lg p-4">
                        <h3 className="font-semibold text-gray-900 mb-3">Tasks Identified</h3>
                        <div className="space-y-3">
                            {summaryData.tasks.map((task, index) => (
                                <div key={index} className="bg-white border border-gray-200 rounded-md p-3">
                                    <div className="font-medium text-gray-900 mb-2">
                                        {index + 1}. {task.task}
                                    </div>
                                    <div className="grid grid-cols-2 md:grid-cols-4 gap-2 text-sm">
                                        <div>
                                            <span className="font-medium text-gray-600">Category:</span>
                                            <div className={`inline-block ml-1 px-2 py-1 rounded-full text-xs font-medium ${getCategoryColor(task.category)}`}>
                                                {task.category}
                                            </div>
                                        </div>
                                        <div>
                                            <span className="font-medium text-gray-600">Tools:</span>
                                            <div className="text-gray-800">
                                                {Array.isArray(task.tools) ? task.tools.join(", ") : task.tools || "N/A"}
                                            </div>
                                        </div>
                                        <div>
                                            <span className="font-medium text-gray-600">Time:</span>
                                            <div className="text-gray-800">{task.timeEstimate || "Unknown"}</div>
                                        </div>
                                        <div>
                                            <span className="font-medium text-gray-600">AI Opportunity:</span>
                                            <div className={`inline-block ml-1 px-2 py-1 rounded-full text-xs font-medium ${getOpportunityColor(task.aiOpportunity)}`}>
                                                {task.aiOpportunity}
                                            </div>
                                        </div>
                                    </div>
                                </div>
                            ))}
                        </div>
                    </div>
                )}
            </div>
        )
    }

    return (
        <div className="whitespace-pre-wrap leading-relaxed">
            {content}
        </div>
    )
}

function getCategoryColor(category: string): string {
    switch (category?.toLowerCase()) {
        case 'repetitive':
            return 'bg-red-100 text-red-800'
        case 'analytical':
            return 'bg-purple-100 text-purple-800'
        case 'communication':
            return 'bg-blue-100 text-blue-800'
        case 'decision-making':
            return 'bg-yellow-100 text-yellow-800'
        case 'knowledge work':
            return 'bg-green-100 text-green-800'
        default:
            return 'bg-gray-100 text-gray-800'
    }
}

function getOpportunityColor(opportunity: string): string {
    switch (opportunity?.toLowerCase()) {
        case 'high':
            return 'bg-green-100 text-green-800'
        case 'medium':
            return 'bg-yellow-100 text-yellow-800'
        case 'low':
            return 'bg-red-100 text-red-800'
        default:
            return 'bg-gray-100 text-gray-800'
    }
}
