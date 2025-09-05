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

    if (summaryData && (summaryData as any).summary) {
        return (
            <div className="space-y-6">
                <div className="bg-gray-50 border border-gray-200 rounded-lg p-4">
                    <h3 className="font-semibold text-gray-900 mb-2">Summary</h3>
                    <p className="text-gray-700 leading-relaxed">{(summaryData as any).summary}</p>
                </div>

                {(summaryData as any).tasks && (summaryData as any).tasks.length > 0 && (
                    <div className="bg-gray-50 border border-gray-200 rounded-lg p-4">
                        <h3 className="font-semibold text-gray-900 mb-3">Tasks Identified</h3>
                        <div className="space-y-3">
                            {(summaryData as any).tasks.map((task: Task, index: number) => (
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

    try {
        const obj = JSON.parse(content)

        if (typeof obj === 'string') {
            return <div className="whitespace-pre-wrap leading-relaxed">{obj}</div>
        }

        const textKeys = ["text", "answer", "result", "message", "content", "response", "output"]
        for (const key of textKeys) {
            const val = (obj as any)[key]
            if (typeof val === 'string' && val.trim().length > 0) {
                return <div className="whitespace-pre-wrap leading-relaxed">{val}</div>
            }
        }

        const isPlainObject = (v: any) => v && typeof v === 'object' && !Array.isArray(v)

        const renderValue = (key: string | null, value: any, depth = 0) => {
            const pad = Math.min(depth, 4)
            const padCls = ["pl-0", "pl-2", "pl-4", "pl-6", "pl-8"][pad]
            const containerCls = depth === 0 ? "space-y-2" : "space-y-1"
            const sectionCls = "border border-gray-200 rounded-md p-3 bg-white"
            const titleCls = "text-sm font-semibold text-gray-900"
            const textCls = "text-sm text-gray-700 whitespace-pre-wrap"
            const listCls = "list-disc list-inside space-y-1 text-sm text-gray-700"

            if (value === null || value === undefined) {
                return <div className={textCls}>{String(value)}</div>
            }

            if (typeof value === 'string' || typeof value === 'number' || typeof value === 'boolean') {
                return (
                    <div className={textCls}>{String(value)}</div>
                )
            }

            if (Array.isArray(value)) {
                const allPrimitives = value.every(v => v === null || ["string", "number", "boolean"].includes(typeof v))
                if (allPrimitives) {
                    return (
                        <ul className={listCls}>
                            {value.map((v, i) => (
                                <li key={(key || 'arr') + '-' + i}>{String(v)}</li>
                            ))}
                        </ul>
                    )
                }
                return (
                    <div className={`space-y-2 ${padCls}`}>
                        {value.map((item, idx) => (
                            <div key={(key || 'obj') + '-' + idx} className={sectionCls}>
                                {isPlainObject(item)
                                    ? renderObject(item, depth + 1)
                                    : renderValue(null, item, depth + 1)
                                }
                            </div>
                        ))}
                    </div>
                )
            }

            if (isPlainObject(value)) {
                return renderObject(value, depth)
            }

            // Fallback
            return <pre className="text-xs text-gray-700 overflow-x-auto">{JSON.stringify(value, null, 2)}</pre>
        }

        const renderObject = (o: Record<string, any>, depth = 0) => {
            const keys = Object.keys(o)
            return (
                <div className="space-y-2">
                    {keys.map((k) => (
                        <div key={k} className="border border-gray-200 rounded-md p-3 bg-gray-50">
                            <div className="text-sm font-semibold text-gray-900 mb-1">{toTitle(k)}</div>
                            {renderValue(k, o[k], depth + 1)}
                        </div>
                    ))}
                </div>
            )
        }

        const toTitle = (s: string) => s
            .replace(/_/g, ' ')
            .replace(/([a-z])([A-Z])/g, '$1 $2')
            .replace(/\s+/g, ' ')
            .replace(/^./, c => c.toUpperCase())

        return (
            <div className="space-y-3">
                {isPlainObject(obj) ? (
                    renderObject(obj)
                ) : Array.isArray(obj) ? (
                    renderValue('root', obj)
                ) : (
                    <pre className="text-sm leading-relaxed bg-gray-50 border border-gray-200 rounded-md p-3 overflow-x-auto">
                        {JSON.stringify(obj, null, 2)}
                    </pre>
                )}
            </div>
        )
    } catch (e) {
        // Fallback to raw content
        return <div className="whitespace-pre-wrap leading-relaxed">{content}</div>
    }
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
