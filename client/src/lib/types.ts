export interface Message {
    id: string
    text: string
    isUser: boolean
    isProcessing?: boolean
    filename?: string
    isVideoSummary?: boolean
}

export interface Task {
    task: string
    category: string
    tools: string[]
    timeEstimate: string
    aiOpportunity: string
}

export interface SummaryData {
    summary: string
    tasks: Task[]
}

export interface SummaryDisplayProps {
    content: string
}

export interface SummaryDocument {
    _id: string
    filename: string
    summary_data: SummaryData
    prompt_used?: string
    processed_at: string
    metadata?: {
        filename?: string
        transcript_length?: number
        has_audio?: boolean
        has_visual?: boolean
        frames_analyzed?: number
        keyframe_interval?: number
        video_duration_seconds?: number
    used_custom_prompt?: boolean
    }
}

export interface JobStatus {
    id: string
    filename: string
    status: 'queued' | 'processing' | 'completed' | 'error'
    error?: string
    document_id?: string
    summary?: string
    metadata?: any
    prompt_used?: string
}