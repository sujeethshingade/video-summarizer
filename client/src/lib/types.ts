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