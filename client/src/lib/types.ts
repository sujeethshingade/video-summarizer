export interface Message {
    id: string
    text: string
    isUser: boolean
    isProcessing?: boolean
    filename?: string
}