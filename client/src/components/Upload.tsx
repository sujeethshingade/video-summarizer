"use client"

import { useState, useRef, useCallback } from 'react'
import { Button } from '@/components/ui/button'
import { Summary } from '@/components/Summary'
import { Upload as UploadIcon, X, FileVideo, AlertCircle, Loader2 } from 'lucide-react'

export function Upload() {
  const [file, setFile] = useState<File | null>(null)
  const [customPrompt, setCustomPrompt] = useState('')
  const [showDefaultPrompt, setShowDefaultPrompt] = useState(false)
  const [isUploading, setIsUploading] = useState(false)
  const [result, setResult] = useState<{ summary: string; prompt: string; metadata?: any } | null>(null)
  const [isDragOver, setIsDragOver] = useState(false)
  const fileInputRef = useRef<HTMLInputElement>(null)

  const apiUrl = process.env.NEXT_PUBLIC_API_URL || (
    process.env.NODE_ENV === 'production'
      ? process.env.NEXT_PUBLIC_API_URL
      : 'http://localhost:8000'
  )

  const defaultPrompt = `You are an AI assistant analyzing employee work session transcripts. Analyze this video content and your goal is to:
1. Summarize the key tasks the employee performed.
2. Categorize each task into the following: Repetitive, Analytical, Communication, Decision Making, Knowledge Work.
3. Identify tools and systems used.
4. Estimate time spent for each task based on the video duration and content analysis.
5. Suggest whether the task has High, Medium, or Low potential for AI support.`

  const onSelectFile = useCallback((f: File | null) => {
    setFile(f)
    setResult(null)
  }, [])

  const onDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    setIsDragOver(false)
    const files = e.dataTransfer.files
    if (files.length > 0) {
      const videoFile = files[0]
      if (videoFile.type.startsWith('video/')) {
        onSelectFile(videoFile)
      }
    }
  }, [onSelectFile])

  const onDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    setIsDragOver(true)
  }, [])

  const onDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    setIsDragOver(false)
  }, [])

  const onFileInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    onSelectFile(e.target.files?.[0] || null)
  }

  const onUpload = async () => {
    if (!file) return
    setIsUploading(true)
    setResult(null)
    try {
      const form = new FormData()
      form.append('file', file)
      if (customPrompt.trim()) form.append('prompt', customPrompt.trim())

      const res = await fetch(`${apiUrl}/api/upload`, { method: 'POST', body: form })
  const data = await res.json()
      if (!res.ok) throw new Error(data?.detail || 'Upload failed')
  setResult({ summary: data.summary, prompt: data.prompt, metadata: data.metadata })
    } catch (err) {
      setResult({
        summary: 'Error processing video. Please try again.',
        prompt: customPrompt || 'Default analysis prompt (server)',
        metadata: null
      })
    } finally {
      setIsUploading(false)
    }
  }

  const reset = () => {
    setFile(null)
    setCustomPrompt('')
    setResult(null)
    setShowDefaultPrompt(false)
    if (fileInputRef.current) fileInputRef.current.value = ''
  }

  const formatFileSize = (bytes: number) => {
    if (bytes === 0) return '0 Bytes'
    const k = 1024
    const sizes = ['Bytes', 'KB', 'MB', 'GB']
    const i = Math.floor(Math.log(bytes) / Math.log(k))
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i]
  }

  return (
    <div className="space-y-8">
      <div className="bg-white border border-gray-200 rounded-xl shadow-sm overflow-hidden">
        <div className="p-8">
          <div className="text-center mb-8">
            <h1 className="text-2xl font-bold text-gray-900 mb-3">Video Summary & Analysis</h1>
          </div>

          <div
            className={`relative border-2 border-dashed rounded-xl p-8 transition-all duration-200 ${
              isDragOver
                ? 'border-blue-500 bg-blue-50'
                : file
                ? 'border-green-500 bg-green-50'
                : 'border-gray-300 hover:border-gray-400 bg-gray-50'
            }`}
            onDrop={onDrop}
            onDragOver={onDragOver}
            onDragLeave={onDragLeave}
          >
            <input
              type="file"
              accept="video/*"
              ref={fileInputRef}
              onChange={onFileInputChange}
              disabled={isUploading}
              className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
            />
            
            <div className="text-center">
              {file ? (
                <div className="space-y-4">
                  <div className="inline-flex items-center justify-center w-16 h-16 bg-green-100 rounded-full">
                    <FileVideo className="w-8 h-8 text-green-600" />
                  </div>
                  <div>
                    <p className="text-md font-semibold text-gray-900">{file.name}</p>
                    <p className="text-sm text-gray-500">{formatFileSize(file.size)}</p>
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={(e) => {
                        e.stopPropagation()
                        reset()
                      }}
                      className="mt-4 text-red-500 hover:text-red-700 bg-red-50 hover:bg-red-100"
                    >
                      <X className="w-4 h-4 mr-1" />
                      Remove
                    </Button>
                  </div>
                </div>
              ) : (
                <div className="space-y-4">
                  <div className="inline-flex items-center justify-center w-16 h-16 bg-gray-100 rounded-full">
                    <UploadIcon className="w-8 h-8 text-gray-600" />
                  </div>
                  <div>
                    <p className="text-lg font-semibold text-gray-900">
                      Drag and drop your video here
                    </p>
                    <p className="text-gray-500">or click to browse files</p>
                  </div>
                </div>
              )}
            </div>
          </div>

          <div className="mt-8 space-y-4">
            <div className="flex items-center justify-between">
              <label className="text-lg font-semibold text-gray-900">Custom Prompt (Optional)</label>
              <Button
                variant="ghost"
                size="sm"
                onClick={() => setShowDefaultPrompt(!showDefaultPrompt)}
                className="text-blue-600 hover:text-blue-700"
              >
                {showDefaultPrompt ? 'Hide' : 'Show'} Default Prompt
              </Button>
            </div>

            {showDefaultPrompt && (
              <div className="bg-gray-50 border border-gray-200 rounded-xl p-4">
                <pre className="whitespace-pre-wrap text-sm text-gray-600 max-h-40 overflow-y-auto">
                  {defaultPrompt}
                </pre>
              </div>
            )}

            <div className="space-y-2">
              <textarea
                className="w-full min-h-26 rounded-xl border border-gray-300 px-4 py-3 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent resize-none"
                placeholder="Enter custom prompt here..."
                value={customPrompt}
                onChange={(e) => setCustomPrompt(e.target.value)}
                disabled={isUploading}
              />
              <p className="text-sm text-gray-500">
                {customPrompt.trim() ? 'Using custom prompt' : 'Will use default prompt if left blank'}
              </p>
            </div>
          </div>

          <div className="mt-8 flex gap-4 justify-center">
            <Button
              onClick={onUpload}
              disabled={!file || isUploading}
              className="bg-blue-600 hover:bg-blue-700 text-white px-8 py-3 rounded-lg font-semibold flex items-center gap-2 disabled:bg-gray-300"
            >
              {isUploading ? (
                <>
                  <Loader2 className="w-5 h-5 animate-spin" />
                  Analyzing Video...
                </>
              ) : (
                <>
                  <FileVideo className="w-5 h-5" />
                  Analyze Video
                </>
              )}
            </Button>
            
            <Button
              variant="outline"
              onClick={reset}
              disabled={isUploading}
              className="px-8 py-3 rounded-lg font-semibold"
            >
              Reset
            </Button>
          </div>
        </div>
      </div>

      {result && (
        <div className="space-y-6">
          <div className="bg-white border border-gray-200 rounded-xl p-6 shadow-sm">
            <h3 className="text-lg font-semibold text-gray-900 mb-4">Video Analysis Result</h3>
            <Summary content={result.summary} />
          </div>
        </div>
      )}
    </div>
  )
}
