"use client"

import { useState, useRef, useCallback, useEffect } from 'react'
import { Button } from '@/components/ui/button'
import { Summary } from '@/components/Summary'
import { Upload as UploadIcon, FileVideo, Loader2, CheckCircle2, Clock, AlertCircle } from 'lucide-react'
import { JobStatus } from '@/lib/types'

export function Upload() {
  const [files, setFiles] = useState<File[]>([])
  const [customPrompt, setCustomPrompt] = useState('')
  const [showDefaultPrompt, setShowDefaultPrompt] = useState(false)
  const [isUploading, setIsUploading] = useState(false)
  const [jobs, setJobs] = useState<JobStatus[]>([])
  const [isDragOver, setIsDragOver] = useState(false)
  const [startedProcessing, setStartedProcessing] = useState(false)
  const fileInputRef = useRef<HTMLInputElement>(null)
  const pollingRef = useRef<NodeJS.Timeout | null>(null)

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

  const onSelectFiles = useCallback((list: File[]) => {
    setFiles(prev => {
      if (!prev.length) return list
      const existingKeys = new Set(prev.map(f => f.name + ':' + f.size))
      const additions = list.filter(f => !existingKeys.has(f.name + ':' + f.size))
      return [...prev, ...additions]
    })
    if (!startedProcessing) setJobs([])
  }, [startedProcessing])

  const onDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    setIsDragOver(false)
    const dtFiles = Array.from(e.dataTransfer.files).filter(f => f.type.startsWith('video/'))
    if (dtFiles.length) {
      onSelectFiles(dtFiles)
    }
  }, [onSelectFiles])

  const onDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    setIsDragOver(true)
  }, [])

  const onDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    setIsDragOver(false)
  }, [])

  const onFileInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const list = e.target.files ? Array.from(e.target.files) : []
    const videos = list.filter(f => f.type.startsWith('video/'))
    if (videos.length) onSelectFiles(videos)
    if (fileInputRef.current) fileInputRef.current.value = ''
  }

  const pollJobs = useCallback(async (jobIds: string[]) => {
    if (!jobIds.length) return
    try {
      const updated: JobStatus[] = []
      for (const id of jobIds) {
        const res = await fetch(`${apiUrl}/api/job/${id}`)
        if (res.ok) {
          const data = await res.json()
          updated.push(data as JobStatus)
        }
      }
      setJobs(prev => {
        const map = new Map(prev.map(j => [j.id, j]))
        updated.forEach(u => map.set(u.id, { ...map.get(u.id), ...u }))
        return Array.from(map.values())
      })
    } catch (e) {
    }
  }, [apiUrl])

  useEffect(() => {
    const active = jobs.filter(j => j.status === 'queued' || j.status === 'processing').length > 0
    if (active) {
      if (!pollingRef.current) {
        pollingRef.current = setInterval(() => {
          pollJobs(jobs.map(j => j.id))
        }, 4000)
      }
    } else {
      if (pollingRef.current) {
        clearInterval(pollingRef.current)
        pollingRef.current = null
      }
    }
    return () => {
      if (pollingRef.current && jobs.every(j => j.status === 'completed' || j.status === 'error')) {
        clearInterval(pollingRef.current)
        pollingRef.current = null
      }
    }
  }, [jobs, pollJobs])

  const onUpload = async () => {
    if (!files.length) return
  setIsUploading(true)
  setStartedProcessing(true)
    try {
      const form = new FormData()
      files.forEach(f => form.append('files', f))
      if (customPrompt.trim()) form.append('prompt', customPrompt.trim())
      const res = await fetch(`${apiUrl}/api/upload-multiple`, { method: 'POST', body: form })
      const data = await res.json()
      if (!res.ok) throw new Error(data?.detail || 'Upload failed')
      const jobIds: string[] = data.job_ids || []
      setJobs(jobIds.map(id => ({ id, filename: files[jobIds.indexOf(id)]?.name || 'unknown', status: 'queued' } as JobStatus)))
      setTimeout(() => pollJobs(jobIds), 500)
    } catch (e) {
      setJobs(files.map((f, idx) => ({ id: `local-${idx}`, filename: f.name, status: 'error', error: 'Failed to start upload' })))
    } finally {
      setIsUploading(false)
    }
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
                : files.length
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
              multiple
              ref={fileInputRef}
              onChange={onFileInputChange}
              className={`absolute inset-0 w-full h-full opacity-0 cursor-pointer ${startedProcessing ? 'pointer-events-none' : ''}`}
            />
            
            <div className="text-center">
              {files.length ? (
                <div className="space-y-4">
                  <div className="flex flex-col items-center">
                    <div className="inline-flex items-center justify-center w-16 h-16 bg-green-100 rounded-full">
                      <FileVideo className="w-8 h-8 text-green-600" />
                    </div>
                    <p className="text-lg font-semibold text-gray-900 mt-2">{files.length} file(s) selected</p>
                  </div>
                  <ul className="max-h-40 overflow-y-auto text-left text-sm space-y-1">
                    {files.map((f, i) => (
                      <li key={i} className="flex justify-between items-center px-2 py-1">
                        <span className="truncate max-w-[700px]" title={f.name}>{f.name}</span>
                        <span className="text-gray-500 ml-2 whitespace-nowrap">{formatFileSize(f.size)}</span>
                      </li>
                    ))}
                  </ul>
                    {!startedProcessing && (
                      <Button
                        type="button"
                        variant="outline"
                        size="sm"
                        onClick={(e) => { e.stopPropagation(); fileInputRef.current?.click() }}
                        className="text-blue-600 border-blue-200 hover:bg-blue-50"
                      >Add More Videos</Button>
                    )}
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
              disabled={!files.length || startedProcessing}
              className="bg-blue-600 hover:bg-blue-700 text-white px-8 py-3 rounded-lg font-semibold flex items-center gap-2 disabled:bg-gray-300"
            >
              {(isUploading || (startedProcessing && jobs.some(j => j.status === 'queued' || j.status === 'processing'))) ? (
                <>
                  <Loader2 className="w-5 h-5 animate-spin" />
                  {isUploading ? 'Submitting...' : 'Processing...'}
                </>
              ) : (
                <>
                  Analyze {files.length > 1 ? 'Videos' : 'Video'}
                </>
              )}
            </Button>
          </div>
        </div>
      </div>

      {jobs.length > 0 && (
        <div className="space-y-4">
          <h3 className="text-lg font-semibold text-gray-900">Processing Status</h3>
          <div className="space-y-3">
            {jobs.map(job => {
              const statusMap: Record<string,string> = {
                queued: 'bg-gray-100 text-gray-700',
                processing: 'bg-blue-100 text-blue-700',
                completed: 'bg-green-100 text-green-700',
                error: 'bg-red-100 text-red-700'
              }
              const statusBadge = statusMap[job.status] || 'bg-gray-100 text-gray-600'
              let icon: JSX.Element
              switch (job.status) {
                case 'completed': icon = <CheckCircle2 className="w-5 h-5 text-green-600" />; break
                case 'processing': icon = <Loader2 className="w-5 h-5 animate-spin text-blue-600" />; break
                case 'error': icon = <AlertCircle className="w-5 h-5 text-red-600" />; break
                default: icon = <Clock className="w-5 h-5 text-gray-500" />; break
              }
              return (
                <div key={job.id} className="bg-white border border-gray-200 rounded-lg p-4 shadow-sm">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-3">
                      {icon}
                      <div>
                        <p className="font-medium text-gray-900 text-sm">{job.filename}</p>
                        {job.error && <p className="text-xs text-red-600 mt-1">{job.error}</p>}
                      </div>
                    </div>
                    <div className="flex items-center gap-2">
                      {job.status === 'completed' && job.summary && <span className="text-xs text-green-600 font-medium">✓</span>}
                    </div>
                  </div>
                  {job.status === 'completed' && job.summary && (
                    <div className="mt-3">
                      <Summary content={job.summary} />
                    </div>
                  )}
                </div>
              )
            })}
          </div>
        </div>
      )}
    </div>
  )
}
