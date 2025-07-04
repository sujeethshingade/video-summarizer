import React, { useState, useEffect } from 'react'
import Link from 'next/link'
import { Circle } from 'lucide-react'

type BackendStatus = 'online' | 'offline' | 'waking' | 'checking'

export function Navbar() {
    const [backendStatus, setBackendStatus] = useState<BackendStatus>('checking')

    const apiUrl = process.env.NEXT_PUBLIC_API_URL || (
        process.env.NODE_ENV === 'production'
            ? process.env.NEXT_PUBLIC_API_URL
            : 'http://localhost:8000'
    )

    const checkBackendStatus = async () => {
        try {
            setBackendStatus('checking')
            const controller = new AbortController()
            const timeoutId = setTimeout(() => controller.abort(), 10000)

            const response = await fetch(`${apiUrl}/`, {
                method: 'GET',
                signal: controller.signal,
            })

            clearTimeout(timeoutId)

            if (response.ok) {
                setBackendStatus('online')
            } else {
                setBackendStatus('offline')
            }
        } catch (error) {
            if (error instanceof Error && error.name === 'AbortError') {
                setBackendStatus('waking')
            } else {
                setBackendStatus('offline')
            }
        }
    }

    useEffect(() => {
        checkBackendStatus()

        const interval = setInterval(checkBackendStatus, 30000)

        return () => clearInterval(interval)
    }, [apiUrl])

    const getStatusColor = () => {
        switch (backendStatus) {
            case 'online':
                return 'text-green-600'
            case 'offline':
                return 'text-red-600'
            case 'waking':
                return 'text-yellow-600'
            case 'checking':
                return 'text-gray-600'
            default:
                return 'text-gray-600'
        }
    }

    const getStatusText = () => {
        switch (backendStatus) {
            case 'online':
                return 'Server Online'
            case 'offline':
                return 'Server Offline'
            case 'waking':
                return 'Server Waking Up...'
            case 'checking':
                return 'Checking Status...'
            default:
                return 'Unknown Status'
        }
    }

    return (
        <nav className="border-b border-gray-400 bg-gray-200 sticky top-0 z-50">
            <div className="max-w-5xl mx-auto px-4 flex items-center justify-between h-12">
                <h1 className="text-xl font-bold text-gray-900">Video to Text Summarizer</h1>

                <div className="flex items-center gap-2">
                    <Link href={apiUrl ?? `${apiUrl}`} target="_blank" rel="noopener noreferrer" className="flex items-center gap-2 cursor-pointer" title="Go to Server Deployment">
                        <Circle
                            className={`h-3 w-3 ${getStatusColor()} ${backendStatus === 'checking' || backendStatus === 'waking' ? 'animate-pulse' : ''}`}
                            fill="currentColor"
                        />
                        <span className={`text-sm font-medium hover:underline ${getStatusColor()}`}>
                            {getStatusText()}
                        </span>
                    </Link>
                </div>
            </div>
        </nav>
    )
}