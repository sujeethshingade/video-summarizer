import React from 'react'

export function Navbar() {
    return (
        <nav className="border-b border-gray-400 bg-gray-200 sticky top-0 z-50">
            <div className="max-w-7xl mx-auto px-4 flex items-center h-12">
                <h1 className="text-xl font-bold text-gray-900">Video to Text Summarizer</h1>
            </div>
        </nav>
    )
}