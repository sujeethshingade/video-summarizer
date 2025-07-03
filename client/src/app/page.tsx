'use client';

import { Chatbot } from '@/components/Chatbot';
import { Navbar } from '@/components/Navbar';

export default function Home() {
  return (
    <div className="h-screen flex flex-col overflow-hidden">
      <Navbar />
      <div className="flex-1 overflow-hidden">
        <Chatbot />
      </div>
    </div>);
}
