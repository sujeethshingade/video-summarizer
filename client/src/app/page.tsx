'use client';

import { Navbar } from '@/components/Navbar';
import { Upload } from '../components/Upload';

export default function Home() {
  return (
    <div className="min-h-screen flex flex-col">
      <Navbar />
      <main className="flex-1 bg-gray-200 py-8">
        <div className="max-w-6xl mx-auto px-4">
          <Upload />
        </div>
      </main>
    </div>
  );
}
