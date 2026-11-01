'use client';

import { useState } from 'react';
import { apiClient } from '@/lib/api';
import { ChatResponse, Citation } from '@/types/api';

interface Message {
  role: 'user' | 'assistant';
  content: string;
  citations?: Citation[];
}

export default function ChatPage() {
  const [message, setMessage] = useState('');
  const [messages, setMessages] = useState<Message[]>([]);
  const [loading, setLoading] = useState(false);

  const handleSend = async () => {
    if (!message.trim() || loading) return;
    const userMsg = message;
    setMessages(prev => [...prev, { role: 'user', content: userMsg }]);
    setMessage('');
    setLoading(true);

    try {
      const response = await apiClient.chat(userMsg) as ChatResponse;
      setMessages(prev => [...prev, { 
        role: 'assistant', 
        content: response.answer,
        citations: response.citations 
      }]);
    } catch (err) {
      setMessages(prev => [...prev, { 
        role: 'assistant', 
        content: err instanceof Error ? `Error: ${err.message}` : 'Error: Failed to get response. Please try again.'
      }]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex flex-col h-full">
      <div className="p-6 border-b border-gray-200 bg-white">
        <h1 className="text-2xl font-semibold text-gray-900">AI Chat</h1>
      </div>
      
      <div className="flex-1 overflow-auto p-6 space-y-4">
        {messages.length === 0 && (
          <div className="text-center text-gray-500 mt-8">
            Ask a question about your documents
          </div>
        )}
        {messages.map((msg, i) => (
          <div key={i}>
            <div className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
              <div className={`max-w-2xl px-4 py-2 rounded-lg ${
                msg.role === 'user' ? 'bg-blue-600 text-white' : 'bg-white border border-gray-200'
              }`}>
                {msg.content}
              </div>
            </div>
            {msg.citations && msg.citations.length > 0 && (
              <div className="flex justify-start mt-2">
                <div className="max-w-2xl bg-gray-50 p-3 rounded-lg border border-gray-200">
                  <p className="text-xs font-medium text-gray-700 mb-1">Sources:</p>
                  {msg.citations.map((cit, idx) => (
                    <p key={idx} className="text-xs text-gray-600">
                      {cit.document_name || 'Document'} - Page {cit.page_number}
                    </p>
                  ))}
                </div>
              </div>
            )}
          </div>
        ))}
        {loading && (
          <div className="flex justify-start">
            <div className="bg-white border border-gray-200 px-4 py-2 rounded-lg">
              Thinking...
            </div>
          </div>
        )}
      </div>
      
      <div className="p-6 border-t border-gray-200 bg-white">
        <div className="flex gap-2">
          <input
            type="text"
            value={message}
            onChange={(e) => setMessage(e.target.value)}
            onKeyPress={(e) => e.key === 'Enter' && handleSend()}
            placeholder="Ask a question..."
            disabled={loading}
            className="flex-1 px-4 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:bg-gray-100"
          />
          <button
            onClick={handleSend}
            disabled={loading || !message.trim()}
            className="px-6 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 disabled:bg-gray-400"
          >
            Send
          </button>
        </div>
      </div>
    </div>
  );
}
