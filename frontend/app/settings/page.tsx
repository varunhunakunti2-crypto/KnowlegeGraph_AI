'use client';

import { useState, useEffect } from 'react';
import { apiClient } from '@/lib/api';
import { HealthResponse, ModelStatus } from '@/types/api';

export default function SettingsPage() {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [models, setModels] = useState<ModelStatus | null>(null);

  useEffect(() => {
    const fetchStatus = async () => {
      try {
        const [h, m] = await Promise.all([
          apiClient.getHealth(),
          apiClient.getModelStatus()
        ]);
        setHealth(h);
        setModels(m);
      } catch (err) {
        console.error(err);
      }
    };
    fetchStatus();
  }, []);

  return (
    <div className="p-8">
      <h1 className="text-2xl font-semibold text-gray-900 mb-6">Settings</h1>
      
      <div className="bg-white rounded-lg border border-gray-200 p-6 mb-6">
        <h2 className="text-lg font-medium text-gray-900 mb-4">Backend Health</h2>
        <div className="space-y-2">
          <div className="flex justify-between py-2">
            <span className="text-gray-600">API Status</span>
            <span className={health?.status === 'ok' ? 'text-green-600' : 'text-gray-500'}>
              {health?.status || 'Not checked'}
            </span>
          </div>
        </div>
      </div>

      <div className="bg-white rounded-lg border border-gray-200 p-6">
        <h2 className="text-lg font-medium text-gray-900 mb-4">Local Models</h2>
        <div className="space-y-2">
          <div className="flex justify-between py-2">
            <span className="text-gray-600">LLM (qwen3:4b)</span>
            <span className={models?.llm?.available ? 'text-green-600' : 'text-red-600'}>
              {models?.llm?.available ? 'Available' : 'Not available'}
            </span>
          </div>
          <div className="flex justify-between py-2">
            <span className="text-gray-600">Embeddings (nomic-embed-text)</span>
            <span className={models?.embeddings?.available ? 'text-green-600' : 'text-red-600'}>
              {models?.embeddings?.available ? 'Available' : 'Not available'}
            </span>
          </div>
        </div>
      </div>
    </div>
  );
}
