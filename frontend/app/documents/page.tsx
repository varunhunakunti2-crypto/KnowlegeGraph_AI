'use client';

import { useState, useEffect } from 'react';
import { apiClient } from '@/lib/api';
import { Document } from '@/types/api';

export default function DocumentsPage() {
  const [file, setFile] = useState<File | null>(null);
  const [documents, setDocuments] = useState<Document[]>([]);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fetchDocuments = async () => {
    try {
      const docs = await apiClient.getDocuments();
      setDocuments(docs as Document[]);
    } catch (err) {
      console.error(err);
    }
  };

  useEffect(() => {
    fetchDocuments();
    const interval = setInterval(fetchDocuments, 5000);
    return () => clearInterval(interval);
  }, []);

  const handleUpload = async () => {
    if (!file) return;
    setUploading(true);
    setError(null);
    try {
      await apiClient.uploadDocument(file);
      setFile(null);
      fetchDocuments();
    } catch (err: any) {
      setError(err.message || 'Upload failed');
    } finally {
      setUploading(false);
    }
  };

  const handleDelete = async (id: string) => {
    if (!confirm('Delete this document?')) return;
    try {
      await apiClient.deleteDocument(id);
      fetchDocuments();
    } catch (err) {
      alert('Delete failed');
    }
  };

  return (
    <div className="p-8">
      <h1 className="text-2xl font-semibold text-gray-900 mb-6">Documents</h1>
      
      <div className="bg-white p-6 rounded-lg border border-gray-200 mb-6">
        <h2 className="text-lg font-medium text-gray-900 mb-4">Upload PDF</h2>
        <div className="border-2 border-dashed border-gray-300 rounded-lg p-6 text-center">
          <input
            type="file"
            accept=".pdf"
            onChange={(e) => setFile(e.target.files?.[0] || null)}
            className="mb-4"
          />
          {file && <p className="text-sm text-gray-600 mb-2">{file.name}</p>}
          {error && <p className="text-sm text-red-600 mb-2">{error}</p>}
          <button
            onClick={handleUpload}
            disabled={!file || uploading}
            className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 disabled:bg-gray-400"
          >
            {uploading ? 'Uploading...' : 'Upload'}
          </button>
        </div>
      </div>

      <div className="bg-white rounded-lg border border-gray-200">
        <div className="px-6 py-4 border-b border-gray-200">
          <h2 className="text-lg font-medium text-gray-900">Document List</h2>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead className="bg-gray-50">
              <tr>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Name</th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Pages</th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Status</th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Size</th>
                <th className="px-6 py-3 text-right text-xs font-medium text-gray-500 uppercase">Actions</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-200">
              {documents.map((doc) => (
                <tr key={doc.document_id}>
                  <td className="px-6 py-4 text-sm text-gray-900">{doc.original_filename}</td>
                  <td className="px-6 py-4 text-sm text-gray-500">{doc.page_count}</td>
                  <td className="px-6 py-4 text-sm">
                    <span className={`px-2 py-1 rounded-full text-xs ${
                      doc.status === 'completed' ? 'bg-green-100 text-green-800' :
                      doc.status === 'failed' ? 'bg-red-100 text-red-800' :
                      'bg-yellow-100 text-yellow-800'
                    }`}>
                      {doc.status}
                    </span>
                  </td>
                  <td className="px-6 py-4 text-sm text-gray-500">{(doc.file_size / 1024).toFixed(1)} KB</td>
                  <td className="px-6 py-4 text-right text-sm">
                    <button onClick={() => handleDelete(doc.document_id)} className="text-red-600 hover:text-red-900">
                      Delete
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
          {documents.length === 0 && (
            <div className="p-6 text-center text-gray-500">No documents yet</div>
          )}
        </div>
      </div>
    </div>
  );
}
