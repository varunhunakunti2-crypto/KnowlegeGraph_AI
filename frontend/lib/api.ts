import { ChatResponse, Document, GraphResponse, HealthResponse, ModelStatus } from '@/types/api';

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

class ApiClient {
  private baseUrl: string;

  constructor(baseUrl: string) {
    this.baseUrl = baseUrl;
  }

  private async request<T>(endpoint: string, options?: RequestInit): Promise<T> {
    const response = await fetch(`${this.baseUrl}${endpoint}`, {
      ...options,
      headers: {
        'Content-Type': 'application/json',
        ...options?.headers,
      },
    });

    if (!response.ok) {
      throw new Error(await this.errorMessage(response, 'API error'));
    }

    return response.json();
  }

  private async errorMessage(response: Response, prefix: string): Promise<string> {
    let detail = '';
    try {
      const payload = await response.json();
      if (typeof payload?.detail === 'string') {
        detail = payload.detail;
      } else if (payload?.detail) {
        detail = JSON.stringify(payload.detail);
      }
    } catch {
      // The response may not be JSON; the status code is still useful.
    }
    return detail
      ? `${prefix}: ${response.status} - ${detail}`
      : `${prefix}: ${response.status}`;
  }

  async getHealth(): Promise<HealthResponse> {
    return this.request<HealthResponse>('/api/v1/health');
  }

  async getReady(): Promise<{ status: string; ready: boolean; dependencies: Record<string, unknown> }> {
    return this.request('/api/v1/health/ready');
  }

  async getDocuments(): Promise<Document[]> {
    return this.request<Document[]>('/api/v1/documents/');
  }

  async uploadDocument(file: File): Promise<{
    document_id: string;
    filename: string;
    status: string;
    message: string;
  }> {
    const formData = new FormData();
    formData.append('file', file);
    
    const response = await fetch(`${this.baseUrl}/api/v1/documents/upload`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      throw new Error(await this.errorMessage(response, 'Upload failed'));
    }

    return response.json();
  }

  async deleteDocument(id: string) {
    return this.request(`/api/v1/documents/${id}`, { method: 'DELETE' });
  }

  async chat(question: string, documentId?: string): Promise<ChatResponse> {
    return this.request<ChatResponse>('/api/v1/chat/', {
      method: 'POST',
      body: JSON.stringify({ question, document_id: documentId }),
    });
  }

  async getGraph(documentId?: string): Promise<GraphResponse> {
    const query = documentId ? `?document_id=${documentId}` : '';
    return this.request<GraphResponse>(`/api/v1/graph/${query}`);
  }

  async getModelStatus(): Promise<ModelStatus> {
    return this.request<ModelStatus>('/api/v1/models/status');
  }
}

export const apiClient = new ApiClient(API_BASE_URL);
