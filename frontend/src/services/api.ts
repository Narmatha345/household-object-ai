import axios, { AxiosError } from 'axios'
import type {
  ApiErrorBody,
  DetectionResponse,
  HealthResponse,
  StatsResponse,
} from '../types/detection'

const client = axios.create({
  baseURL: import.meta.env.VITE_API_BASE_URL || '',
  // Generous: CPU-only hosts (e.g. Render free tier) can take ~1 minute per image.
  timeout: 150_000,
})

export async function detectObject(image: Blob, filename: string): Promise<DetectionResponse> {
  const form = new FormData()
  form.append('file', image, filename)
  const { data } = await client.post<DetectionResponse>('/api/detect', form)
  return data
}

export async function getHealth(): Promise<HealthResponse> {
  const { data } = await client.get<HealthResponse>('/api/health')
  return data
}

export async function getStats(): Promise<StatsResponse> {
  const { data } = await client.get<StatsResponse>('/api/stats')
  return data
}

/** Convert any API/network failure into a message that is safe to show users. */
export function toFriendlyError(error: unknown): string {
  if (error instanceof AxiosError) {
    if (error.code === 'ECONNABORTED') {
      return 'The request timed out. Please try again.'
    }
    if (!error.response) {
      return 'Cannot reach the detection server. Make sure the backend is running on port 8000.'
    }
    // The Vite dev proxy returns 5xx with no JSON body when the backend is down.
    const body = error.response.data as Partial<ApiErrorBody> | undefined
    if (body && typeof body.detail === 'string') {
      return body.detail
    }
    if (error.response.status >= 500) {
      return 'The detection server is unavailable. Please try again shortly.'
    }
  }
  return 'Something went wrong. Please try again.'
}
