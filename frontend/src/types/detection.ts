export type DetectionSource = 'local' | 'openai'

export interface DetectedObject {
  label: string
  confidence: number | null
}

export interface DetectionResponse {
  source: DetectionSource
  objects: DetectedObject[]
  fallback_used: boolean
  fallback_reason: string | null
  description: string | null
  reliability_note: string | null
  local_candidates: DetectedObject[]
  confidence_threshold: number
}

export interface HealthResponse {
  status: 'ok'
  local_model_loaded: boolean
  openai_fallback_configured: boolean
}

export interface StatsResponse {
  total_requests: number
  local_detections: number
  openai_fallbacks: number
  failed_requests: number
  fallback_rate: number
}

export interface ApiErrorBody {
  detail: string
  code: string
}
