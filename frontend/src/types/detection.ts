export type DetectionSource = 'local' | 'openai'

export interface DetectedObject {
  label: string
  confidence: number | null
  /** [x1, y1, x2, y2] in image pixels; set for local detections only. */
  box?: number[] | null
}

/** Server-side durations in milliseconds. */
export interface DetectionTiming {
  local_inference_ms: number | null
  local_processing_ms: number | null
  openai_ms: number | null
  total_ms: number
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
  /** Optional: older backends don't send it. */
  timing?: DetectionTiming | null
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
