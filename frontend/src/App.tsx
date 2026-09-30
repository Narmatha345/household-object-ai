import { useCallback, useEffect, useRef, useState, type KeyboardEvent } from 'react'
import AppHeader from './components/AppHeader'
import CameraCapture from './components/CameraCapture'
import DetectionResult from './components/DetectionResult'
import { AlertIcon, CameraIcon, UploadCloudIcon } from './components/Icons'
import ImageUpload from './components/ImageUpload'
import LoadingState, { type LoadingPhase } from './components/LoadingState'
import PipelineCard from './components/PipelineCard'
import StatsCard from './components/StatsCard'
import { detectObject, getHealth, getStats, toFriendlyError } from './services/api'
import type { DetectionResponse, HealthResponse, StatsResponse } from './types/detection'

type InputMode = 'camera' | 'upload'

const TABS: Array<{ id: InputMode; label: string; icon: typeof CameraIcon }> = [
  { id: 'camera', label: 'Camera', icon: CameraIcon },
  { id: 'upload', label: 'Upload Photo', icon: UploadCloudIcon },
]

// Local YOLO answers in well under a second. A request still running after this
// long is almost certainly waiting on the OpenAI fallback.
const FALLBACK_HINT_DELAY_MS = 2000

export default function App() {
  const [mode, setMode] = useState<InputMode>('upload')
  const [loading, setLoading] = useState(false)
  const [loadingPhase, setLoadingPhase] = useState<LoadingPhase>('local')
  const [result, setResult] = useState<DetectionResponse | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [analyzedPreview, setAnalyzedPreview] = useState<string | null>(null)
  const [stats, setStats] = useState<StatsResponse | null>(null)
  const [health, setHealth] = useState<HealthResponse | null>(null)
  const [backendDown, setBackendDown] = useState(false)
  const [threshold, setThreshold] = useState<number | null>(null)
  const tabRefs = useRef<Record<InputMode, HTMLButtonElement | null>>({ camera: null, upload: null })

  const refreshStatus = useCallback(async () => {
    try {
      const [healthData, statsData] = await Promise.all([getHealth(), getStats()])
      setHealth(healthData)
      setStats(statsData)
      setBackendDown(false)
    } catch {
      setBackendDown(true)
    }
  }, [])

  useEffect(() => {
    void refreshStatus()
  }, [refreshStatus])

  // The result panel owns its own preview URL so it survives input tab switches.
  useEffect(() => {
    return () => {
      if (analyzedPreview) URL.revokeObjectURL(analyzedPreview)
    }
  }, [analyzedPreview])

  const runDetection = async (image: Blob, filename: string) => {
    setLoading(true)
    setLoadingPhase('local')
    setError(null)
    setResult(null)
    setAnalyzedPreview(URL.createObjectURL(image))
    // Without a configured fallback a slow request can only be a slow local model.
    const slowPhase: LoadingPhase = health?.openai_fallback_configured ? 'fallback' : 'slow'
    const fallbackHint = window.setTimeout(() => setLoadingPhase(slowPhase), FALLBACK_HINT_DELAY_MS)
    try {
      const response = await detectObject(image, filename)
      setResult(response)
      setThreshold(response.confidence_threshold)
    } catch (err) {
      setError(toFriendlyError(err))
    } finally {
      window.clearTimeout(fallbackHint)
      setLoading(false)
      void refreshStatus()
    }
  }

  const onTabKeyDown = (event: KeyboardEvent<HTMLDivElement>) => {
    if (event.key !== 'ArrowLeft' && event.key !== 'ArrowRight') return
    event.preventDefault()
    const next: InputMode = mode === 'camera' ? 'upload' : 'camera'
    setMode(next)
    tabRefs.current[next]?.focus()
  }

  const thresholdPercent = threshold === null ? null : Math.round(threshold * 100)

  return (
    <div className="min-h-screen">
      <AppHeader
        health={health}
        backendDown={backendDown}
        confidenceThreshold={threshold}
        onRefreshStatus={() => void refreshStatus()}
      />

      <div className="mx-auto grid max-w-7xl gap-6 px-4 py-6 sm:px-6 sm:py-8 lg:grid-cols-[minmax(0,1.7fr)_minmax(0,1fr)] lg:px-8">
        <main className="flex min-w-0 flex-col gap-6">
          <section className="card p-4 sm:p-6" aria-label="Detection workspace">
            <div
              role="tablist"
              aria-label="Image input method"
              onKeyDown={onTabKeyDown}
              className="mb-5 grid grid-cols-2 gap-1 rounded-lg border border-slate-200 bg-slate-50 p-1"
            >
              {TABS.map(({ id, label, icon: Icon }) => {
                const selected = mode === id
                return (
                  <button
                    key={id}
                    ref={(el) => {
                      tabRefs.current[id] = el
                    }}
                    type="button"
                    role="tab"
                    id={`tab-${id}`}
                    aria-selected={selected}
                    aria-controls={`panel-${id}`}
                    tabIndex={selected ? 0 : -1}
                    onClick={() => setMode(id)}
                    className={`focus-ring flex items-center justify-center gap-2 rounded-md px-3 py-2.5 text-sm font-semibold transition-colors ${
                      selected
                        ? 'bg-white text-blue-700 shadow-sm ring-1 ring-blue-200'
                        : 'text-slate-500 hover:bg-white/60 hover:text-slate-800'
                    }`}
                  >
                    <Icon className="h-4 w-4" />
                    {label}
                  </button>
                )
              })}
            </div>

            <div role="tabpanel" id={`panel-${mode}`} aria-labelledby={`tab-${mode}`}>
              {mode === 'camera' ? (
                <CameraCapture disabled={loading} onCapture={(blob) => runDetection(blob, 'camera-capture.jpg')} />
              ) : (
                <ImageUpload disabled={loading} onDetect={(file) => runDetection(file, file.name)} />
              )}
            </div>
          </section>

          {(loading || result || error) && (
            <section className="card p-4 sm:p-6" aria-live="polite" aria-label="Detection result">
              <div className="flex flex-col gap-5 sm:flex-row sm:gap-6">
                {analyzedPreview && (
                  <img
                    src={analyzedPreview}
                    alt="Image that was analyzed"
                    className="h-44 w-full shrink-0 rounded-lg border border-slate-200 bg-slate-50 object-cover sm:h-36 sm:w-48"
                  />
                )}
                <div className="min-w-0 flex-1">
                  {loading && <LoadingState phase={loadingPhase} />}
                  {error && (
                    <div role="alert" className="flex items-start gap-3 rounded-lg border border-red-200 bg-red-50 px-4 py-3">
                      <AlertIcon className="mt-0.5 h-5 w-5 shrink-0 text-red-600" />
                      <div>
                        <p className="text-sm font-semibold text-red-800">Detection failed</p>
                        <p className="mt-0.5 text-sm text-red-700">{error}</p>
                      </div>
                    </div>
                  )}
                  {result && <DetectionResult result={result} />}
                </div>
              </div>
            </section>
          )}
        </main>

        <aside className="flex min-w-0 flex-col gap-6" aria-label="Statistics and pipeline">
          <StatsCard stats={stats} />
          <PipelineCard thresholdPercent={thresholdPercent} lastSource={result?.source ?? null} />
        </aside>
      </div>
    </div>
  )
}
