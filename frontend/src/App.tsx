import { useCallback, useEffect, useState } from 'react'
import CameraCapture from './components/CameraCapture'
import DetectionResult from './components/DetectionResult'
import ImageUpload from './components/ImageUpload'
import LoadingState from './components/LoadingState'
import StatsCard from './components/StatsCard'
import { detectObject, getHealth, getStats, toFriendlyError } from './services/api'
import type { DetectionResponse, HealthResponse, StatsResponse } from './types/detection'

type InputMode = 'camera' | 'upload'

export default function App() {
  const [mode, setMode] = useState<InputMode>('upload')
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState<DetectionResponse | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [analyzedPreview, setAnalyzedPreview] = useState<string | null>(null)
  const [stats, setStats] = useState<StatsResponse | null>(null)
  const [health, setHealth] = useState<HealthResponse | null>(null)
  const [backendDown, setBackendDown] = useState(false)

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
    setError(null)
    setResult(null)
    setAnalyzedPreview(URL.createObjectURL(image))
    try {
      setResult(await detectObject(image, filename))
    } catch (err) {
      setError(toFriendlyError(err))
    } finally {
      setLoading(false)
      void refreshStatus()
    }
  }

  const tabClass = (tab: InputMode) =>
    `flex-1 rounded-lg px-4 py-2 text-sm font-medium transition ${
      mode === tab ? 'bg-white text-slate-900 shadow-sm' : 'text-slate-600 hover:text-slate-900'
    }`

  return (
    <div className="mx-auto max-w-5xl px-4 py-10">
      <header className="mb-8">
        <h1 className="text-3xl font-bold tracking-tight">Household Object AI</h1>
        <p className="mt-1 text-slate-600">Local-first object detection with AI fallback</p>
        <div className="mt-3 flex flex-wrap gap-2 text-xs">
          {backendDown ? (
            <span className="rounded-full bg-red-100 px-2.5 py-1 font-medium text-red-700">Backend unavailable</span>
          ) : health ? (
            <>
              <span
                className={`rounded-full px-2.5 py-1 font-medium ${
                  health.local_model_loaded ? 'bg-teal-100 text-teal-800' : 'bg-amber-100 text-amber-800'
                }`}
              >
                Local model {health.local_model_loaded ? 'loaded' : 'not loaded'}
              </span>
              <span
                className={`rounded-full px-2.5 py-1 font-medium ${
                  health.openai_fallback_configured ? 'bg-slate-200 text-slate-700' : 'bg-amber-100 text-amber-800'
                }`}
              >
                OpenAI fallback {health.openai_fallback_configured ? 'configured' : 'not configured'}
              </span>
            </>
          ) : null}
        </div>
      </header>

      <div className="grid gap-6 lg:grid-cols-[1fr_320px]">
        <main className="space-y-6">
          <section className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
            <div className="mb-5 flex gap-1 rounded-xl bg-slate-100 p-1" role="tablist">
              <button type="button" role="tab" aria-selected={mode === 'camera'} className={tabClass('camera')} onClick={() => setMode('camera')}>
                Camera
              </button>
              <button type="button" role="tab" aria-selected={mode === 'upload'} className={tabClass('upload')} onClick={() => setMode('upload')}>
                Upload Photo
              </button>
            </div>

            {mode === 'camera' ? (
              <CameraCapture disabled={loading} onCapture={(blob) => runDetection(blob, 'camera-capture.jpg')} />
            ) : (
              <ImageUpload disabled={loading} onDetect={(file) => runDetection(file, file.name)} />
            )}
          </section>

          {(loading || result || error) && (
            <section className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm" aria-live="polite">
              <h2 className="mb-4 text-lg font-semibold">Result</h2>
              <div className="flex flex-col gap-5 sm:flex-row">
                {analyzedPreview && (
                  <img
                    src={analyzedPreview}
                    alt="Analyzed image"
                    className="h-32 w-full rounded-lg object-cover sm:w-44"
                  />
                )}
                <div className="flex-1">
                  {loading && <LoadingState />}
                  {error && (
                    <p role="alert" className="rounded-lg bg-red-50 px-3 py-2 text-sm text-red-700">
                      {error}
                    </p>
                  )}
                  {result && <DetectionResult result={result} />}
                </div>
              </div>
            </section>
          )}
        </main>

        <aside className="space-y-6">
          <StatsCard stats={stats} />
          <section className="rounded-2xl border border-slate-200 bg-white p-5 text-sm text-slate-600 shadow-sm">
            <h2 className="mb-2 font-semibold text-slate-700">How it works</h2>
            <ol className="list-decimal space-y-1 pl-5">
              <li>Your image is analyzed by the local YOLO model first.</li>
              <li>
                If confidence is at least {Math.round((result?.confidence_threshold ?? 0.7) * 100)}%, the local
                result is returned.
              </li>
              <li>Otherwise the image is sent to OpenAI as a fallback.</li>
            </ol>
          </section>
        </aside>
      </div>
    </div>
  )
}
