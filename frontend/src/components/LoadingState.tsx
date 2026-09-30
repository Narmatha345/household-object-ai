export type LoadingPhase = 'local' | 'fallback' | 'slow'

interface LoadingStateProps {
  phase?: LoadingPhase
}

const COPY: Record<LoadingPhase, { title: string; detail: string }> = {
  local: {
    title: 'Analyzing image...',
    detail: 'Local ML is checking the image first.',
  },
  fallback: {
    title: 'Checking with AI fallback...',
    detail: 'Local model could not confidently identify the object.',
  },
  slow: {
    title: 'Still analyzing...',
    detail: 'Local ML is taking longer than usual on this server. Please keep this page open.',
  },
}

export default function LoadingState({ phase = 'local' }: LoadingStateProps) {
  const copy = COPY[phase]
  const spinnerColor = phase === 'fallback' ? 'border-t-violet-600' : 'border-t-blue-600'

  return (
    <div role="status" aria-live="polite" className="flex items-start gap-4 py-2">
      <span
        className={`mt-0.5 h-6 w-6 shrink-0 animate-spin rounded-full border-2 border-slate-200 ${spinnerColor}`}
        aria-hidden="true"
      />
      <div>
        <p className="font-semibold text-slate-900">{copy.title}</p>
        <p className="mt-1 text-sm text-slate-500">{copy.detail}</p>
      </div>
    </div>
  )
}
