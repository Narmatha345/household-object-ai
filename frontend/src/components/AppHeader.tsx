import { useEffect, useId, useRef, useState } from 'react'
import type { HealthResponse } from '../types/detection'
import { ChevronDownIcon, ExternalLinkIcon, HomeIcon, RefreshIcon, SettingsIcon } from './Icons'

interface AppHeaderProps {
  health: HealthResponse | null
  backendDown: boolean
  /** Known once a detection has returned; the server owns the real value. */
  confidenceThreshold: number | null
  onRefreshStatus: () => void
}

type Tone = 'success' | 'error' | 'warning' | 'fallback' | 'neutral'

const TONE_CLASSES: Record<Tone, { badge: string; dot: string }> = {
  success: { badge: 'bg-emerald-50 text-emerald-700 ring-1 ring-inset ring-emerald-200', dot: 'bg-emerald-500' },
  error: { badge: 'bg-red-50 text-red-700 ring-1 ring-inset ring-red-200', dot: 'bg-red-500' },
  warning: { badge: 'bg-amber-50 text-amber-800 ring-1 ring-inset ring-amber-200', dot: 'bg-amber-500' },
  fallback: { badge: 'bg-violet-50 text-violet-700 ring-1 ring-inset ring-violet-200', dot: 'bg-violet-500' },
  neutral: { badge: 'bg-slate-100 text-slate-600 ring-1 ring-inset ring-slate-200', dot: 'bg-slate-400' },
}

interface StatusItem {
  label: string
  tone: Tone
}

function getStatusItems(health: HealthResponse | null, backendDown: boolean): StatusItem[] {
  if (backendDown) {
    return [{ label: 'Local ML Unavailable', tone: 'error' }]
  }
  if (!health) {
    return [{ label: 'Checking status…', tone: 'neutral' }]
  }
  return [
    health.local_model_loaded
      ? { label: 'Local ML Active', tone: 'success' }
      : { label: 'Local ML Unavailable', tone: 'error' },
    health.openai_fallback_configured
      ? { label: 'OpenAI Fallback Ready', tone: 'fallback' }
      : { label: 'OpenAI Fallback Not Configured', tone: 'warning' },
  ]
}

function StatusBadge({ label, tone }: StatusItem) {
  const classes = TONE_CLASSES[tone]
  return (
    <span className={`badge ${classes.badge}`}>
      <span className={`h-1.5 w-1.5 rounded-full ${classes.dot}`} aria-hidden="true" />
      {label}
    </span>
  )
}

function SettingsMenu({ health, backendDown, confidenceThreshold, onRefreshStatus }: AppHeaderProps) {
  const [open, setOpen] = useState(false)
  const containerRef = useRef<HTMLDivElement>(null)
  const buttonRef = useRef<HTMLButtonElement>(null)
  const panelId = useId()

  useEffect(() => {
    if (!open) return
    const onPointerDown = (event: PointerEvent) => {
      if (!containerRef.current?.contains(event.target as Node)) setOpen(false)
    }
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        setOpen(false)
        buttonRef.current?.focus()
      }
    }
    document.addEventListener('pointerdown', onPointerDown)
    document.addEventListener('keydown', onKeyDown)
    return () => {
      document.removeEventListener('pointerdown', onPointerDown)
      document.removeEventListener('keydown', onKeyDown)
    }
  }, [open])

  const apiDocsUrl = `${import.meta.env.VITE_API_BASE_URL || ''}/docs`
  const rows: Array<[string, string]> = [
    ['Backend', backendDown ? 'Unreachable' : health ? 'Connected' : 'Checking…'],
    ['Local model', health?.local_model_loaded ? 'Loaded (YOLO)' : 'Not loaded'],
    ['OpenAI fallback', health?.openai_fallback_configured ? 'Configured' : 'Not configured'],
    [
      'Confidence threshold',
      confidenceThreshold === null ? 'Set on server' : `${Math.round(confidenceThreshold * 100)}%`,
    ],
  ]

  return (
    <div ref={containerRef} className="relative">
      <button
        ref={buttonRef}
        type="button"
        onClick={() => setOpen((value) => !value)}
        aria-expanded={open}
        aria-controls={panelId}
        className="btn-secondary px-3 py-2"
      >
        <SettingsIcon className="h-4 w-4 text-slate-500" />
        <span className="hidden sm:inline">Settings</span>
        <ChevronDownIcon className={`h-4 w-4 text-slate-400 transition-transform ${open ? 'rotate-180' : ''}`} />
      </button>

      {open && (
        <div
          id={panelId}
          role="dialog"
          aria-label="System settings"
          className="card absolute right-0 z-20 mt-2 w-[min(20rem,calc(100vw-2rem))] p-4 shadow-lg shadow-slate-900/5"
        >
          <h2 className="text-sm font-semibold text-slate-900">System status</h2>
          <dl className="mt-3 space-y-2 text-sm">
            {rows.map(([label, value]) => (
              <div key={label} className="flex items-center justify-between gap-4">
                <dt className="text-slate-500">{label}</dt>
                <dd className="font-medium text-slate-800">{value}</dd>
              </div>
            ))}
          </dl>
          <p className="mt-3 text-xs leading-relaxed text-slate-500">
            Model, threshold and API key are configured in <code className="text-slate-700">backend/.env</code>.
          </p>
          <div className="mt-4 flex gap-2 border-t border-slate-100 pt-4">
            <button
              type="button"
              onClick={() => {
                onRefreshStatus()
                setOpen(false)
              }}
              className="btn-secondary flex-1 whitespace-nowrap px-3 py-2 text-xs"
            >
              <RefreshIcon className="h-3.5 w-3.5" />
              Refresh status
            </button>
            <a href={apiDocsUrl} target="_blank" rel="noreferrer" className="btn-secondary flex-1 px-3 py-2 text-xs">
              API docs
              <ExternalLinkIcon className="h-3.5 w-3.5" />
            </a>
          </div>
        </div>
      )}
    </div>
  )
}

export default function AppHeader(props: AppHeaderProps) {
  const statusItems = getStatusItems(props.health, props.backendDown)

  return (
    <header className="sticky top-0 z-10 border-b border-slate-200/80 bg-white/95 backdrop-blur">
      <div className="mx-auto flex max-w-7xl items-center justify-between gap-4 px-4 py-3 sm:px-6 lg:px-8">
        <div className="flex min-w-0 items-center gap-3">
          <span className="flex h-10 w-10 shrink-0 items-center justify-center rounded-lg bg-blue-50 text-blue-600 ring-1 ring-inset ring-blue-100">
            <HomeIcon className="h-5 w-5" />
          </span>
          <div className="flex min-w-0 flex-col md:flex-row md:items-center md:gap-4">
            <h1 className="truncate text-lg font-bold tracking-tight text-slate-900 sm:text-xl">Household Object AI</h1>
            <span className="hidden h-5 w-px bg-slate-200 md:block" aria-hidden="true" />
            <p className="text-xs text-slate-500 sm:truncate sm:text-sm">Local-first object detection with AI fallback</p>
          </div>
        </div>

        <div className="flex shrink-0 items-center gap-3">
          <div className="hidden items-center gap-2 lg:flex" role="status" aria-label="System status">
            {statusItems.map((item) => (
              <StatusBadge key={item.label} {...item} />
            ))}
          </div>
          <SettingsMenu {...props} />
        </div>
      </div>

      {/* Compact status row for tablet and mobile. */}
      <div
        className="mx-auto flex max-w-7xl flex-wrap gap-2 px-4 pb-3 sm:px-6 lg:hidden"
        role="status"
        aria-label="System status"
      >
        {statusItems.map((item) => (
          <StatusBadge key={item.label} {...item} />
        ))}
      </div>
    </header>
  )
}
