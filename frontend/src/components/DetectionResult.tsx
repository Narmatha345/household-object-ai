import type { ReactNode } from 'react'
import type { DetectedObject, DetectionResponse } from '../types/detection'
import { ChipIcon, CloudIcon, InfoIcon } from './Icons'

interface DetectionResultProps {
  result: DetectionResponse
}

const capitalize = (text: string) => text.charAt(0).toUpperCase() + text.slice(1)

const formatConfidence = (value: number | null) => (value === null ? 'N/A' : `${Math.round(value * 100)}%`)

function Metric({ label, children }: { label: string; children: ReactNode }) {
  return (
    <div className="min-w-0">
      <dt className="text-xs font-medium uppercase tracking-wide text-slate-500">{label}</dt>
      <dd className="mt-1.5">{children}</dd>
    </div>
  )
}

function ObjectList({ title, objects, barClass }: { title: string; objects: DetectedObject[]; barClass: string }) {
  return (
    <div>
      <h3 className="text-sm font-semibold text-slate-700">{title}</h3>
      <ul className="mt-2 divide-y divide-slate-100 rounded-lg border border-slate-200">
        {objects.map((obj) => (
          <li key={obj.label} className="flex items-center gap-4 px-4 py-2.5 text-sm">
            <span className="min-w-0 flex-1 truncate text-slate-800">{capitalize(obj.label)}</span>
            {obj.confidence !== null && (
              <span className="hidden h-1.5 w-24 overflow-hidden rounded-full bg-slate-100 sm:block" aria-hidden="true">
                <span className={`block h-full rounded-full ${barClass}`} style={{ width: `${obj.confidence * 100}%` }} />
              </span>
            )}
            <span className="w-10 text-right font-medium tabular-nums text-slate-600">{formatConfidence(obj.confidence)}</span>
          </li>
        ))}
      </ul>
    </div>
  )
}

export default function DetectionResult({ result }: DetectionResultProps) {
  const primary = result.objects[0]
  const isLocal = result.source === 'local'
  const threshold = Math.round(result.confidence_threshold * 100)

  return (
    <div className="flex flex-col gap-5">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <h2 className="text-base font-semibold text-slate-900">Detection result</h2>
        <span
          className={`badge uppercase tracking-wide ${
            isLocal
              ? 'bg-emerald-50 text-emerald-700 ring-1 ring-inset ring-emerald-200'
              : 'bg-violet-50 text-violet-700 ring-1 ring-inset ring-violet-200'
          }`}
        >
          {isLocal ? <ChipIcon className="h-3.5 w-3.5" /> : <CloudIcon className="h-3.5 w-3.5" />}
          {isLocal ? 'Local detection' : 'OpenAI fallback'}
        </span>
      </div>

      <dl className="grid grid-cols-2 gap-x-6 gap-y-5 sm:grid-cols-3">
        <div className="col-span-2 sm:col-span-1">
          <Metric label="Detected object">
            <span className="block truncate text-2xl font-bold tracking-tight text-slate-900">
              {primary ? capitalize(primary.label) : 'No household object found'}
            </span>
          </Metric>
        </div>
        <Metric label="Confidence">
          <span className="text-2xl font-bold tabular-nums tracking-tight text-slate-900">
            {primary ? formatConfidence(primary.confidence) : '—'}
          </span>
        </Metric>
        <Metric label="Source">
          <span className="inline-flex items-center gap-2 text-base font-semibold text-slate-900">
            <span
              className={`flex h-7 w-7 items-center justify-center rounded-md ${
                isLocal ? 'bg-emerald-50 text-emerald-600' : 'bg-violet-50 text-violet-600'
              }`}
            >
              {isLocal ? <ChipIcon className="h-4 w-4" /> : <CloudIcon className="h-4 w-4" />}
            </span>
            {isLocal ? 'Local ML' : 'OpenAI Fallback'}
          </span>
        </Metric>
      </dl>

      <div
        className={`flex items-start gap-3 rounded-lg border px-4 py-3 text-sm ${
          isLocal ? 'border-emerald-100 bg-emerald-50/60 text-emerald-900' : 'border-violet-100 bg-violet-50/60 text-violet-900'
        }`}
      >
        <InfoIcon className={`mt-0.5 h-4 w-4 shrink-0 ${isLocal ? 'text-emerald-600' : 'text-violet-600'}`} />
        <p>
          {isLocal ? (
            <>Detected on this device with at least {threshold}% confidence. OpenAI was not called.</>
          ) : (
            <>
              Local model confidence was below the configured threshold.
              {result.fallback_reason && <span className="block text-violet-800/80">{result.fallback_reason}</span>}
            </>
          )}
        </p>
      </div>

      {(result.description || result.reliability_note) && (
        <div className="space-y-1.5 text-sm">
          {result.description && <p className="text-slate-700">{result.description}</p>}
          {result.reliability_note && (
            <p className="text-xs text-slate-500">
              <span className="font-medium text-slate-600">Reliability:</span> {result.reliability_note}
            </p>
          )}
        </div>
      )}

      {result.objects.length > 1 && (
        <ObjectList
          title="All detected objects"
          objects={result.objects}
          barClass={isLocal ? 'bg-emerald-500' : 'bg-violet-500'}
        />
      )}

      {result.local_candidates.length > 0 && (
        <ObjectList
          title={`Local model guesses (below ${threshold}% threshold)`}
          objects={result.local_candidates}
          barClass="bg-slate-400"
        />
      )}
    </div>
  )
}
