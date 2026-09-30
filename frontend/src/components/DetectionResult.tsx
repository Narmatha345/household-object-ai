import type { ReactNode } from 'react'
import type { DetectedObject, DetectionResponse } from '../types/detection'

interface DetectionResultProps {
  result: DetectionResponse
}

const capitalize = (text: string) => text.charAt(0).toUpperCase() + text.slice(1)

const formatConfidence = (value: number | null) =>
  value === null ? 'Not available' : `${Math.round(value * 100)}%`

function Field({ label, children }: { label: string; children: ReactNode }) {
  return (
    <div>
      <dt className="text-xs font-medium uppercase tracking-wide text-slate-500">{label}</dt>
      <dd className="mt-1 text-slate-900">{children}</dd>
    </div>
  )
}

function ObjectList({ objects }: { objects: DetectedObject[] }) {
  return (
    <ul className="divide-y divide-slate-100 rounded-lg border border-slate-200">
      {objects.map((obj) => (
        <li key={obj.label} className="flex items-center justify-between px-3 py-2 text-sm">
          <span>{capitalize(obj.label)}</span>
          <span className="tabular-nums text-slate-500">{formatConfidence(obj.confidence)}</span>
        </li>
      ))}
    </ul>
  )
}

export default function DetectionResult({ result }: DetectionResultProps) {
  const primary = result.objects[0]
  const isLocal = result.source === 'local'

  return (
    <div className="space-y-5">
      <dl className="grid gap-4 sm:grid-cols-2">
        <Field label="Detected object">
          <span className="text-2xl font-semibold">
            {primary ? capitalize(primary.label) : 'No household object found'}
          </span>
        </Field>
        <Field label="Confidence">
          <span className="text-2xl font-semibold tabular-nums">
            {primary ? formatConfidence(primary.confidence) : '—'}
          </span>
        </Field>
        <Field label="Source">
          <span
            className={`inline-flex items-center rounded-full px-2.5 py-0.5 text-sm font-medium ${
              isLocal ? 'bg-teal-100 text-teal-800' : 'bg-violet-100 text-violet-800'
            }`}
          >
            {isLocal ? 'Local ML' : 'OpenAI fallback'}
          </span>
        </Field>
        <Field label="OpenAI fallback">
          {result.fallback_used ? (
            <span>Used. {result.fallback_reason}</span>
          ) : (
            <span>Not used</span>
          )}
        </Field>
      </dl>

      {result.description && (
        <p className="rounded-lg bg-slate-50 px-3 py-2 text-sm text-slate-700">{result.description}</p>
      )}
      {result.reliability_note && (
        <p className="text-xs text-slate-500">Reliability: {result.reliability_note}</p>
      )}

      {result.objects.length > 1 && (
        <div className="space-y-2">
          <h3 className="text-sm font-semibold text-slate-700">All detected objects</h3>
          <ObjectList objects={result.objects} />
        </div>
      )}

      {result.local_candidates.length > 0 && (
        <div className="space-y-2">
          <h3 className="text-sm font-semibold text-slate-700">
            Local model guesses (below {Math.round(result.confidence_threshold * 100)}% threshold)
          </h3>
          <ObjectList objects={result.local_candidates} />
        </div>
      )}
    </div>
  )
}
