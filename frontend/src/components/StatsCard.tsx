import type { StatsResponse } from '../types/detection'

interface StatsCardProps {
  stats: StatsResponse | null
}

export default function StatsCard({ stats }: StatsCardProps) {
  const rows: Array<[string, string]> = [
    ['Total requests', String(stats?.total_requests ?? 0)],
    ['Local detections', String(stats?.local_detections ?? 0)],
    ['OpenAI fallbacks', String(stats?.openai_fallbacks ?? 0)],
    ['Fallback rate', `${Math.round((stats?.fallback_rate ?? 0) * 100)}%`],
  ]

  return (
    <section className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
      <h2 className="text-sm font-semibold text-slate-700">Statistics</h2>
      <dl className="mt-3 grid grid-cols-2 gap-3">
        {rows.map(([label, value]) => (
          <div key={label} className="rounded-lg bg-slate-50 px-3 py-2">
            <dt className="text-xs text-slate-500">{label}</dt>
            <dd className="text-xl font-semibold tabular-nums">{value}</dd>
          </div>
        ))}
      </dl>
      {stats && stats.failed_requests > 0 && (
        <p className="mt-3 text-xs text-slate-500">Failed requests: {stats.failed_requests}</p>
      )}
      <p className="mt-3 text-xs text-slate-400">In-memory; resets when the backend restarts.</p>
    </section>
  )
}
