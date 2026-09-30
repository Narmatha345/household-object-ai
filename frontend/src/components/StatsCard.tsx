import type { ReactNode } from 'react'
import type { StatsResponse } from '../types/detection'
import { BarChartIcon, CheckIcon, CloudIcon, ListIcon, PercentIcon, RefreshIcon } from './Icons'

interface StatsCardProps {
  stats: StatsResponse | null
}

interface Metric {
  label: string
  value: string
  icon: ReactNode
  iconClass: string
}

export default function StatsCard({ stats }: StatsCardProps) {
  const metrics: Metric[] = [
    {
      label: 'Total requests',
      value: String(stats?.total_requests ?? 0),
      icon: <ListIcon className="h-5 w-5" />,
      iconClass: 'bg-blue-50 text-blue-600',
    },
    {
      label: 'Local detections',
      value: String(stats?.local_detections ?? 0),
      icon: <CheckIcon className="h-5 w-5" />,
      iconClass: 'bg-emerald-50 text-emerald-600',
    },
    {
      label: 'OpenAI fallbacks',
      value: String(stats?.openai_fallbacks ?? 0),
      icon: <CloudIcon className="h-5 w-5" />,
      iconClass: 'bg-violet-50 text-violet-600',
    },
    {
      label: 'Fallback rate',
      value: `${Math.round((stats?.fallback_rate ?? 0) * 100)}%`,
      icon: <PercentIcon className="h-5 w-5" />,
      iconClass: 'bg-slate-100 text-slate-600',
    },
  ]

  return (
    <section className="card p-5 sm:p-6" aria-labelledby="stats-title">
      <h2 id="stats-title" className="flex items-center gap-2.5 text-base font-semibold text-slate-900">
        <BarChartIcon className="h-5 w-5 text-slate-500" />
        Statistics
      </h2>

      <dl className="mt-5 grid grid-cols-2 gap-3">
        {metrics.map((metric) => (
          <div
            key={metric.label}
            className="flex flex-col gap-3 rounded-lg border border-slate-200/80 p-3.5 sm:flex-row sm:items-start lg:flex-col xl:flex-row"
          >
            <span className={`flex h-9 w-9 shrink-0 items-center justify-center rounded-lg ${metric.iconClass}`}>
              {metric.icon}
            </span>
            <div className="min-w-0">
              <dt className="text-xs font-medium text-slate-500">{metric.label}</dt>
              <dd className="mt-0.5 text-2xl font-bold tabular-nums tracking-tight text-slate-900">{metric.value}</dd>
            </div>
          </div>
        ))}
      </dl>

      <div className="mt-5 flex items-start gap-2.5 border-t border-slate-100 pt-4">
        <RefreshIcon className="mt-0.5 h-4 w-4 shrink-0 text-slate-400" />
        <div className="text-sm">
          <p className="font-medium text-slate-700">
            Failed requests: <span className="tabular-nums">{stats?.failed_requests ?? 0}</span>
          </p>
          <p className="mt-0.5 text-xs text-slate-500">In-memory statistics reset when the backend restarts.</p>
        </div>
      </div>
    </section>
  )
}
