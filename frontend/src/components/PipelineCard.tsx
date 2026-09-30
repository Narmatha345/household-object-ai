import { Fragment, type ReactNode } from 'react'
import type { DetectionSource } from '../types/detection'
import { ArrowDownIcon, ArrowRightIcon, ChipIcon, CloudIcon, PipelineIcon, ShieldCheckIcon } from './Icons'

interface PipelineCardProps {
  thresholdPercent: number | null
  lastSource: DetectionSource | null
}

interface Step {
  title: string
  description: string
  detail?: string
  icon: ReactNode
  tileClass: string
  numberClass: string
}

export default function PipelineCard({ thresholdPercent, lastSource }: PipelineCardProps) {
  const steps: Step[] = [
    {
      title: 'Local YOLO',
      description: 'Run detection using local model',
      icon: <ChipIcon className="h-7 w-7" />,
      tileClass: 'bg-blue-50 text-blue-600 ring-blue-100',
      numberClass: 'bg-blue-600',
    },
    {
      title: 'Confidence check',
      description: 'Verify whether the result is reliable',
      detail: thresholdPercent === null ? undefined : `≥ ${thresholdPercent}% confidence`,
      icon: <ShieldCheckIcon className="h-7 w-7" />,
      tileClass: 'bg-emerald-50 text-emerald-600 ring-emerald-100',
      numberClass: 'bg-emerald-600',
    },
    {
      title: 'OpenAI fallback',
      description: 'Used only when local result is uncertain',
      icon: <CloudIcon className="h-7 w-7" />,
      tileClass: 'bg-violet-50 text-violet-600 ring-violet-100',
      numberClass: 'bg-violet-600',
    },
  ]

  return (
    <section className="card p-5 sm:p-6" aria-labelledby="pipeline-title">
      <h2 id="pipeline-title" className="flex items-center gap-2.5 text-base font-semibold text-slate-900">
        <PipelineIcon className="h-5 w-5 text-slate-500" />
        Detection pipeline
      </h2>

      <ol className="mt-5 flex flex-col items-stretch gap-2 sm:flex-row sm:items-start sm:gap-1">
        {steps.map((step, index) => (
          <Fragment key={step.title}>
            <li className="flex flex-1 items-center gap-4 sm:flex-col sm:gap-3 sm:text-center">
              <span className={`relative flex h-14 w-14 shrink-0 items-center justify-center rounded-xl ring-1 ring-inset ${step.tileClass}`}>
                {step.icon}
                <span
                  className={`absolute -left-2 -top-2 flex h-5 w-5 items-center justify-center rounded-full text-[11px] font-bold text-white ring-2 ring-white ${step.numberClass}`}
                >
                  {index + 1}
                </span>
              </span>
              <div>
                <p className="text-sm font-semibold text-slate-900">{step.title}</p>
                <p className="mt-0.5 text-xs leading-relaxed text-slate-500">{step.description}</p>
                {step.detail && (
                  <p className="mt-1.5 inline-block whitespace-nowrap rounded bg-emerald-50 px-1.5 py-0.5 text-[11px] font-semibold text-emerald-700">
                    {step.detail}
                  </p>
                )}
              </div>
            </li>
            {index < steps.length - 1 && (
              <li aria-hidden="true" className="flex justify-start pl-4 text-slate-400 sm:justify-center sm:pl-0 sm:pt-4">
                <ArrowDownIcon className="h-5 w-5 sm:hidden" />
                <ArrowRightIcon className="hidden h-5 w-5 sm:block" />
              </li>
            )}
          </Fragment>
        ))}
      </ol>

      <p className="mt-5 rounded-lg bg-slate-50 px-3.5 py-2.5 text-xs leading-relaxed text-slate-600">
        {lastSource === 'local' && (
          <>
            <span className="font-semibold text-emerald-700">Last request:</span> resolved locally. OpenAI was not called.
          </>
        )}
        {lastSource === 'openai' && (
          <>
            <span className="font-semibold text-violet-700">Last request:</span> local confidence was too low, so the
            OpenAI fallback answered.
          </>
        )}
        {lastSource === null && (
          <>
            <span className="font-semibold text-slate-700">OpenAI is not the primary detector.</span> Every image is
            checked locally first to keep API usage low.
          </>
        )}
      </p>
    </section>
  )
}
