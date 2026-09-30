interface LoadingStateProps {
  message?: string
}

export default function LoadingState({ message = 'Detecting object...' }: LoadingStateProps) {
  return (
    <div role="status" aria-live="polite" className="flex items-center gap-3 py-6 text-slate-600">
      <span className="h-5 w-5 animate-spin rounded-full border-2 border-slate-300 border-t-teal-700" />
      <span className="text-sm font-medium">{message}</span>
    </div>
  )
}
