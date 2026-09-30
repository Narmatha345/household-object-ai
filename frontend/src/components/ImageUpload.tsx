import { useEffect, useRef, useState, type ChangeEvent, type DragEvent } from 'react'

const ACCEPTED_TYPES = ['image/jpeg', 'image/png', 'image/webp']
const ACCEPT_ATTR = '.jpg,.jpeg,.png,.webp,image/jpeg,image/png,image/webp'
const MAX_BYTES = 10 * 1024 * 1024

interface ImageUploadProps {
  disabled?: boolean
  onDetect: (file: File) => void
}

export default function ImageUpload({ disabled, onDetect }: ImageUploadProps) {
  const inputRef = useRef<HTMLInputElement>(null)
  const [file, setFile] = useState<File | null>(null)
  const [previewUrl, setPreviewUrl] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [dragging, setDragging] = useState(false)

  useEffect(() => {
    return () => {
      if (previewUrl) URL.revokeObjectURL(previewUrl)
    }
  }, [previewUrl])

  const selectFile = (candidate: File | undefined) => {
    setError(null)
    if (!candidate) return
    if (!ACCEPTED_TYPES.includes(candidate.type)) {
      setError('Unsupported file type. Please choose a JPG, PNG, or WEBP image.')
      return
    }
    if (candidate.size === 0) {
      setError('The selected file is empty.')
      return
    }
    if (candidate.size > MAX_BYTES) {
      setError('The image is too large. Maximum size is 10 MB.')
      return
    }
    setFile(candidate)
    setPreviewUrl(URL.createObjectURL(candidate))
  }

  const onChange = (event: ChangeEvent<HTMLInputElement>) => {
    selectFile(event.target.files?.[0])
    event.target.value = ''
  }

  const onDrop = (event: DragEvent<HTMLDivElement>) => {
    event.preventDefault()
    setDragging(false)
    selectFile(event.dataTransfer.files?.[0])
  }

  return (
    <div className="space-y-4">
      <div
        role="button"
        tabIndex={0}
        onClick={() => inputRef.current?.click()}
        onKeyDown={(e) => (e.key === 'Enter' || e.key === ' ') && inputRef.current?.click()}
        onDragOver={(e) => {
          e.preventDefault()
          setDragging(true)
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={onDrop}
        className={`relative flex aspect-video cursor-pointer items-center justify-center overflow-hidden rounded-xl border-2 border-dashed transition ${
          dragging ? 'border-teal-600 bg-teal-50' : 'border-slate-300 bg-white hover:border-slate-400'
        }`}
      >
        {previewUrl ? (
          <img src={previewUrl} alt="Selected upload preview" className="h-full w-full object-contain" />
        ) : (
          <div className="flex flex-col items-center gap-2 px-4 text-center text-slate-500">
            <svg viewBox="0 0 24 24" className="h-10 w-10" fill="none" stroke="currentColor" strokeWidth="1.5">
              <path d="M12 16V4m0 0-4 4m4-4 4 4M4 16v3a1 1 0 0 0 1 1h14a1 1 0 0 0 1-1v-3" />
            </svg>
            <span className="text-sm font-medium">Click to choose a photo, or drag it here</span>
            <span className="text-xs">JPG, JPEG, PNG or WEBP · up to 10 MB</span>
          </div>
        )}
      </div>

      <input ref={inputRef} type="file" accept={ACCEPT_ATTR} onChange={onChange} className="hidden" />

      {error && (
        <p role="alert" className="rounded-lg bg-amber-50 px-3 py-2 text-sm text-amber-800">
          {error}
        </p>
      )}

      <div className="flex flex-wrap items-center gap-3">
        <button
          type="button"
          disabled={!file || disabled}
          onClick={() => file && onDetect(file)}
          className="rounded-lg bg-teal-700 px-5 py-2.5 text-sm font-semibold text-white hover:bg-teal-800 disabled:cursor-not-allowed disabled:opacity-50"
        >
          Detect
        </button>
        {file && <span className="truncate text-sm text-slate-500">{file.name}</span>}
      </div>
    </div>
  )
}
