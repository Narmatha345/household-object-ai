import { useEffect, useId, useRef, useState, type ChangeEvent, type DragEvent, type KeyboardEvent } from 'react'
import { AlertIcon, ImagePlusIcon, InfoIcon, RefreshIcon, ScanIcon, XIcon } from './Icons'

const ACCEPTED_TYPES = ['image/jpeg', 'image/png', 'image/webp']
const ACCEPT_ATTR = '.jpg,.jpeg,.png,.webp,image/jpeg,image/png,image/webp'
const MAX_BYTES = 10 * 1024 * 1024

interface ImageUploadProps {
  disabled?: boolean
  onDetect: (file: File) => void
}

const formatSize = (bytes: number) =>
  bytes < 1024 * 1024 ? `${Math.max(1, Math.round(bytes / 1024))} KB` : `${(bytes / (1024 * 1024)).toFixed(1)} MB`

export default function ImageUpload({ disabled, onDetect }: ImageUploadProps) {
  const inputRef = useRef<HTMLInputElement>(null)
  const hintId = useId()
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

  const clearSelection = () => {
    setFile(null)
    setPreviewUrl(null)
    setError(null)
  }

  const openPicker = () => inputRef.current?.click()

  const onChange = (event: ChangeEvent<HTMLInputElement>) => {
    selectFile(event.target.files?.[0])
    event.target.value = ''
  }

  const onDrop = (event: DragEvent<HTMLDivElement>) => {
    event.preventDefault()
    setDragging(false)
    selectFile(event.dataTransfer.files?.[0])
  }

  const onKeyDown = (event: KeyboardEvent<HTMLDivElement>) => {
    if (event.key === 'Enter' || event.key === ' ') {
      event.preventDefault()
      openPicker()
    }
  }

  const detect = () => {
    if (!file) return
    onDetect(file)
    // Reset for the next photo; the result panel keeps its own preview.
    setFile(null)
    setPreviewUrl(null)
  }

  return (
    <div className="flex flex-col gap-5">
      <div
        role="button"
        tabIndex={0}
        aria-label={file ? `Selected image ${file.name}. Press Enter to choose a different photo.` : 'Choose a photo to upload'}
        aria-describedby={hintId}
        onClick={openPicker}
        onKeyDown={onKeyDown}
        onDragOver={(e) => {
          e.preventDefault()
          setDragging(true)
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={onDrop}
        className={`focus-ring group relative flex aspect-[4/3] cursor-pointer items-center justify-center overflow-hidden rounded-xl border-2 border-dashed transition-colors sm:aspect-video ${
          dragging
            ? 'border-blue-500 bg-blue-50'
            : previewUrl
              ? 'border-slate-200 bg-slate-50'
              : 'border-slate-200 bg-slate-50/60 hover:border-blue-300 hover:bg-blue-50/40'
        }`}
      >
        {previewUrl && file ? (
          <>
            <img src={previewUrl} alt="Selected upload preview" className="h-full w-full object-contain" />
            <div className="absolute right-3 top-3 flex gap-2">
              <button
                type="button"
                onClick={(e) => {
                  e.stopPropagation()
                  openPicker()
                }}
                className="btn-secondary px-3 py-1.5 text-xs shadow-sm"
              >
                <RefreshIcon className="h-3.5 w-3.5" />
                Change
              </button>
              <button
                type="button"
                onClick={(e) => {
                  e.stopPropagation()
                  clearSelection()
                }}
                aria-label="Remove selected image"
                className="btn-secondary px-3 py-1.5 text-xs shadow-sm hover:text-red-600"
              >
                <XIcon className="h-3.5 w-3.5" />
                Remove
              </button>
            </div>
            <span className="absolute bottom-3 left-3 max-w-[70%] truncate rounded-md bg-white/95 px-2.5 py-1 text-xs font-medium text-slate-700 shadow-sm ring-1 ring-slate-200">
              {file.name} · {formatSize(file.size)}
            </span>
          </>
        ) : (
          <div className="pointer-events-none flex flex-col items-center gap-4 px-6 text-center">
            <span className="relative flex h-20 w-20 items-center justify-center rounded-2xl border border-slate-200 bg-white text-slate-400 shadow-sm transition-colors group-hover:text-blue-500">
              <ImagePlusIcon className="h-9 w-9" />
            </span>
            <div>
              <p className="font-semibold text-slate-800">
                {dragging ? 'Drop the image to upload' : 'Click to choose a photo, or drag it here'}
              </p>
              <p className="mt-1 text-sm text-slate-500">Supports JPG, JPEG, PNG or WEBP • Up to 10 MB</p>
            </div>
          </div>
        )}
      </div>

      <input
        ref={inputRef}
        type="file"
        accept={ACCEPT_ATTR}
        onChange={onChange}
        tabIndex={-1}
        aria-label="Upload a photo"
        className="sr-only"
      />

      {error && (
        <div role="alert" className="flex items-start gap-3 rounded-lg border border-amber-200 bg-amber-50 px-4 py-3 text-sm text-amber-900">
          <AlertIcon className="mt-0.5 h-4 w-4 shrink-0 text-amber-600" />
          <span>{error}</span>
        </div>
      )}

      <div className="flex flex-col gap-4 sm:flex-row sm:items-center">
        <button type="button" disabled={!file || disabled} onClick={detect} className="btn-primary px-8 py-3 text-base sm:min-w-44">
          <ScanIcon className="h-5 w-5" />
          {disabled ? 'Analyzing…' : 'Detect'}
        </button>
        <div className="hidden h-10 w-px bg-slate-200 sm:block" aria-hidden="true" />
        <div id={hintId} className="flex items-start gap-2 text-xs text-slate-500">
          <InfoIcon className="mt-px h-4 w-4 shrink-0 text-slate-400" />
          <div>
            <p className="font-semibold text-slate-700">Supported formats</p>
            <p className="mt-0.5">JPG, JPEG, PNG, WEBP • Max size: 10 MB</p>
          </div>
        </div>
      </div>
    </div>
  )
}
