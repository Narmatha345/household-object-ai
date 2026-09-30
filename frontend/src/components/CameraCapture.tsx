import { useCallback, useEffect, useRef, useState } from 'react'

interface CameraCaptureProps {
  disabled?: boolean
  onCapture: (image: Blob) => void
}

function cameraErrorMessage(error: unknown): string {
  if (!window.isSecureContext) {
    return 'Camera access requires HTTPS or localhost.'
  }
  if (error instanceof DOMException) {
    switch (error.name) {
      case 'NotAllowedError':
      case 'SecurityError':
        return 'Camera permission was denied. Allow camera access in your browser settings, or upload a photo instead.'
      case 'NotFoundError':
      case 'OverconstrainedError':
        return 'No camera was found on this device. Try uploading a photo instead.'
      case 'NotReadableError':
        return 'The camera is already in use by another application.'
    }
  }
  return 'Could not start the camera. Try uploading a photo instead.'
}

export default function CameraCapture({ disabled, onCapture }: CameraCaptureProps) {
  const videoRef = useRef<HTMLVideoElement>(null)
  const streamRef = useRef<MediaStream | null>(null)
  const [status, setStatus] = useState<'idle' | 'starting' | 'live'>('idle')
  const [error, setError] = useState<string | null>(null)

  const stopCamera = useCallback(() => {
    streamRef.current?.getTracks().forEach((track) => track.stop())
    streamRef.current = null
    setStatus('idle')
  }, [])

  const startCamera = useCallback(async () => {
    setError(null)
    if (!navigator.mediaDevices?.getUserMedia) {
      setError(cameraErrorMessage(null))
      return
    }
    setStatus('starting')
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: { ideal: 'environment' }, width: { ideal: 1280 }, height: { ideal: 720 } },
        audio: false,
      })
      streamRef.current = stream
      if (videoRef.current) {
        videoRef.current.srcObject = stream
        await videoRef.current.play()
      }
      setStatus('live')
    } catch (err) {
      stopCamera()
      setError(cameraErrorMessage(err))
    }
  }, [stopCamera])

  // Release the camera when the component unmounts (e.g. switching tabs).
  useEffect(() => stopCamera, [stopCamera])

  const capture = () => {
    const video = videoRef.current
    if (!video || !video.videoWidth) return

    const canvas = document.createElement('canvas')
    canvas.width = video.videoWidth
    canvas.height = video.videoHeight
    canvas.getContext('2d')?.drawImage(video, 0, 0, canvas.width, canvas.height)
    canvas.toBlob(
      (blob) => {
        if (!blob) {
          setError('Could not capture an image from the camera. Please try again.')
          return
        }
        onCapture(blob)
      },
      'image/jpeg',
      0.92,
    )
  }

  return (
    <div className="space-y-4">
      <div className="relative aspect-video overflow-hidden rounded-xl bg-slate-900">
        <video
          ref={videoRef}
          playsInline
          muted
          className={`h-full w-full object-cover ${status === 'live' ? '' : 'hidden'}`}
        />
        {status !== 'live' && (
          <div className="absolute inset-0 flex flex-col items-center justify-center gap-2 text-slate-400">
            <svg viewBox="0 0 24 24" className="h-10 w-10" fill="none" stroke="currentColor" strokeWidth="1.5">
              <path d="M3 8a2 2 0 0 1 2-2h2l1.5-2h7L17 6h2a2 2 0 0 1 2 2v10a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z" />
              <circle cx="12" cy="13" r="3.5" />
            </svg>
            <span className="text-sm">{status === 'starting' ? 'Starting camera…' : 'Camera is off'}</span>
          </div>
        )}
      </div>

      {error && (
        <p role="alert" className="rounded-lg bg-amber-50 px-3 py-2 text-sm text-amber-800">
          {error}
        </p>
      )}

      <div className="flex flex-wrap gap-3">
        {status === 'live' ? (
          <>
            <button
              type="button"
              onClick={capture}
              disabled={disabled}
              className="rounded-lg bg-teal-700 px-5 py-2.5 text-sm font-semibold text-white hover:bg-teal-800 disabled:cursor-not-allowed disabled:opacity-50"
            >
              Capture &amp; detect
            </button>
            <button
              type="button"
              onClick={stopCamera}
              className="rounded-lg border border-slate-300 px-4 py-2.5 text-sm font-medium text-slate-700 hover:bg-slate-100"
            >
              Stop camera
            </button>
          </>
        ) : (
          <button
            type="button"
            onClick={startCamera}
            disabled={status === 'starting'}
            className="rounded-lg bg-teal-700 px-5 py-2.5 text-sm font-semibold text-white hover:bg-teal-800 disabled:opacity-50"
          >
            Start camera
          </button>
        )}
      </div>
    </div>
  )
}
