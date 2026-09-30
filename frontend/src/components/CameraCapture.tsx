import { useCallback, useEffect, useRef, useState } from 'react'
import { AlertIcon, CameraIcon, InfoIcon, ScanIcon, SwitchCameraIcon, XIcon } from './Icons'

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
  const [cameras, setCameras] = useState<MediaDeviceInfo[]>([])
  const [activeDeviceId, setActiveDeviceId] = useState<string | null>(null)

  const releaseStream = () => {
    streamRef.current?.getTracks().forEach((track) => track.stop())
    streamRef.current = null
  }

  const stopCamera = useCallback(() => {
    releaseStream()
    setStatus('idle')
  }, [])

  // With no deviceId, prefer the back camera (phones); otherwise open that exact camera.
  const startCamera = useCallback(
    async (deviceId?: string) => {
      setError(null)
      if (!navigator.mediaDevices?.getUserMedia) {
        setError(cameraErrorMessage(null))
        return
      }
      // Phones often can't open two cameras at once, so release the current one first.
      releaseStream()
      setStatus('starting')
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: {
            ...(deviceId ? { deviceId: { exact: deviceId } } : { facingMode: { ideal: 'environment' } }),
            width: { ideal: 1280 },
            height: { ideal: 720 },
          },
          audio: false,
        })
        streamRef.current = stream
        setActiveDeviceId(stream.getVideoTracks()[0]?.getSettings().deviceId ?? null)
        if (videoRef.current) {
          videoRef.current.srcObject = stream
          await videoRef.current.play()
        }
        setStatus('live')

        // Device list is only complete once permission has been granted.
        const devices = await navigator.mediaDevices.enumerateDevices()
        setCameras(devices.filter((device) => device.kind === 'videoinput'))
      } catch (err) {
        stopCamera()
        setError(cameraErrorMessage(err))
      }
    },
    [stopCamera],
  )

  const switchCamera = () => {
    if (cameras.length < 2) return
    const currentIndex = cameras.findIndex((camera) => camera.deviceId === activeDeviceId)
    const next = cameras[(currentIndex + 1) % cameras.length]
    void startCamera(next.deviceId)
  }

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

  const isLive = status === 'live'
  const canSwitch = cameras.length > 1

  return (
    <div className="flex flex-col gap-5">
      <div
        className={`relative aspect-[4/3] overflow-hidden rounded-xl sm:aspect-video ${
          isLive ? 'bg-slate-900' : 'border-2 border-dashed border-slate-200 bg-slate-50/60'
        }`}
      >
        <video
          ref={videoRef}
          playsInline
          muted
          aria-label="Live camera preview"
          className={`h-full w-full object-cover ${isLive ? '' : 'hidden'}`}
        />

        {isLive && (
          <span className="badge absolute left-3 top-3 bg-slate-900/70 text-white backdrop-blur">
            <span className="h-1.5 w-1.5 rounded-full bg-red-500" aria-hidden="true" />
            Live
          </span>
        )}

        {!isLive && (
          <div className="absolute inset-0 flex flex-col items-center justify-center gap-4 px-6 text-center">
            <span className="flex h-16 w-16 items-center justify-center rounded-2xl border border-slate-200 bg-white text-slate-400 shadow-sm">
              <CameraIcon className="h-8 w-8" />
            </span>
            <div>
              <p className="font-semibold text-slate-800">
                {status === 'starting' ? 'Starting camera…' : 'Camera is off'}
              </p>
              <p className="mt-1 text-sm text-slate-500">
                {status === 'starting'
                  ? 'Allow camera access if your browser asks.'
                  : 'Start the camera, frame a household object, then capture.'}
              </p>
            </div>
            {status === 'idle' && (
              <button
                type="button"
                onClick={() => void startCamera(activeDeviceId ?? undefined)}
                className="btn-primary"
              >
                <CameraIcon className="h-4 w-4" />
                Start camera
              </button>
            )}
          </div>
        )}
      </div>

      {error && (
        <div role="alert" className="flex items-start gap-3 rounded-lg border border-amber-200 bg-amber-50 px-4 py-3 text-sm text-amber-900">
          <AlertIcon className="mt-0.5 h-4 w-4 shrink-0 text-amber-600" />
          <span>{error}</span>
        </div>
      )}

      <div className="flex flex-col gap-4 sm:flex-row sm:items-center">
        <div className="flex gap-3">
          <button
            type="button"
            onClick={capture}
            disabled={!isLive || disabled}
            className="btn-primary flex-1 whitespace-nowrap px-5 py-3 text-base sm:flex-none sm:px-7"
          >
            <ScanIcon className="h-5 w-5" />
            {disabled ? 'Analyzing…' : 'Capture & detect'}
          </button>
          {isLive && (
            <>
              <button
                type="button"
                onClick={switchCamera}
                disabled={!canSwitch}
                aria-label="Switch camera"
                title={canSwitch ? 'Switch to the next camera' : 'Only one camera was found on this device'}
                className="btn-secondary"
              >
                <SwitchCameraIcon className="h-4 w-4" />
                <span className="hidden sm:inline">Switch</span>
              </button>
              <button type="button" onClick={stopCamera} aria-label="Stop camera" className="btn-secondary">
                <XIcon className="h-4 w-4" />
                <span className="hidden sm:inline">Stop</span>
              </button>
            </>
          )}
        </div>
        <p className="flex items-start gap-2 text-xs leading-relaxed text-slate-500 sm:ml-auto sm:max-w-56">
          <InfoIcon className="mt-px h-4 w-4 shrink-0 text-slate-400" />
          {isLive && !canSwitch
            ? 'Only one camera found. Switching works on phones or with a USB webcam.'
            : 'The captured frame is checked by local ML first.'}
        </p>
      </div>
    </div>
  )
}
