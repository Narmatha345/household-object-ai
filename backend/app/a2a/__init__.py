"""A2A (Agent-to-Agent) protocol adapter for the household object detector.

This package only translates between A2A messages/tasks and the existing
detection pipeline (app.services.detection_router). It contains no ML code.
"""

import os

# Ultralytics patches PIL.Image.open and pip-installs extra packages (e.g. pi-heif) when it
# meets an unreadable image. A network-facing agent must never let a caller trigger installs.
os.environ.setdefault("YOLO_AUTOINSTALL", "false")
