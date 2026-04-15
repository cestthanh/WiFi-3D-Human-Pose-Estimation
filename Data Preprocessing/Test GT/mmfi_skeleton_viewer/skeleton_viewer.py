"""
MMFi Dataset: 3D Skeleton Ground Truth Viewer
===================================================
Features:
  - Runtime file switching: browse E/S/A hierarchy and reload without restart
  - No-cache headers on all API responses
  - Correct Human3.6M coordinate → Three.js mapping
  - Optional video playback
"""
import http.server
import json
import numpy as np
import os
import sys
import webbrowser
import mimetypes
from pathlib import Path
from urllib.parse import urlparse, parse_qs

# ============================================================
#  CONFIGURATION — change these defaults as needed
# ============================================================
DATASET_ROOT = r"D:\MMFi-dataset\MMFi_Dataset"
NPY_PATH     = r"D:\MMFi-dataset\MMFi_Dataset\E01\S01\A05\ground_truth.npy"
VIDEO_PATH   = ""
VIDEO_FPS    = 10
PORT         = 8082

# ============================================================
#  SKELETON CONFIG  (Human3.6M 17-joint layout for MMFi)
# ============================================================
KEYPOINT_NAMES = [
    "Pelvis", "R.Hip", "R.Knee", "R.Ankle",
    "L.Hip",  "L.Knee", "L.Ankle", "Spine",
    "Neck",   "Nose",   "Head",    "L.Shoulder",
    "L.Elbow","L.Wrist","R.Shoulder","R.Elbow","R.Wrist"
]

EDGES = [
    [0, 1], [1, 2], [2, 3],           # Right leg
    [0, 4], [4, 5], [5, 6],           # Left leg
    [0, 7], [7, 8], [8, 9], [9, 10],  # Spine to Head
    [8, 11],[11,12],[12,13],           # Left arm
    [8, 14],[14,15],[15,16]            # Right arm
]

JOINT_COLORS = [
    0xFF4081,  # 0  Pelvis
    0xE040FB,  # 1  R.Hip
    0xE040FB,  # 2  R.Knee
    0xE040FB,  # 3  R.Ankle
    0x40C4FF,  # 4  L.Hip
    0x40C4FF,  # 5  L.Knee
    0x40C4FF,  # 6  L.Ankle
    0xFF4081,  # 7  Spine
    0xFF4081,  # 8  Neck
    0xFF4081,  # 9  Nose
    0xFF4081,  # 10 Head
    0x69F0AE,  # 11 L.Shoulder
    0x69F0AE,  # 12 L.Elbow
    0x69F0AE,  # 13 L.Wrist
    0xFFD740,  # 14 R.Shoulder
    0xFFD740,  # 15 R.Elbow
    0xFFD740,  # 16 R.Wrist
]

EDGE_COLORS = [
    0xE040FB, 0xE040FB, 0xE040FB,           # Right leg
    0x40C4FF, 0x40C4FF, 0x40C4FF,           # Left leg
    0xFF4081, 0xFF4081, 0xFF4081, 0xFF4081, # Spine/Head
    0x69F0AE, 0x69F0AE, 0x69F0AE,           # Left arm
    0xFFD740, 0xFFD740, 0xFFD740,           # Right arm
]

LEGEND = [
    {"label": "Core/Head", "color": "#FF4081"},
    {"label": "L.Arm",     "color": "#69F0AE"},
    {"label": "R.Arm",     "color": "#FFD740"},
    {"label": "L.Leg",     "color": "#40C4FF"},
    {"label": "R.Leg",     "color": "#E040FB"},
]


# ============================================================
#  Data helpers
# ============================================================
def load_pose_data(path_str):
    path = Path(path_str)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path_str}")
    data = np.load(str(path))          # expected shape (T, 17, 3)
    if data.ndim == 2:                 # (17,3) single frame
        data = data[np.newaxis]
    if data.ndim == 3:                 # (T, 17, 3) → (T, 1, 17, 3)
        data = data[:, np.newaxis, :, :]
    # data is now (T, persons, joints, 3)
    action_id = "_".join(str(path).split(os.sep)[-4:-1])
    names = [f"[{action_id}] Frame {i}" for i in range(len(data))]
    return data.tolist(), names


def browse_dataset(root, env=None, subject=None):
    """Return available environments / subjects / actions from dataset root."""
    root = Path(root)
    if env is None:
        dirs = sorted(p.name for p in root.iterdir() if p.is_dir())
        return {"items": dirs, "level": "env"}
    if subject is None:
        dirs = sorted(p.name for p in (root / env).iterdir() if p.is_dir())
        return {"items": dirs, "level": "subject"}
    dirs = sorted(
        p.name for p in (root / env / subject).iterdir()
        if p.is_dir() and (p / "ground_truth.npy").exists()
    )
    return {"items": dirs, "level": "action"}


# ============================================================
#  HTTP handler
# ============================================================
class ViewerHandler(http.server.SimpleHTTPRequestHandler):
    # Mutable shared state (class-level so all requests share it)
    pose_data   = None
    frame_names = []
    video_path  = ""
    current_npy = NPY_PATH

    def log_message(self, format, *args):
        pass  # suppress console spam

    def _send_json(self, obj, status=200):
        content = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type",  "application/json")
        self.send_header("Content-Length", len(content))
        # Prevent ANY caching so switching files is instant
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
        self.send_header("Pragma",        "no-cache")
        self.end_headers()
        self.wfile.write(content)

    def do_GET(self):
        parsed = urlparse(self.path)
        qs     = parse_qs(parsed.query)

        # ── /api/config ──────────────────────────────────────
        if parsed.path == "/api/config":
            self._send_json({
                "keypoint_names": KEYPOINT_NAMES,
                "edges":          EDGES,
                "joint_colors":   [f"#{c:06X}" for c in JOINT_COLORS],
                "edge_colors":    [f"#{c:06X}" for c in EDGE_COLORS],
                "legend":         LEGEND,
                "video_fps":      VIDEO_FPS,
                "dataset_root":   DATASET_ROOT,
            })

        # ── /api/data ─────────────────────────────────────────
        elif parsed.path == "/api/data":
            self._send_json({
                "frames":      ViewerHandler.pose_data,
                "num_frames":  len(ViewerHandler.pose_data),
                "frame_names": ViewerHandler.frame_names,
                "has_video":   bool(ViewerHandler.video_path),
                "video_url":   "/video" if ViewerHandler.video_path else "",
                "current_npy": ViewerHandler.current_npy,
            })

        # ── /api/browse ───────────────────────────────────────
        elif parsed.path == "/api/browse":
            env     = qs.get("env",     [None])[0]
            subject = qs.get("subject", [None])[0]
            try:
                result = browse_dataset(DATASET_ROOT, env, subject)
                self._send_json(result)
            except Exception as e:
                self._send_json({"error": str(e)}, 500)

        # ── /api/reload ───────────────────────────────────────
        elif parsed.path == "/api/reload":
            npy = qs.get("path", [""])[0]
            if not npy:
                self._send_json({"error": "Missing ?path= parameter"}, 400)
                return
            try:
                data, names = load_pose_data(npy)
                ViewerHandler.pose_data   = data
                ViewerHandler.frame_names = names
                ViewerHandler.current_npy = npy
                print(f"  Loaded: {npy}  ({len(data)} frames)")
                self._send_json({"ok": True, "num_frames": len(data), "path": npy})
            except FileNotFoundError as e:
                self._send_json({"error": str(e)}, 404)
            except Exception as e:
                self._send_json({"error": str(e)}, 500)

        # ── /video ────────────────────────────────────────────
        elif parsed.path == "/video" and ViewerHandler.video_path:
            self._serve_video()

        # ── static files (index.html, etc.) ──────────────────
        else:
            super().do_GET()

    def _serve_video(self):
        filepath = ViewerHandler.video_path
        try:
            file_size = os.path.getsize(filepath)
            mime_type, _ = mimetypes.guess_type(filepath)
            mime_type = mime_type or "video/mp4"
            self.send_response(200)
            self.send_header("Content-Type",   mime_type)
            self.send_header("Content-Length",  file_size)
            self.send_header("Accept-Ranges",  "bytes")
            self.end_headers()
            with open(filepath, "rb") as f:
                self.wfile.write(f.read())
        except FileNotFoundError:
            self.send_error(404, "Video file not found")


# ============================================================
#  Entry point
# ============================================================
def main():
    print("=" * 60)
    print("  MMFi-Dataset: GT Skeleton Viewer")
    print("=" * 60)

    data, names = load_pose_data(NPY_PATH)
    ViewerHandler.pose_data   = data
    ViewerHandler.frame_names = names
    ViewerHandler.video_path  = VIDEO_PATH
    ViewerHandler.current_npy = NPY_PATH
    print(f"  Loaded: {NPY_PATH}")
    print(f"  Frames: {len(data)}")

    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    http.server.HTTPServer.allow_reuse_address = True
    server = http.server.HTTPServer(("localhost", PORT), ViewerHandler)

    url = f"http://localhost:{PORT}/index.html"
    print(f"\n  Open: {url}")
    print("  Press Ctrl+C to stop\n")
    webbrowser.open(url)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n  Server stopped.")
        server.server_close()


if __name__ == "__main__":
    main()
