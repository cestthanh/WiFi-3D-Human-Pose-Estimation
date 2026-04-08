"""
Person-in-WiFi-3D: 3D Skeleton Ground Truth Viewer
===================================================
Edit NPY_PATH and VIDEO_PATH below, then run:
    python skeleton_viewer.py
"""
import http.server
import json
import numpy as np
import os
import sys
import glob
import webbrowser
import mimetypes
from pathlib import Path
from urllib.parse import urlparse

# ============================================================
#  CONFIGURATION — Edit these paths before running
# ============================================================

# Option 1: Single .npy file
# NPY_PATH = r"D:\Thesis_Docs\Person-in-WiFi-3D-data\test_data\keypoint\S11_01_308.npy"

# Option 2: Load a sequence (glob pattern)
NPY_PATH = r"D:\Thesis\WiFi-3D-Human-Pose-Estimation\Data\Person-in-WiFi-3D\train_data\keypoint\S11_19_*.npy"

# Option 3: Load entire directory
# NPY_PATH = r"D:\Thesis_Docs\Person-in-WiFi-3D-data\test_data\keypoint"

# Video path (leave empty if no video)
VIDEO_PATH = r"D:\Thesis\WiFi-3D-Human-Pose-Estimation\Data Preprocessing\Test GT\skeleton_viewer\Video demo\S11_19\output_h264.mp4"

# Time list path — maps frame_id to real timestamp for accurate video sync
# Format per line: "frame_id_YYYY-MM-DD HH:MM:SS.ffffff"
# Leave empty to fall back to VIDEO_FPS-based sync
TIME_LIST_PATH = r"D:\Thesis\WiFi-3D-Human-Pose-Estimation\Data Preprocessing\Test GT\skeleton_viewer\Video demo\S11_19\time_list.txt"

# Video source FPS — used ONLY if TIME_LIST_PATH is empty
# Confirmed 15fps from ffmpeg (Azure Kinect color stream)
VIDEO_FPS = 15

PORT = 8891

# ============================================================
#  SKELETON CONFIG — Technical constants for the 14-keypoint format
#  Source: wifi_pose.py keypoint_process + distance analysis
#  Format: Person-in-WiFi-3D ground truth .npy files
# ============================================================

# Ground truth keypoint ordering (index → body part name)
KEYPOINT_NAMES = [
    "Head",       # 0
    "Neck",       # 1
    "L.Shoulder", # 2
    "R.Shoulder", # 3
    "L.Elbow",    # 4
    "L.Hip",      # 5
    "R.Elbow",    # 6
    "R.Hip",      # 7
    "L.Wrist",    # 8
    "L.Knee",     # 9
    "R.Wrist",    # 10
    "R.Knee",     # 11
    "L.Ankle",    # 12
    "R.Ankle",    # 13
]

# Skeleton edges: [joint_a, joint_b]
# Derived from wifi_pose.py keypoint_process() and validated by bone-length analysis
EDGES = [
    [0, 1],   # Head      — Neck
    [1, 2],   # Neck      — L.Shoulder
    [1, 3],   # Neck      — R.Shoulder
    [2, 4],   # L.Shoulder — L.Elbow
    [4, 8],   # L.Elbow   — L.Wrist
    [3, 6],   # R.Shoulder — R.Elbow
    [6, 10],  # R.Elbow   — R.Wrist
    [2, 5],   # L.Shoulder — L.Hip  (torso)
    [3, 7],   # R.Shoulder — R.Hip  (torso)
    [5, 7],   # L.Hip     — R.Hip
    [5, 9],   # L.Hip     — L.Knee
    [9, 12],  # L.Knee    — L.Ankle
    [7, 11],  # R.Hip     — R.Knee
    [11, 13], # R.Knee    — R.Ankle
]

# Joint colors (hex, per keypoint index)
# Pink=Head/Neck, Green=Left arm, Yellow=Right arm, Cyan=Left leg, Purple=Right leg, Blue=Torso
JOINT_COLORS = [
    0xFF4081,  # 0  Head
    0xFF4081,  # 1  Neck
    0x69F0AE,  # 2  L.Shoulder
    0xFFD740,  # 3  R.Shoulder
    0x69F0AE,  # 4  L.Elbow
    0x40C4FF,  # 5  L.Hip
    0xFFD740,  # 6  R.Elbow
    0xE040FB,  # 7  R.Hip
    0x69F0AE,  # 8  L.Wrist
    0x40C4FF,  # 9  L.Knee
    0xFFD740,  # 10 R.Wrist
    0xE040FB,  # 11 R.Knee
    0x40C4FF,  # 12 L.Ankle
    0xE040FB,  # 13 R.Ankle
]

# Edge colors (hex, per edge index, same order as EDGES)
EDGE_COLORS = [
    0xFF4081,  # Head — Neck
    0xFF6E40,  # Neck — L.Shoulder
    0xFF6E40,  # Neck — R.Shoulder
    0x69F0AE,  # L.Shoulder — L.Elbow
    0x69F0AE,  # L.Elbow — L.Wrist
    0xFFD740,  # R.Shoulder — R.Elbow
    0xFFD740,  # R.Elbow — R.Wrist
    0x448AFF,  # L.Shoulder — L.Hip
    0x448AFF,  # R.Shoulder — R.Hip
    0x448AFF,  # L.Hip — R.Hip
    0x40C4FF,  # L.Hip — L.Knee
    0x40C4FF,  # L.Knee — L.Ankle
    0xE040FB,  # R.Hip — R.Knee
    0xE040FB,  # R.Knee — R.Ankle
]

# Legend groups shown in the viewer
LEGEND = [
    {"label": "Head/Neck", "color": "#FF4081"},
    {"label": "L.Arm",     "color": "#69F0AE"},
    {"label": "R.Arm",     "color": "#FFD740"},
    {"label": "Torso",     "color": "#448AFF"},
    {"label": "L.Leg",     "color": "#40C4FF"},
    {"label": "R.Leg",     "color": "#E040FB"},
]

# Coordinate system: camera mounted overhead, looking down
# Data axes:   X = horizontal,  Y = depth,  Z = height (increases downward)
# Three.js:    X = right,       Y = up,     Z = toward camera
# Mapping:     three_x = data_x,  three_y = -data_z,  three_z = -data_y
AXIS_MAPPING = {"x": "data_x", "y": "-data_z", "z": "-data_y"}
# ============================================================


def _natural_sort_key(filepath):
    """Sort by the last numeric part in filename: S11_01_10 → 10."""
    import re
    stem = Path(filepath).stem
    nums = re.findall(r'\d+', stem)
    return int(nums[-1]) if nums else 0


def load_time_list(path_str):
    """Parse time_list.txt → dict {frame_id (int): video_time_seconds (float)}.
    Each line format: '506_2023-04-04 13:03:31.816916'
    video_time = timestamp - timestamp[0] (relative to start of video).
    Returns empty dict if path is empty or file not found."""
    if not path_str:
        return {}
    try:
        from datetime import datetime
        timestamps = {}  # frame_id -> datetime
        with open(path_str, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                underscore_pos = line.index('_')
                frame_id = int(line[:underscore_pos])
                dt_str = line[underscore_pos + 1:]
                dt = datetime.strptime(dt_str, '%Y-%m-%d %H:%M:%S.%f')
                timestamps[frame_id] = dt

        if not timestamps:
            return {}

        t0 = min(timestamps.values())
        result = {fid: (ts - t0).total_seconds() for fid, ts in timestamps.items()}
        print(f"  Timelist: {len(result)} entries, duration={max(result.values()):.1f}s")
        return result
    except Exception as e:
        print(f"  WARNING: Could not load time list: {e}")
        return {}


def load_pose_data(path_str):
    """Load pose data from .npy file(s). Returns (frames_list, names_list).
    Each frame is shape (num_persons, 14, 3)."""
    path = Path(path_str)
    frames = []
    names = []

    if path.is_file() and path.suffix == '.npy':
        data = np.load(str(path))
        data = np.squeeze(data)
        if data.ndim == 2:
            data = data[np.newaxis, ...]  # (14,3) -> (1,14,3)
        frames.append(data.tolist())
        names.append(path.stem)

    elif path.is_dir():
        files = sorted(path.glob('*.npy'), key=_natural_sort_key)
        if not files:
            print(f"  ERROR: No .npy files found in {path}")
            sys.exit(1)
        for f in files:
            d = np.squeeze(np.load(str(f)))
            if d.ndim == 2:
                d = d[np.newaxis, ...]
            frames.append(d.tolist())
            names.append(f.stem)

    elif '*' in path_str or '?' in path_str:
        files = sorted(glob.glob(path_str), key=_natural_sort_key)
        if not files:
            print(f"  ERROR: No files matching {path_str}")
            sys.exit(1)
        for f in files:
            d = np.squeeze(np.load(f))
            if d.ndim == 2:
                d = d[np.newaxis, ...]
            frames.append(d.tolist())
            names.append(Path(f).stem)
    else:
        print(f"  ERROR: {path_str} not found")
        sys.exit(1)

    return frames, names


class ViewerHandler(http.server.SimpleHTTPRequestHandler):
    pose_data = None
    frame_names = []
    frame_timestamps = []   # list of video_time_seconds per skeleton frame, or []
    video_path = ""

    def log_message(self, format, *args):
        pass

    def do_GET(self):
        parsed = urlparse(self.path)

        if parsed.path == '/api/config':
            # Serve all skeleton technical constants to the browser
            response = {
                'keypoint_names': KEYPOINT_NAMES,
                'edges': EDGES,
                'joint_colors': [f'#{c:06X}' for c in JOINT_COLORS],
                'edge_colors': [f'#{c:06X}' for c in EDGE_COLORS],
                'legend': LEGEND,
                'video_fps': VIDEO_FPS,
                'axis_mapping': AXIS_MAPPING,
            }
            content = json.dumps(response).encode()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', len(content))
            self.end_headers()
            self.wfile.write(content)

        elif parsed.path == '/api/data':
            response = {
                'frames': self.pose_data,
                'num_frames': len(self.pose_data),
                'frame_names': self.frame_names,
                'frame_timestamps': self.frame_timestamps,  # [] if no time list
                'has_video': bool(self.video_path),
                'video_url': '/video' if self.video_path else ''
            }
            content = json.dumps(response).encode()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', len(content))
            self.end_headers()
            self.wfile.write(content)

        elif parsed.path == '/video' and self.video_path:
            self._serve_video()

        else:
            super().do_GET()

    def _serve_video(self):
        filepath = self.video_path
        try:
            file_size = os.path.getsize(filepath)
            mime_type, _ = mimetypes.guess_type(filepath)
            mime_type = mime_type or 'video/mp4'

            range_header = self.headers.get('Range')
            if range_header:
                byte_range = range_header.replace('bytes=', '')
                start, end = byte_range.split('-')
                start = int(start)
                end = int(end) if end else file_size - 1
                length = end - start + 1

                self.send_response(206)
                self.send_header('Content-Type', mime_type)
                self.send_header('Content-Range', f'bytes {start}-{end}/{file_size}')
                self.send_header('Content-Length', length)
                self.send_header('Accept-Ranges', 'bytes')
                self.end_headers()

                with open(filepath, 'rb') as f:
                    f.seek(start)
                    self.wfile.write(f.read(length))
            else:
                self.send_response(200)
                self.send_header('Content-Type', mime_type)
                self.send_header('Content-Length', file_size)
                self.send_header('Accept-Ranges', 'bytes')
                self.end_headers()
                with open(filepath, 'rb') as f:
                    self.wfile.write(f.read())

        except FileNotFoundError:
            self.send_error(404, 'Video file not found')


def main():
    print("=" * 60)
    print("  Person-in-WiFi-3D: GT Skeleton Viewer")
    print("=" * 60)

    data, names = load_pose_data(NPY_PATH)
    num_persons_first = len(data[0]) if data else 0
    print(f"  Loaded : {NPY_PATH}")
    print(f"  Frames : {len(data)}")
    print(f"  Persons: {num_persons_first} (in first frame)")

    # Build per-frame video timestamps using time_list
    time_map = load_time_list(TIME_LIST_PATH)
    import re
    frame_timestamps = []
    for name in names:
        nums = re.findall(r'\d+', name)
        fid = int(nums[-1]) if nums else -1
        frame_timestamps.append(time_map.get(fid, None))
    has_timestamps = any(t is not None for t in frame_timestamps)

    if VIDEO_PATH:
        print(f"  Video  : {VIDEO_PATH}")
    else:
        print(f"  Video  : (none)")
    if has_timestamps:
        print(f"  Sync   : timestamp-based (time_list.txt)")
    else:
        print(f"  Sync   : FPS-based ({VIDEO_FPS} fps)")

    ViewerHandler.pose_data = data
    ViewerHandler.frame_names = names
    ViewerHandler.frame_timestamps = frame_timestamps if has_timestamps else []
    ViewerHandler.video_path = VIDEO_PATH

    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    server = http.server.HTTPServer(('localhost', PORT), ViewerHandler)
    url = f"http://localhost:{PORT}/index.html"
    print(f"\n  Open: {url}")
    print(f"  Press Ctrl+C to stop\n")

    webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n  Server stopped.")
        server.server_close()


if __name__ == '__main__':
    main()
