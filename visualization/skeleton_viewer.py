"""
MMFi Dataset: 3D Skeleton + CSI Viewer
=======================================
Features:
  - Synchronized RGB image, 3D skeleton, and CSI visualization
  - CSI preloaded into RAM for instant frame access
  - No-cache headers on all API responses

Chạy:
    cd D:\\Thesis_Docs\\WiFi-3D-Human-Pose-Estimation\\visualization
    python skeleton_viewer.py --dataset_root "D:\\MMFi-dataset\\MMFi_Dataset"
                              --npy "D:\\MMFi-dataset\\MMFi_Dataset\\E01\\S01\\A01\\ground_truth.npy"
                              --rgb_dir "D:\\RGB_image-MMFi\\MMFi_Defaced_RGB"
                              --port 8082
"""
import argparse
import http.server
import json
import numpy as np
import os
import sys
import webbrowser
import mimetypes
import scipy.io as sio
from pathlib import Path
from urllib.parse import urlparse, parse_qs

# ============================================================
#  ARGUMENT PARSING
# ============================================================
def parse_args():
    parser = argparse.ArgumentParser(description="MMFi Skeleton + CSI Viewer")
    parser.add_argument("--dataset_root", type=str,
                        default=r"D:\MMFi-dataset\MMFi_Dataset",
                        help="Đường dẫn thư mục gốc MMFi Dataset")
    parser.add_argument("--npy", type=str,
                        default=r"G:\My Drive\MMFi_lite\E01\S05\A24\ground_truth.npy",
                        help="Ground truth .npy để load ban đầu")
    parser.add_argument("--rgb_dir", type=str,
                        default=r"D:\RGB_image-MMFi\MMFi_Defaced_RGB",
                        help="Thư mục chứa ảnh RGB (có thể để trống nếu không có)")
    parser.add_argument("--port", type=int, default=8082,
                        help="Port cho HTTP server (default: 8082)")
    return parser.parse_args()

ARGS = parse_args()

DATASET_ROOT = ARGS.dataset_root
NPY_PATH     = ARGS.npy
RGB_DIR      = ARGS.rgb_dir
PORT         = ARGS.port

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
    [0, 1], [1, 2], [2, 3],
    [0, 4], [4, 5], [5, 6],
    [0, 7], [7, 8], [8, 9], [9, 10],
    [8, 11],[11,12],[12,13],
    [8, 14],[14,15],[15,16]
]

JOINT_COLORS = [
    0xFF4081, 0xE040FB, 0xE040FB, 0xE040FB,
    0x40C4FF, 0x40C4FF, 0x40C4FF,
    0xFF4081, 0xFF4081, 0xFF4081, 0xFF4081,
    0x69F0AE, 0x69F0AE, 0x69F0AE,
    0xFFD740, 0xFFD740, 0xFFD740,
]

EDGE_COLORS = [
    0xE040FB, 0xE040FB, 0xE040FB,
    0x40C4FF, 0x40C4FF, 0x40C4FF,
    0xFF4081, 0xFF4081, 0xFF4081, 0xFF4081,
    0x69F0AE, 0x69F0AE, 0x69F0AE,
    0xFFD740, 0xFFD740, 0xFFD740,
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
    data = np.load(str(path))
    if data.ndim == 2:
        data = data[np.newaxis]
    if data.ndim == 3:
        data = data[:, np.newaxis, :, :]
    parts = str(path).split(os.sep)
    action_id = "_".join(parts[-4:-1])
    names = [f"[{action_id}] frame{i+1:03d}" for i in range(len(data))]
    return data.tolist(), names


def load_csi_data(npy_path):
    """Preload ALL CSI .mat files for the action matching the given npy_path.
    Returns a dict with:
      - heatmap: (num_frames, 3, 114)  amplitude averaged over 10 packets
      - amp_min / amp_max: global range for color mapping
    """
    npy = Path(npy_path)
    parts = npy.parts
    env, sub, act = parts[-4], parts[-3], parts[-2]
    csi_dir = Path(DATASET_ROOT) / env / sub / act / "wifi-csi"

    if not csi_dir.exists():
        print(f"  CSI dir not found: {csi_dir}")
        return None

    mat_files = sorted(csi_dir.glob("frame*.mat"))
    if not mat_files:
        print(f"  No .mat files in {csi_dir}")
        return None

    print(f"  Loading {len(mat_files)} CSI frames from {csi_dir} ...")
    all_amp = []
    for mf in mat_files:
        m = sio.loadmat(str(mf))
        amp = m["CSIamp"]         # (3, 114, 10)
        amp_avg = amp.mean(axis=2) # (3, 114) — average over packets
        all_amp.append(amp_avg)

    heatmap = np.stack(all_amp, axis=0)  # (T, 3, 114)
    # Replace -inf/inf/NaN with finite values
    heatmap = np.nan_to_num(heatmap, nan=0.0, posinf=0.0, neginf=0.0)
    amp_min = float(heatmap.min())
    amp_max = float(heatmap.max())

    result = {
        "heatmap": np.round(heatmap, 2).tolist(),
        "num_frames": len(mat_files),
        "num_antennas": 3,
        "num_subcarriers": 114,
        "amp_min": round(amp_min, 2),
        "amp_max": round(amp_max, 2),
    }
    print(f"  CSI loaded: {len(mat_files)} frames, amp range [{amp_min:.1f}, {amp_max:.1f}]")
    return result


# ============================================================
#  HTTP handler
# ============================================================
class ViewerHandler(http.server.SimpleHTTPRequestHandler):
    pose_data   = None
    frame_names = []
    current_npy = NPY_PATH
    csi_data    = None   # preloaded CSI

    def log_message(self, format, *args):
        pass

    def _send_json(self, obj, status=200):
        content = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type",  "application/json")
        self.send_header("Content-Length", len(content))
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
        self.send_header("Pragma",        "no-cache")
        self.end_headers()
        self.wfile.write(content)

    def do_GET(self):
        parsed = urlparse(self.path)
        qs     = parse_qs(parsed.query)

        if parsed.path == "/api/config":
            self._send_json({
                "keypoint_names": KEYPOINT_NAMES,
                "edges":          EDGES,
                "joint_colors":   [f"#{c:06X}" for c in JOINT_COLORS],
                "edge_colors":    [f"#{c:06X}" for c in EDGE_COLORS],
                "legend":         LEGEND,
            })

        elif parsed.path == "/api/data":
            self._send_json({
                "frames":      ViewerHandler.pose_data,
                "num_frames":  len(ViewerHandler.pose_data),
                "frame_names": ViewerHandler.frame_names,
                "current_npy": ViewerHandler.current_npy,
            })

        elif parsed.path == "/api/csi":
            if ViewerHandler.csi_data:
                self._send_json(ViewerHandler.csi_data)
            else:
                self._send_json({"error": "No CSI data available"}, 404)

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
                ViewerHandler.csi_data    = load_csi_data(npy)
                print(f"  Loaded: {npy}  ({len(data)} frames)")
                self._send_json({"ok": True, "num_frames": len(data), "path": npy})
            except FileNotFoundError as e:
                self._send_json({"error": str(e)}, 404)
            except Exception as e:
                self._send_json({"error": str(e)}, 500)

        elif parsed.path == "/api/rgb":
            frame_idx = int(qs.get("frame", ["0"])[0])
            npy_parts = Path(ViewerHandler.current_npy).parts
            env, sub, act = npy_parts[-4], npy_parts[-3], npy_parts[-2]
            img_filename = f"frame{frame_idx + 1:03d}.png"
            img_path = Path(RGB_DIR) / env / sub / act / "rgb" / img_filename
            try:
                file_size = os.path.getsize(str(img_path))
                self.send_response(200)
                self.send_header("Content-Type", "image/png")
                self.send_header("Content-Length", str(file_size))
                self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
                self.end_headers()
                with open(img_path, "rb") as f:
                    self.wfile.write(f.read())
            except FileNotFoundError:
                self.send_error(404, "RGB frame not found")

        else:
            super().do_GET()


# ============================================================
#  Entry point
# ============================================================
def main():
    print("=" * 60)
    print("  MMFi-Dataset: GT Skeleton + CSI Viewer")
    print("=" * 60)
    print(f"  Dataset root: {DATASET_ROOT}")
    print(f"  Default NPY:  {NPY_PATH}")

    data, names = load_pose_data(NPY_PATH)
    ViewerHandler.pose_data   = data
    ViewerHandler.frame_names = names
    ViewerHandler.current_npy = NPY_PATH
    print(f"  Skeleton: {NPY_PATH}")
    print(f"  Frames:   {len(data)}")

    ViewerHandler.csi_data = load_csi_data(NPY_PATH)

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
