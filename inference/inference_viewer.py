"""
MMFi Inference Viewer — GT vs Model Predicted Skeleton
=======================================================
Chỉnh đường dẫn bên dưới rồi chạy:
    cd D:\Thesis_Docs\WiFi-3D-Human-Pose-Estimation\inference
    python inference_viewer.py
"""
import threading
import http.server
import json
import os
import sys
import webbrowser
import mimetypes
import numpy as np
import scipy.io as sio
from pathlib import Path
from urllib.parse import urlparse, parse_qs

# ============================================================
#  ▼▼▼  CHỈNH CÁC ĐƯỜNG DẪN TẠI ĐÂY  ▼▼▼
# ============================================================
DATASET_ROOT = r"G:\My Drive\MMFi_lite"          # Thư mục gốc dataset
NPY_PATH     = r"G:\My Drive\MMFi_lite\E01\S01\A02\ground_truth.npy"  # File GT ban đầu
MODEL_PATH   = r"D:\Thesis_Docs\WiFi-3D-Human-Pose-Estimation\results\Baseline HPE-Li Weight\S1_best.pt"
PROJECT_ROOT = r"D:\Thesis_Docs\WiFi-3D-Human-Pose-Estimation"  # Để import model/
RGB_DIR      = r"D:\RGB_image-MMFi\MMFi_Defaced_RGB"            # Ảnh RGB (để trống nếu không có)
PORT         = 8082
# ============================================================

# ============================================================
#  SKELETON CONFIG
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
# GT skeleton colors (teal/blue-green)
GT_JOINT_COLORS = [
    0x00E5FF, 0x69F0AE, 0x69F0AE, 0x69F0AE,
    0x40C4FF, 0x40C4FF, 0x40C4FF,
    0x00E5FF, 0x00E5FF, 0x00E5FF, 0x00E5FF,
    0x69F0AE, 0x69F0AE, 0x69F0AE,
    0x40C4FF, 0x40C4FF, 0x40C4FF,
]
GT_EDGE_COLORS = [
    0x69F0AE, 0x69F0AE, 0x69F0AE,
    0x40C4FF, 0x40C4FF, 0x40C4FF,
    0x00E5FF, 0x00E5FF, 0x00E5FF, 0x00E5FF,
    0x69F0AE, 0x69F0AE, 0x69F0AE,
    0x40C4FF, 0x40C4FF, 0x40C4FF,
]
# Predicted skeleton colors (orange/red)
PRED_JOINT_COLORS = [
    0xFF6D00, 0xFF4081, 0xFF4081, 0xFF4081,
    0xFFD740, 0xFFD740, 0xFFD740,
    0xFF6D00, 0xFF6D00, 0xFF6D00, 0xFF6D00,
    0xFF4081, 0xFF4081, 0xFF4081,
    0xFFD740, 0xFFD740, 0xFFD740,
]
PRED_EDGE_COLORS = [
    0xFF4081, 0xFF4081, 0xFF4081,
    0xFFD740, 0xFFD740, 0xFFD740,
    0xFF6D00, 0xFF6D00, 0xFF6D00, 0xFF6D00,
    0xFF4081, 0xFF4081, 0xFF4081,
    0xFFD740, 0xFFD740, 0xFFD740,
]
LEGEND = [
    {"label": "GT — Core/Head",   "color": "#00E5FF"},
    {"label": "GT — L.Arm/Leg",   "color": "#69F0AE"},
    {"label": "GT — R.Arm/Leg",   "color": "#40C4FF"},
    {"label": "Pred — Core/Head", "color": "#FF6D00"},
    {"label": "Pred — L.Arm/Leg", "color": "#FF4081"},
    {"label": "Pred — R.Arm/Leg", "color": "#FFD740"},
]


# ============================================================
#  MODEL — LOAD ONCE AT STARTUP
# ============================================================
def load_model(model_path, project_root):
    """Load DSKNetTransMMFI model from .pt checkpoint."""
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    try:
        import torch
        from model import DSKNetTransMMFI
        model = DSKNetTransMMFI()
        state = torch.load(model_path, map_location="cpu")
        model.load_state_dict(state)
        model.eval()
        print(f"  Model loaded: {model_path}")
        return model
    except Exception as e:
        print(f"  [WARN] Không load được model: {e}")
        print(f"         Sẽ chỉ hiển thị Ground Truth skeleton.")
        return None


# ============================================================
#  INFERENCE
# ============================================================
def run_inference(mat_files, model):
    """
    Chạy model trên list .mat files → predicted keypoints.
    Model output: (batch, 17, 2) — chỉ X,Y.
    Returns ndarray shape (T, 17, 3) với Z=0 để hiển thị 3D.

    QUAN TRỌNG: CSI phải được normalize per-frame theo đúng cách
    mmfi.py làm trong training (min-max → [0,1] sau khi xử lý NaN).
    """
    if model is None:
        return None
    try:
        import torch
        all_preds = []
        with torch.no_grad():
            for mf in mat_files:
                m = sio.loadmat(str(mf))
                amp = m["CSIamp"].astype(np.float32)   # (3, 114, 10)

                # ── Bước 1: xử lý inf → NaN (giống mmfi.py) ──
                amp[np.isinf(amp)] = np.nan
                # ── Bước 2: thay NaN bằng mean của cột đó ──
                for i in range(amp.shape[2]):           # 10 packets
                    col = amp[:, :, i]
                    n_nan = np.count_nonzero(np.isnan(col))
                    if n_nan > 0:
                        col_mean = col[~np.isnan(col)].mean()
                        col[np.isnan(col)] = col_mean if not np.isnan(col_mean) else 0.0
                # ── Bước 3: min-max normalize → [0, 1] ──
                a_min, a_max = amp.min(), amp.max()
                if a_max > a_min:
                    amp = (amp - a_min) / (a_max - a_min)
                else:
                    amp = np.zeros_like(amp)

                x = torch.from_numpy(amp).unsqueeze(0)  # (1, 3, 114, 10)
                # Model returns tuple: (output, time_sum)
                result = model(x)
                if isinstance(result, tuple):
                    pred_tensor = result[0]              # (1, 17, 2)
                else:
                    pred_tensor = result
                pred_np = pred_tensor.cpu().numpy()      # (1, 17, 2)
                if pred_np.ndim == 2:
                    pred_np = pred_np.reshape(1, 17, -1)
                kpts_2d = pred_np[0]                     # (17, 2) — X, Y only
                # Pad Z=0 for 3D display
                kpts_3d = np.concatenate([kpts_2d, np.zeros((17, 1))], axis=-1)  # (17, 3)
                all_preds.append(kpts_3d)
        return np.stack(all_preds, axis=0)               # (T, 17, 3)
    except Exception as e:
        print(f"  [WARN] Inference thất bại: {e}")
        import traceback; traceback.print_exc()
        return None



def get_mat_files(npy_path):
    """Lấy list file .mat CSI cùng action với file .npy."""
    npy = Path(npy_path)
    parts = npy.parts
    env, sub, act = parts[-4], parts[-3], parts[-2]
    csi_dir = Path(DATASET_ROOT) / env / sub / act / "wifi-csi"
    if not csi_dir.exists():
        return []
    return sorted(csi_dir.glob("frame*.mat"))


# ============================================================
#  DATA HELPERS
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
    npy = Path(npy_path)
    parts = npy.parts
    env, sub, act = parts[-4], parts[-3], parts[-2]
    csi_dir = Path(DATASET_ROOT) / env / sub / act / "wifi-csi"
    if not csi_dir.exists():
        return None
    mat_files = sorted(csi_dir.glob("frame*.mat"))
    if not mat_files:
        return None
    print(f"  Loading {len(mat_files)} CSI frames...")
    all_amp = []
    for mf in mat_files:
        m = sio.loadmat(str(mf))
        amp = m["CSIamp"]
        amp_avg = amp.mean(axis=2)
        all_amp.append(amp_avg)
    heatmap = np.stack(all_amp, axis=0)
    heatmap = np.nan_to_num(heatmap, nan=0.0, posinf=0.0, neginf=0.0)
    return {
        "heatmap": np.round(heatmap, 2).tolist(),
        "num_frames": len(mat_files),
        "num_antennas": 3,
        "num_subcarriers": 114,
        "amp_min": round(float(heatmap.min()), 2),
        "amp_max": round(float(heatmap.max()), 2),
    }


def compute_mpjpe_per_frame(gt_data, pred_data):
    """
    gt_data: list-of-list (T, 1, 17, 3) (from load_pose_data)
    pred_data: (T, 17, 3) numpy — model only predicts XY so Z=0
    Returns MPJPE computed on XY only (2D MPJPE) per frame.
    """
    if pred_data is None:
        return None
    mpjpe = []
    T = min(len(gt_data), len(pred_data))
    for t in range(T):
        gt_kpts = np.array(gt_data[t][0])[:, :2]  # (17, 2) — XY only
        pr_kpts = pred_data[t][:, :2]              # (17, 2)
        err = float(np.mean(np.sqrt(np.sum((gt_kpts - pr_kpts) ** 2, axis=-1))))
        mpjpe.append(round(err, 4))
    return mpjpe


# ============================================================
#  MODULE-LEVEL GLOBAL MODEL + LOADING STATE
# ============================================================
MODEL = None
_loading_state = {"ready": False, "status": "init"}  # shared across handler instances


# ============================================================
#  HTTP HANDLER
# ============================================================
class InferenceHandler(http.server.SimpleHTTPRequestHandler):
    gt_data      = None
    frame_names  = []
    current_npy  = NPY_PATH
    csi_data     = None
    pred_data    = None   # (T, 17, 3) numpy
    mpjpe_frames = None   # list of float per frame

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
                "keypoint_names":  KEYPOINT_NAMES,
                "edges":           EDGES,
                "gt_joint_colors": [f"#{c:06X}" for c in GT_JOINT_COLORS],
                "gt_edge_colors":  [f"#{c:06X}" for c in GT_EDGE_COLORS],
                "pred_joint_colors": [f"#{c:06X}" for c in PRED_JOINT_COLORS],
                "pred_edge_colors":  [f"#{c:06X}" for c in PRED_EDGE_COLORS],
                "legend":          LEGEND,
                "has_model":       MODEL is not None,
            })

        elif parsed.path == "/api/data":
            if InferenceHandler.gt_data is None:
                self._send_json({"error": "Data not ready yet", "loading": True}, 503)
                return
            self._send_json({
                "frames":      InferenceHandler.gt_data,
                "num_frames":  len(InferenceHandler.gt_data),
                "frame_names": InferenceHandler.frame_names,
                "current_npy": InferenceHandler.current_npy,
            })

        elif parsed.path == "/api/predict":
            if InferenceHandler.pred_data is not None:
                # Convert to same format as /api/data: list of [[person_0_kpts]]
                pred_list = []
                for t in range(len(InferenceHandler.pred_data)):
                    kpts = InferenceHandler.pred_data[t].tolist()  # (17, 3)
                    pred_list.append([kpts])                        # 1 person
                self._send_json({
                    "frames":      pred_list,
                    "num_frames":  len(pred_list),
                    "mpjpe_frames": InferenceHandler.mpjpe_frames,
                })
            else:
                self._send_json({"error": "No prediction available (model not loaded or inference failed)"}, 404)

        elif parsed.path == "/api/csi":
            if InferenceHandler.csi_data:
                self._send_json(InferenceHandler.csi_data)
            else:
                self._send_json({"error": "No CSI data"}, 404)

        elif parsed.path == "/api/status":
            self._send_json({
                "ready":  _loading_state["ready"],
                "status": _loading_state["status"],
            })

        elif parsed.path == "/api/reload":
            npy = qs.get("path", [""])[0]
            if not npy:
                self._send_json({"error": "Missing ?path="}, 400)
                return
            try:
                data, names = load_pose_data(npy)
                InferenceHandler.gt_data     = data
                InferenceHandler.frame_names = names
                InferenceHandler.current_npy = npy
                InferenceHandler.csi_data    = load_csi_data(npy)
                # Re-run inference
                mat_files = get_mat_files(npy)
                pred = run_inference(mat_files, MODEL)
                InferenceHandler.pred_data    = pred
                InferenceHandler.mpjpe_frames = compute_mpjpe_per_frame(data, pred)
                self._send_json({"ok": True, "num_frames": len(data)})
            except Exception as e:
                self._send_json({"error": str(e)}, 500)

        elif parsed.path == "/api/rgb":
            frame_idx = int(qs.get("frame", ["0"])[0])
            npy_parts = Path(InferenceHandler.current_npy).parts
            env, sub, act = npy_parts[-4], npy_parts[-3], npy_parts[-2]
            img_path = Path(RGB_DIR) / env / sub / act / "rgb" / f"frame{frame_idx+1:03d}.png"
            try:
                self.send_response(200)
                self.send_header("Content-Type", "image/png")
                self.send_header("Content-Length", str(os.path.getsize(str(img_path))))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                with open(img_path, "rb") as f:
                    self.wfile.write(f.read())
            except FileNotFoundError:
                self.send_error(404)

        else:
            super().do_GET()


# ============================================================
#  ENTRY POINT
# ============================================================
def _load_data_thread():
    """Load model + data trong background sau khi server đã sẵn sàng."""
    global MODEL
    try:
        _loading_state["status"] = "Loading model..."
        MODEL = load_model(MODEL_PATH, PROJECT_ROOT)

        _loading_state["status"] = "Loading ground truth..."
        data, names = load_pose_data(NPY_PATH)
        InferenceHandler.gt_data     = data
        InferenceHandler.frame_names = names
        InferenceHandler.current_npy = NPY_PATH
        print(f"  GT skeleton  : {len(data)} frames")

        _loading_state["status"] = "Loading CSI data..."
        InferenceHandler.csi_data = load_csi_data(NPY_PATH)

        if MODEL is not None:
            _loading_state["status"] = "Running inference..."
            print("  Running inference...")
            mat_files = get_mat_files(NPY_PATH)
            pred = run_inference(mat_files, MODEL)
            InferenceHandler.pred_data    = pred
            InferenceHandler.mpjpe_frames = compute_mpjpe_per_frame(data, pred)
            if pred is not None:
                print(f"  Inference OK : {len(pred)} frames predicted")
            else:
                print("  Inference failed — only GT will be shown")

        _loading_state["ready"]  = True
        _loading_state["status"] = "ready"
        print("  ✔ Data ready. Refresh browser if needed.")
    except Exception as e:
        _loading_state["status"] = f"ERROR: {e}"
        print(f"  [ERROR] {e}")
        import traceback; traceback.print_exc()


def main():
    print("=" * 60)
    print("  MMFi Inference Viewer — GT vs Predicted")
    print("=" * 60)
    print(f"  Dataset root : {DATASET_ROOT}")
    print(f"  Model        : {MODEL_PATH}")

    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    # 1️⃣  Start HTTP server FIRST in a background daemon thread
    http.server.HTTPServer.allow_reuse_address = True   
    server = http.server.HTTPServer(("localhost", PORT), InferenceHandler)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()

    url = f"http://localhost:{PORT}/index.html"
    print(f"\n  Server running: {url}")
    print("  ⏳ Loading data in background... browser will be ready shortly.")
    print("  Press Ctrl+C to stop\n")

    # 2️⃣  Open browser immediately
    webbrowser.open(url)

    # 3️⃣  Load all data in a background thread
    loader = threading.Thread(target=_load_data_thread, daemon=True)
    loader.start()

    # 4️⃣  Keep main thread alive
    try:
        loader.join()          # wait for loading to finish
        server_thread.join()   # then block on server (runs until Ctrl+C)
    except KeyboardInterrupt:
        print("\n  Server stopped.")
        server.shutdown()
        server.server_close()


if __name__ == "__main__":
    main()
