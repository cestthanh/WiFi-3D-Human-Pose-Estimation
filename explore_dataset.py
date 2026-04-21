"""
explore_dataset.py — Kiểm tra cấu trúc và format dữ liệu MMFi Dataset

Chạy từ thư mục project:
    python explore_dataset.py --dataset_root "D:\\MMFi-dataset\\MMFi_Dataset"

Script này sẽ:
1. Kiểm tra cấu trúc thư mục (E/S/A)
2. Load và in shape của 1 file CSI .mat → xác nhận format (3, 114, 10)
3. Load và in shape của ground_truth.npy → xác nhận (T, 17, 3)
4. Đếm tổng số frames toàn dataset theo từng Environment
5. Kiểm tra DataLoader hoạt động được không (1 batch)
"""

import argparse
import os
import sys

import numpy as np
import scipy.io as sio

# ----------------------------------------------------------------
# 1. Argument
# ----------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--dataset_root", type=str,
                    default=r"D:\MMFi-dataset\MMFi_Dataset",
                    help="Đường dẫn thư mục gốc MMFi Dataset")
args = parser.parse_args()

DATASET_ROOT = args.dataset_root
print("=" * 60)
print("MMFi Dataset Explorer")
print(f"Root: {DATASET_ROOT}")
print("=" * 60)

# ----------------------------------------------------------------
# 2. Kiểm tra cấu trúc thư mục
# ----------------------------------------------------------------
print("\n[1] Cấu trúc dataset:")
envs = sorted([d for d in os.listdir(DATASET_ROOT)
               if os.path.isdir(os.path.join(DATASET_ROOT, d)) and d.startswith("E")])
print(f"    Environments: {envs}  ({len(envs)} envs)")

for env in envs:
    env_path = os.path.join(DATASET_ROOT, env)
    subjects = sorted([d for d in os.listdir(env_path)
                       if os.path.isdir(os.path.join(env_path, d))])
    # Lấy số actions từ subject đầu tiên
    first_subj_path = os.path.join(env_path, subjects[0])
    actions = sorted([d for d in os.listdir(first_subj_path)
                      if os.path.isdir(os.path.join(first_subj_path, d))])
    print(f"    {env}: {len(subjects)} subjects ({subjects[0]}..{subjects[-1]}), "
          f"{len(actions)} actions ({actions[0]}..{actions[-1]})")

# ----------------------------------------------------------------
# 3. Kiểm tra 1 file CSI .mat
# ----------------------------------------------------------------
print("\n[2] Kiểm tra 1 file CSI .mat (E01/S01/A01/wifi-csi/frame001.mat):")
sample_csi_path = os.path.join(DATASET_ROOT, "E01", "S01", "A01",
                               "wifi-csi", "frame001.mat")
if os.path.exists(sample_csi_path):
    try:
        # Thử load bằng scipy.io (dạng mat v5)
        mat = sio.loadmat(sample_csi_path)
        keys = [k for k in mat.keys() if not k.startswith("_")]
        print(f"    Keys trong .mat: {keys}")
        for k in keys:
            arr = mat[k]
            print(f"    '{k}': shape={arr.shape}, dtype={arr.dtype}")
    except Exception:
        # Nếu là HDF5 mat (v7.3) dùng h5py
        try:
            import h5py
            with h5py.File(sample_csi_path, "r") as f:
                keys = list(f.keys())
                print(f"    Keys trong .mat (HDF5): {keys}")
                for k in keys:
                    arr = np.array(f[k])
                    print(f"    '{k}': shape={arr.shape}, dtype={arr.dtype}")
        except Exception as e:
            print(f"    ❌ Không thể đọc file: {e}")
else:
    print(f"    ❌ File không tồn tại: {sample_csi_path}")

# ----------------------------------------------------------------
# 4. Kiểm tra ground_truth.npy
# ----------------------------------------------------------------
print("\n[3] Kiểm tra ground_truth.npy (E01/S01/A01):")
gt_path = os.path.join(DATASET_ROOT, "E01", "S01", "A01", "ground_truth.npy")
if os.path.exists(gt_path):
    gt = np.load(gt_path)
    print(f"    Shape: {gt.shape}   (kỳ vọng: (T, 17, 3))")
    print(f"    Dtype: {gt.dtype}")
    print(f"    Value range: [{gt.min():.4f}, {gt.max():.4f}]")
    print(f"    Sample frame 0, joint 0: {gt[0, 0]}")
    if gt.shape[1] == 17 and gt.shape[2] == 3:
        print("    ✅ Ground truth format đúng!")
    else:
        print(f"    ⚠️ Format khác kỳ vọng! Nhận được shape: {gt.shape}")
else:
    print(f"    ❌ Không tìm thấy: {gt_path}")

# ----------------------------------------------------------------
# 5. Đếm tổng frames theo Environment
# ----------------------------------------------------------------
print("\n[4] Thống kê tổng số frames:")
total_all = 0
for env in envs:
    env_path = os.path.join(DATASET_ROOT, env)
    total_env = 0
    subjects = sorted([d for d in os.listdir(env_path)
                       if os.path.isdir(os.path.join(env_path, d))])
    for subj in subjects:
        subj_path = os.path.join(env_path, subj)
        actions = sorted([d for d in os.listdir(subj_path)
                          if os.path.isdir(os.path.join(subj_path, d))])
        for act in actions:
            csi_folder = os.path.join(subj_path, act, "wifi-csi")
            if os.path.isdir(csi_folder):
                n_frames = len([f for f in os.listdir(csi_folder) if f.endswith(".mat")])
                total_env += n_frames
    total_all += total_env
    print(f"    {env}: {total_env:,} frames ({total_env//1000}k)")
print(f"    TỔNG: {total_all:,} frames ({total_all//1000}k)")

# ----------------------------------------------------------------
# 6. Thử DataLoader (nếu có torch + dataset_lib)
# ----------------------------------------------------------------
print("\n[5] Thử DataLoader (1 batch):")
try:
    import yaml
    import torch
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from dataset_lib import make_dataset, make_dataloader

    config_path = os.path.join(os.path.dirname(__file__), "configs", "config_s1.yaml")
    with open(config_path, encoding='utf-8') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    train_ds, val_ds = make_dataset(DATASET_ROOT, config)
    rng = torch.manual_seed(42)
    loader = make_dataloader(train_ds, is_training=False, generator=rng,
                             batch_size=2, num_workers=0)
    batch = next(iter(loader))

    csi  = batch["input_wifi-csi"]
    pose = batch["output"]
    print(f"    CSI  shape: {tuple(csi.shape)}   (kỳ vọng: (B, 3, 114, 10))")
    print(f"    Pose shape: {tuple(pose.shape)}  (kỳ vọng: (B, 17, 3))")
    print(f"    CSI  dtype: {csi.dtype}")
    print(f"    Pose dtype: {pose.dtype}")
    print("    ✅ DataLoader hoạt động bình thường!")

except ImportError as e:
    print(f"    ⚠️ Bỏ qua test DataLoader (thiếu thư viện: {e})")
except Exception as e:
    print(f"    ❌ DataLoader lỗi: {e}")

print("\n" + "=" * 60)
print("Hoàn thành kiểm tra dataset!")
print("=" * 60)
