"""
verify_dataset.py
=================
Kiểm tra shape và tính hợp lệ của toàn bộ dataset Person-in-WiFi-3D.
- CSI: mỗi .mat phải load được và có shape (Tx, Rx, subcarriers, time)
- Keypoint: mỗi .npy phải có shape (N_persons, 14, 3)
- data_list.txt: mọi tên trong list phải có file tương ứng
"""

import os
import sys
import glob
import numpy as np
from pathlib import Path
from collections import Counter

# ============================================================
#  CONFIG
# ============================================================
BASE = r"D:\Thesis\WiFi-3D-Human-Pose-Estimation\Data\Person-in-WiFi-3D"

SPLITS = ["train_data", "test_data"]

# ============================================================

def check_csi(csi_dir, sample_names=None, max_sample=5):
    """Kiểm tra CSI .mat files."""
    import h5py

    mat_files = sorted(glob.glob(os.path.join(csi_dir, "*.mat")))
    if not mat_files:
        print(f"  [ERROR] Không tìm thấy .mat file trong: {csi_dir}")
        return False

    print(f"  Tổng số .mat files: {len(mat_files)}")

    shape_counter = Counter()
    errors = []

    # Kiểm tra toàn bộ (nhanh - chỉ open header, không load data)
    print(f"  Đang kiểm tra {len(mat_files)} files... ", end="", flush=True)
    for i, f in enumerate(mat_files):
        try:
            with h5py.File(f, 'r') as h:
                if 'csi_out' not in h:
                    errors.append(f"  [ERR] Thiếu key 'csi_out': {Path(f).name}")
                    continue
                csi_raw = h['csi_out']
                if hasattr(csi_raw.dtype, 'names') and csi_raw.dtype.names:
                    shape = csi_raw['real'].shape
                else:
                    shape = csi_raw.shape
                shape_counter[shape] += 1
        except Exception as e:
            errors.append(f"  [ERR] {Path(f).name}: {e}")
        if (i+1) % 500 == 0:
            print(f"{i+1}...", end="", flush=True)
    print("xong!")

    print(f"\n  Phân bố shape CSI (H5 raw, trước transpose):")
    for shape, count in shape_counter.most_common():
        print(f"    {shape} → {count} files")

    # Load đầy đủ vài file để xem shape sau transpose
    print(f"\n  Chi tiết {min(max_sample, len(mat_files))} file đầu (load đầy đủ):")
    for f in mat_files[:max_sample]:
        try:
            with h5py.File(f, 'r') as h:
                csi_raw = h['csi_out']
                if hasattr(csi_raw.dtype, 'names') and csi_raw.dtype.names:
                    real = csi_raw['real'][()]
                    imag = csi_raw['imag'][()]
                    csi = real + imag * 1j
                else:
                    csi = np.array(csi_raw[()])
                # Áp dụng transpose như trong wifi_pose.py
                csi = np.array(csi).transpose(3, 2, 1, 0)
                print(f"    {Path(f).name}: raw_shape={real.shape} → sau_transpose={csi.shape}")
                print(f"      dtype={csi.dtype}, min_amp={np.abs(csi).min():.4f}, max_amp={np.abs(csi).max():.4f}")
        except Exception as e:
            print(f"    {Path(f).name}: [LOAD ERROR] {e}")

    if errors:
        print(f"\n  [!] {len(errors)} lỗi phát hiện:")
        for err in errors[:10]:
            print(f"    {err}")
    else:
        print(f"\n  ✓ Tất cả {len(mat_files)} .mat files hợp lệ")

    return len(errors) == 0


def check_keypoints(kpt_dir, max_sample=5):
    """Kiểm tra keypoint .npy files."""
    npy_files = sorted(glob.glob(os.path.join(kpt_dir, "*.npy")))
    if not npy_files:
        print(f"  [ERROR] Không tìm thấy .npy file trong: {kpt_dir}")
        return False

    print(f"  Tổng số .npy files: {len(npy_files)}")

    shape_counter = Counter()
    errors = []
    n_persons_counter = Counter()

    print(f"  Đang kiểm tra {len(npy_files)} files... ", end="", flush=True)
    for i, f in enumerate(npy_files):
        try:
            data = np.load(f)
            shape_counter[data.shape] += 1
            if data.ndim == 3:
                n_persons_counter[data.shape[0]] += 1
            elif data.ndim == 2:
                n_persons_counter[1] += 1  # single person
        except Exception as e:
            errors.append(f"  [ERR] {Path(f).name}: {e}")
        if (i+1) % 500 == 0:
            print(f"{i+1}...", end="", flush=True)
    print("xong!")

    print(f"\n  Phân bố shape keypoint:")
    for shape, count in shape_counter.most_common():
        print(f"    {shape} → {count} files")

    print(f"\n  Số người per frame:")
    for n, count in sorted(n_persons_counter.items()):
        print(f"    {n} người → {count} frames")

    # Load vài file chi tiết
    print(f"\n  Chi tiết {min(max_sample, len(npy_files))} file đầu:")
    for f in npy_files[:max_sample]:
        try:
            data = np.load(f)
            print(f"    {Path(f).name}: shape={data.shape}, "
                  f"min={data.min():.3f}, max={data.max():.3f}")
        except Exception as e:
            print(f"    {Path(f).name}: [LOAD ERROR] {e}")

    if errors:
        print(f"\n  [!] {len(errors)} lỗi:")
        for err in errors[:10]:
            print(f"    {err}")
    else:
        print(f"\n  ✓ Tất cả {len(npy_files)} .npy files hợp lệ")

    return len(errors) == 0


def check_data_list(data_dir, split):
    """Kiểm tra data_list.txt — mọi tên phải có đủ CSI + keypoint."""
    list_path = os.path.join(data_dir, f"{split}_data_list.txt")
    csi_dir = os.path.join(data_dir, "csi")
    kpt_dir = os.path.join(data_dir, "keypoint")

    if not os.path.exists(list_path):
        print(f"  [ERROR] Không tìm thấy: {list_path}")
        return

    with open(list_path) as f:
        names = [line.strip().split()[0] for line in f if line.strip()]

    print(f"  Tổng số sample trong list: {len(names)}")

    missing_csi, missing_kpt = [], []
    for name in names:
        if not os.path.exists(os.path.join(csi_dir, f"{name}.mat")):
            missing_csi.append(name)
        if not os.path.exists(os.path.join(kpt_dir, f"{name}.npy")):
            missing_kpt.append(name)

    if missing_csi:
        print(f"  [!] Thiếu {len(missing_csi)} CSI files:")
        for n in missing_csi[:5]:
            print(f"      {n}.mat")
        if len(missing_csi) > 5:
            print(f"      ... và {len(missing_csi)-5} file khác")
    else:
        print(f"  ✓ Tất cả {len(names)} CSI files tồn tại")

    if missing_kpt:
        print(f"  [!] Thiếu {len(missing_kpt)} keypoint files:")
        for n in missing_kpt[:5]:
            print(f"      {n}.npy")
    else:
        print(f"  ✓ Tất cả {len(names)} keypoint files tồn tại")

    # Kiểm tra files thừa (có file nhưng không trong list)
    csi_names_on_disk = {Path(f).stem for f in glob.glob(os.path.join(csi_dir, "*.mat"))}
    kpt_names_on_disk = {Path(f).stem for f in glob.glob(os.path.join(kpt_dir, "*.npy"))}
    names_in_list = set(names)

    extra_csi = csi_names_on_disk - names_in_list
    extra_kpt = kpt_names_on_disk - names_in_list
    if extra_csi:
        print(f"  [i] {len(extra_csi)} CSI files trên disk nhưng KHÔNG trong list (bình thường nếu ít)")
    if extra_kpt:
        print(f"  [i] {len(extra_kpt)} keypoint files trên disk nhưng KHÔNG trong list")


def main():
    print("=" * 60)
    print("  Person-in-WiFi-3D Dataset Verification")
    print("=" * 60)

    for split in SPLITS:
        data_dir = os.path.join(BASE, split)
        if not os.path.exists(data_dir):
            print(f"\n[SKIP] Không tìm thấy: {data_dir}")
            continue

        csi_dir = os.path.join(data_dir, "csi")
        kpt_dir = os.path.join(data_dir, "keypoint")

        print(f"\n{'='*60}")
        print(f"  SPLIT: {split}")
        print(f"{'='*60}")

        # --- CSI ---
        print(f"\n[CSI] {csi_dir}")
        if os.path.exists(csi_dir):
            check_csi(csi_dir)
        else:
            print(f"  [SKIP] Thư mục không tồn tại")

        # --- Keypoint ---
        print(f"\n[KEYPOINT] {kpt_dir}")
        if os.path.exists(kpt_dir):
            check_keypoints(kpt_dir)
        else:
            print(f"  [SKIP] Thư mục không tồn tại")

        # --- Data list ---
        print(f"\n[DATA LIST] {split}_data_list.txt")
        check_data_list(data_dir, split)

    print(f"\n{'='*60}")
    print("  Verification hoàn thành!")
    print("=" * 60)


if __name__ == "__main__":
    main()
