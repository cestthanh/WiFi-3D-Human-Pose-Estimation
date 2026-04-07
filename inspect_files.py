"""
Xem chi tiết nội dung bên trong file .mat (CSI) và .npy (keypoint)
"""
import h5py
import numpy as np
import os

DATA_ROOT = r"D:\Thesis\WiFi-3D-Human-Pose-Estimation\Data\Person-in-WiFi-3D"

def explore_h5(obj, prefix=""):
    """Recursively explore h5py structure"""
    if isinstance(obj, h5py.File) or isinstance(obj, h5py.Group):
        for key in obj.keys():
            print(f"   {prefix}{key}/")
            explore_h5(obj[key], prefix + "  ")
    elif isinstance(obj, h5py.Dataset):
        print(f"   {prefix}→ shape={obj.shape}, dtype={obj.dtype}")

# Lấy 3 sample
with open(os.path.join(DATA_ROOT, "train_data", "train_data_list.txt")) as f:
    samples = [f.readline().strip() for _ in range(3)]

for sample_name in samples:
    print("=" * 70)
    print(f"📁 SAMPLE: {sample_name}")
    print("=" * 70)
    
    # ===================== FILE .MAT (CSI) =====================
    mat_path = os.path.join(DATA_ROOT, "train_data", "csi", f"{sample_name}.mat")
    print(f"\n📡 FILE .MAT: {os.path.basename(mat_path)}")
    print(f"   File size: {os.path.getsize(mat_path):,} bytes")
    
    f = h5py.File(mat_path, 'r')
    
    # Show full structure
    print(f"\n   === Cấu trúc bên trong file .mat ===")
    explore_h5(f)
    
    # Read CSI data
    csi_out = f['csi_out']
    
    # Check if compound dataset (real+imag) or group
    if isinstance(csi_out, h5py.Dataset):
        raw = np.array(csi_out)
        print(f"\n   Raw data shape: {raw.shape}, dtype: {raw.dtype}")
        
        if raw.dtype.names:
            # Compound type with 'real' and 'imag' fields
            real_part = raw['real']
            imag_part = raw['imag']
        else:
            # Try as complex directly
            real_part = np.real(raw)
            imag_part = np.imag(raw)
    else:
        # Group with separate real/imag datasets
        real_part = np.array(csi_out['real'])
        imag_part = np.array(csi_out['imag'])
    
    print(f"\n   Real part shape: {real_part.shape}  dtype: {real_part.dtype}")
    print(f"   Imag part shape: {imag_part.shape}  dtype: {imag_part.dtype}")
    
    csi = real_part + imag_part * 1j
    print(f"\n   Complex CSI shape (trước transpose): {csi.shape}")
    
    # Transpose to standard format
    csi = csi.transpose(3, 2, 1, 0)
    print(f"   Complex CSI shape (sau transpose):  {csi.shape}")
    print(f"   → (3 receivers, 3 antennas, 30 subcarriers, 20 timesteps)")
    
    # Print actual values
    print(f"\n   === Giá trị thực tế (RX=0, Antenna=0, 5 subcarriers đầu, 5 timesteps đầu) ===")
    
    print(f"\n   Amplitude (cường độ tín hiệu):")
    amp = np.abs(csi[0, 0, :5, :5])
    for i in range(5):
        vals = "  ".join([f"{v:7.2f}" for v in amp[i]])
        print(f"     Subcarrier {i}: [{vals}]")
    
    print(f"\n   Phase (pha, radians):")
    phase = np.angle(csi[0, 0, :5, :5])
    for i in range(5):
        vals = "  ".join([f"{v:7.4f}" for v in phase[i]])
        print(f"     Subcarrier {i}: [{vals}]")
    
    print(f"\n   Thống kê toàn bộ CSI:")
    print(f"     Amplitude: min={np.abs(csi).min():.4f}  max={np.abs(csi).max():.4f}  mean={np.abs(csi).mean():.4f}")
    print(f"     Phase:     min={np.angle(csi).min():.4f}  max={np.angle(csi).max():.4f}")
    f.close()
    
    # ===================== FILE .NPY (Keypoint) =====================
    npy_path = os.path.join(DATA_ROOT, "train_data", "keypoint", f"{sample_name}.npy")
    print(f"\n🦴 FILE .NPY: {os.path.basename(npy_path)}")
    print(f"   File size: {os.path.getsize(npy_path):,} bytes")
    
    kp = np.load(npy_path)
    print(f"   Shape: {kp.shape}   dtype: {kp.dtype}")
    print(f"   → ({kp.shape[0]} người, {kp.shape[1]} joints, {kp.shape[2]} tọa độ [x, y, z])")
    
    joint_names = [
        "Head", "Neck", "R-Shoulder", "R-Elbow", "R-Wrist",
        "L-Shoulder", "L-Elbow", "L-Wrist", "R-Hip", "R-Knee", 
        "R-Ankle", "L-Hip", "L-Knee", "L-Ankle"
    ]
    
    for person_idx in range(kp.shape[0]):
        print(f"\n   === Person {person_idx + 1} ===")
        print(f"   {'#':<4} {'Joint':<15} {'X':>10} {'Y':>10} {'Z':>10}")
        print(f"   {'-'*53}")
        for j, name in enumerate(joint_names):
            x, y, z = kp[person_idx, j]
            print(f"   {j:<4} {name:<15} {x:>10.4f} {y:>10.4f} {z:>10.4f}")
    
    print("\n")
