#!/bin/bash
# ============================================================
# Setup & Train Script — WiFi 3D Human Pose Estimation
# Dùng trên FPT AI Factory GPU VM
# Chạy: bash setup_and_train.sh
# ============================================================

set -e  # Dừng nếu có lỗi

REPO_URL="https://github.com/cestthanh/WiFi-3D-Human-Pose-Estimation.git"
DATASET_FOLDER_ID="1ujsqUOb51niwzQ9CwuHPzLkzBXu5Sm8e"
DATASET_DIR="$HOME/MMFi_Lite"
PROJECT_DIR="$HOME/WiFi-3D-Human-Pose-Estimation"
CONFIG="configs/config_s1.yaml"

echo "============================================="
echo "  WiFi HPE - FPT AI Factory Setup Script"
echo "============================================="

# ---- 1. Clone code ----
if [ ! -d "$PROJECT_DIR" ]; then
    echo "[1/4] Cloning code từ GitHub..."
    git clone "$REPO_URL" "$PROJECT_DIR"
else
    echo "[1/4] Code đã có, pull bản mới nhất..."
    git -C "$PROJECT_DIR" pull
fi

# ---- 2. Cài dependencies ----
echo "[2/4] Cài Python dependencies..."
pip install -q -r "$PROJECT_DIR/requirements.txt"

# ---- 3. Dataset ----
if [ ! -d "$DATASET_DIR" ]; then
    echo "[3/4] ❌ Lỗi: Chưa tìm thấy dataset tại $DATASET_DIR."
    echo "       Vui lòng unzip file dataset bạn upload qua scp ra đây trước."
    exit 1
else
    echo "[3/4] ✅ Đã tìm thấy dataset tại: $DATASET_DIR"
fi

# Kiểm tra dataset
MAT_COUNT=$(find "$DATASET_DIR" -name "*.mat" | wc -l)
echo "     Số file .mat tìm thấy: $MAT_COUNT"

# ---- 4. Bắt đầu training ----
echo "[4/4] Bắt đầu training S1 (random split)..."
cd "$PROJECT_DIR"

nohup python train_baseline.py \
    --config "$CONFIG" \
    --dataset_root "$DATASET_DIR" \
    --gpu 0 \
    > "$HOME/train_s1.log" 2>&1 &

TRAIN_PID=$!
echo "============================================="
echo "  Training đang chạy nền! PID: $TRAIN_PID"
echo "  Xem log: tail -f $HOME/train_s1.log"
echo "  Dừng training: kill $TRAIN_PID"
echo "============================================="
