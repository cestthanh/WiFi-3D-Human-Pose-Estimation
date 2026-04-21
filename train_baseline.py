#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_baseline.py — V1 Baseline: DSKNetTransMMFI (HPE-Li) trên MMFi dataset
Chạy: python train_baseline.py --config configs/config_s1.yaml --dataset_root D:/path/to/MMFi

Đây là baseline không có domain adaptation — để chứng minh cross-domain gap (S3 drop).
"""
import argparse
import os
import sys

import numpy as np
import torch
import yaml
from sklearn.model_selection import train_test_split
from torch import nn
from tqdm import tqdm

from dataset_lib import make_dataloader, make_dataset
from model import DSKNetTransMMFI
from utils import calulate_error, compute_pck_pckh

# -------------------------------------------------------------------
# 1. Argument Parsing
# -------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="V1 Baseline Training — MMFi WiFi HPE")
    parser.add_argument("--config", type=str, required=True,
                        help="Đường dẫn file config YAML (e.g., configs/config_s1.yaml)")
    parser.add_argument("--dataset_root", type=str, default="./Data/mmfi",
                        help="Đường dẫn thư mục gốc MMFi dataset")
    parser.add_argument("--gpu", type=str, default="0", help="GPU id (default: 0)")
    return parser.parse_args()


# -------------------------------------------------------------------
# 2. Evaluation (chỉ tính MPJPE và PA-MPJPE trên tập val/test)
# -------------------------------------------------------------------
def evaluate(model, loader, criterion, device, split_name="val"):
    model.eval()
    metric_list = []
    loss_list = []
    pck50_list = []

    with torch.no_grad():
        for data in loader:
            csi_data = data["input_wifi-csi"].float().to(device)
            keypoint = data["output"].to(device)        # (B, 17, 3) — [x, y, conf]
            xy_gt = keypoint[:, :, 0:2]                 # (B, 17, 2)
            confidence = keypoint[:, :, 2:3]            # (B, 17, 1)

            pred_xy, _ = model(csi_data)                # (B, 2, 17) or (B, 17, 2)
            pred_xy = pred_xy.squeeze()

            loss = criterion(
                torch.mul(confidence, pred_xy),
                torch.mul(confidence, xy_gt)
            )
            loss_list.append(loss.item())
            metric_list.append(calulate_error(pred_xy.cpu(), xy_gt.cpu()))

            pred_pck = torch.transpose(pred_xy.cpu(), 1, 2)
            gt_pck = torch.transpose(xy_gt.cpu(), 1, 2)
            pck50_list.append(compute_pck_pckh(pred_pck, gt_pck, 0.5))

    mean_errors = np.mean(metric_list, axis=0) * 1000   # convert → mm
    mpjpe = mean_errors[0]
    pa_mpjpe = mean_errors[1]
    pck50_overall = np.mean(pck50_list, axis=0)[-1]     # overall (last element)
    mean_loss = np.mean(loss_list)

    print(
        f"[{split_name}] Loss: {mean_loss:.4f} | MPJPE: {mpjpe:.2f}mm "
        f"| PA-MPJPE: {pa_mpjpe:.2f}mm | PCK@50: {pck50_overall:.3f}"
    )
    return mpjpe, pa_mpjpe, pck50_overall


# -------------------------------------------------------------------
# 3. Main Training Loop
# -------------------------------------------------------------------
def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"==> Thiết bị: {device}")

    # Load config
    with open(args.config, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    print(f"==> Đã load config: {args.config}")
    print(f"    Split: {config.get('split_to_use', '?')}")

    # Dataloader
    train_dataset, val_test_dataset = make_dataset(args.dataset_root, config)
    rng = torch.manual_seed(config["init_rand_seed"])

    # Chia val/test 50-50 từ tập val gốc
    val_data, test_data = train_test_split(
        val_test_dataset, test_size=0.5, random_state=41
    )

    train_loader = make_dataloader(train_dataset, is_training=True, generator=rng,
                                   **config["train_loader"])
    val_loader   = make_dataloader(val_data, is_training=False, generator=rng,
                                   **config["val_loader"])
    test_loader  = make_dataloader(test_data, is_training=False, generator=rng,
                                   **config["test_loader"])
    print(f"==> Dữ liệu: {len(train_dataset)} train | {len(val_data)} val | {len(test_data)} test")

    # Model
    model = DSKNetTransMMFI().to(device)
    print(f"==> Model: {model._get_name()}")

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.get("learning_rate", 0.001))

    num_epochs = config.get("epoch", 100)
    n_epochs = 20
    n_epochs_decay = 30
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda ep: 1.0 - max(0, ep + 1 - n_epochs) / float(n_epochs_decay + 1)
    )

    checkpoint_dir = os.path.join(config.get("checkpoint", "./checkpoints/"), "baseline")
    os.makedirs(checkpoint_dir, exist_ok=True)

    best_mpjpe = float("inf")
    train_loss_log = []
    val_mpjpe_log = []

    # ------- Training Epochs -------
    print(f"\n{'='*60}")
    print(f"Bắt đầu training {num_epochs} epochs...")
    print(f"{'='*60}")

    for epoch in tqdm(range(num_epochs), desc="Epochs"):
        model.train()
        train_losses = []

        for data in train_loader:
            csi_data = data["input_wifi-csi"].float().to(device)
            keypoint = data["output"].to(device)
            xy_gt = keypoint[:, :, 0:2]
            confidence = keypoint[:, :, 2:3]

            pred_xy, _ = model(csi_data)
            pred_xy = pred_xy.squeeze()

            loss = criterion(
                torch.mul(confidence, pred_xy),
                torch.mul(confidence, xy_gt)
            ) / config["train_loader"]["batch_size"]

            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()
            train_losses.append(loss.item())

        scheduler.step()
        mean_train_loss = np.mean(train_losses)
        train_loss_log.append(mean_train_loss)

        # Validation mỗi epoch
        mpjpe, pa_mpjpe, pck50 = evaluate(model, val_loader, criterion, device, split_name=f"Epoch {epoch+1:03d}/Val")
        val_mpjpe_log.append(mpjpe)

        # Lưu best model
        if mpjpe < best_mpjpe:
            best_mpjpe = mpjpe
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, "best.pt"))
            print(f"    ✅ Saved best model — MPJPE: {mpjpe:.2f}mm")

    # ------- Test với best model -------
    print(f"\n{'='*60}")
    print("Đánh giá model tốt nhất trên tập TEST...")
    print(f"{'='*60}")
    model.load_state_dict(torch.load(os.path.join(checkpoint_dir, "best.pt")))
    test_mpjpe, test_pa_mpjpe, test_pck50 = evaluate(model, test_loader, criterion, device, split_name="TEST")

    # ------- Lưu kết quả -------
    result_path = os.path.join(config.get("checkpoint", "./checkpoints/"), "results.txt")
    split_name = config.get("split_to_use", "unknown")
    with open(result_path, "a") as f:
        f.write(f"[{split_name}] MPJPE: {test_mpjpe:.2f}mm | PA-MPJPE: {test_pa_mpjpe:.2f}mm | PCK@50: {test_pck50:.4f}\n")
    print(f"\n==> Kết quả đã lưu vào: {result_path}")

    # ------- Lưu log loss -------
    np.save(os.path.join(checkpoint_dir, "train_loss_log.npy"), np.array(train_loss_log))
    np.save(os.path.join(checkpoint_dir, "val_mpjpe_log.npy"), np.array(val_mpjpe_log))
    print("==> Đã lưu training log (train_loss_log.npy, val_mpjpe_log.npy)")


if __name__ == "__main__":
    main()
