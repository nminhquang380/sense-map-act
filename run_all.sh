#!/usr/bin/env bash
# Chạy toàn bộ benchmark trên máy 4x RTX 3090. Tổng thời gian ~1-2 giờ (phần lớn là CPU baseline).
# Mỗi bước ghi một file JSON trong results/. Một bước lỗi KHÔNG làm dừng các bước sau; hình luôn được vẽ ở cuối.
# Chạy lại một số bước:   STEPS="3 4 5 6" bash run_all.sh
set -o pipefail
cd "$(dirname "$0")"
mkdir -p results figs logs
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
STEPS=${STEPS:-"0 1 2 3 4 5 6"}
FAILED=""
want() { [[ " $STEPS " == *" $1 "* ]]; }
fail() { echo "!! Bước $1 lỗi (xem logs/). Tiếp tục các bước sau."; FAILED="$FAILED $1"; }

if want 0; then
  echo "== 0. Kiểm tra môi trường"
  nvidia-smi | tee logs/nvidia-smi.txt
  nvidia-smi topo -m > logs/gpu_topology.txt 2>&1
  lscpu > logs/cpu.txt; free -g > logs/memory.txt
  python check_env.py | tee logs/check_env.txt || { echo "Môi trường chưa sẵn sàng: xem các dòng FAIL ở trên"; exit 1; }
fi

if want 1; then
  echo "== 1. MARL env (ICCC'26): kiểm tra tương đương + throughput + quét N  (~10-20 phút)"
  CUDA_VISIBLE_DEVICES=0 python bench_marl_env.py --out results/marl.json | tee logs/marl.txt || fail 1
fi

if want 2; then
  echo "== 2. NSADM (WCL'26): CPU vs GPU FP32/TF32/BF16/FP16  (~10-20 phút)"
  # Sửa --dataset/--epochs/--batch cho đúng cấu hình bạn đã train. Dùng --plugin để đo model thật.
  CUDA_VISIBLE_DEVICES=0 python bench_model.py --name nsadm --batch 16 --infer-batch 16 \
      --dataset 800 --epochs 200 --compile --out results/nsadm.json | tee logs/nsadm.txt || fail 2
fi

if want 3; then
  echo "== 3. 3D-DRM (ICC'26): CPU vs GPU  (~20-40 phút, CPU chậm)"
  # 7.8 M voxel/frame, 7,500 token/frame; CPU và GPU cùng batch 4.
  CUDA_VISIBLE_DEVICES=0 python bench_model.py --name drm --batch 4 --infer-batch 4 \
      --dataset 18000 --epochs 100 --compile --out results/drm.json | tee logs/drm.txt || fail 3
fi

if want 4; then
  echo "== 4. 3D-DRM multi-GPU scaling (DDP/NCCL)  (~5 phút)"
  rm -f results/ddp_drm.json
  for k in 1 2 4; do
    torchrun --standalone --nproc_per_node=$k bench_ddp.py --name drm --batch 4 --out results/ddp_drm.json \
      2>&1 | tee -a logs/ddp.txt || fail "4 (${k} GPU)"
  done
fi

if want 5; then
  echo "== 5. (Tuỳ chọn) Sionna RT 3D radio map, CPU vs GPU  (~10-20 phút)"
  python bench_sionna_rt.py --out results/sionna.json | tee logs/sionna.txt || fail 5
fi

if want 6; then
  echo "== 6. Vẽ hình cho poster"
  python make_figs.py --results results --out figs --budget-ms 100 | tee logs/figs.txt || fail 6
fi

if [ -n "$FAILED" ]; then echo "XONG, nhưng các bước lỗi:$FAILED. Gửi lại results/, figs/ và logs/."
else echo "XONG. Gửi lại results/, figs/ và logs/ (hoặc zip) để làm poster."; fi
