#!/usr/bin/env bash
# Chạy toàn bộ benchmark trên máy 4x RTX 3090. Tổng thời gian ~1-2 giờ (phần lớn là CPU baseline).
# Có thể chạy từng khối riêng; mỗi khối ghi một file JSON trong results/.
set -e
set -o pipefail
cd "$(dirname "$0")"
mkdir -p results figs logs
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}

echo "== 0. Kiểm tra môi trường"; nvidia-smi | tee logs/nvidia-smi.txt
python check_env.py | tee logs/check_env.txt || { echo "Môi trường chưa sẵn sàng: xem các dòng FAIL ở trên"; exit 1; }

echo "== 1. MARL env (ICCC'26): kiểm tra tương đương + throughput + quét N  (~10-20 phút)"
CUDA_VISIBLE_DEVICES=0 python bench_marl_env.py --out results/marl.json | tee logs/marl.txt

echo "== 2. NSADM (WCL'26): CPU vs GPU FP32/TF32/BF16/FP16  (~10-20 phút)"
# Sửa --dataset/--epochs/--batch cho đúng cấu hình bạn đã train. Dùng --plugin để đo model thật.
CUDA_VISIBLE_DEVICES=0 python bench_model.py --name nsadm --batch 16 --infer-batch 16 \
    --dataset 800 --epochs 200 --compile --out results/nsadm.json | tee logs/nsadm.txt

echo "== 3. 3D-DRM (ICC'26): CPU vs GPU  (~20-40 phút, CPU chậm)"
CUDA_VISIBLE_DEVICES=0 python bench_model.py --name drm --batch 4 --infer-batch 4 \
    --dataset 18000 --epochs 100 --compile --out results/drm.json | tee logs/drm.txt

echo "== 4. 3D-DRM multi-GPU scaling (DDP/NCCL)  (~5 phút)"
rm -f results/ddp_drm.json
for k in 1 2 4; do
  torchrun --standalone --nproc_per_node=$k bench_ddp.py --name drm --batch 4 --out results/ddp_drm.json
done

echo "== 5. (Tuỳ chọn) Sionna RT 3D radio map, CPU vs GPU  (~10-20 phút)"
python bench_sionna_rt.py --out results/sionna.json | tee logs/sionna.txt || echo "Sionna RT bị bỏ qua"

echo "== 6. Vẽ hình cho poster"
python make_figs.py --results results --out figs --budget-ms 100
echo "XONG. Gửi lại thư mục results/ và figs/ (hoặc zip) để làm poster."
