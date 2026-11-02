# Bộ benchmark GPU cho poster GTC 2027

Mục tiêu: đo **cái GPU làm được** cho ba công trình đã công bố. Kết quả accuracy đã có sẵn trong paper, không cần chạy lại.

## Cài đặt trên server mới (không cần sudo)

**Bước 0: xem driver.** Chạy `nvidia-smi` và ghi lại `CUDA Version` ở góc trên bên phải. Đó là phiên bản CUDA cao nhất mà driver hỗ trợ.

**Bước 1: Miniforge (conda, chỉ cài một lần).**
```bash
cd ~
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
bash Miniforge3-Linux-x86_64.sh -b -p $HOME/miniforge3
$HOME/miniforge3/bin/conda init bash && source ~/.bashrc
```

**Bước 2: môi trường riêng.**
```bash
conda create -n smact python=3.12 -y
conda activate smact
```

**Bước 3: PyTorch đúng với driver.** Chọn **một** dòng theo `CUDA Version` ở bước 0:
```bash
pip install torch                                                     # CUDA Version >= 13.0
pip install torch --index-url https://download.pytorch.org/whl/cu128  # 12.8 – 12.9
pip install torch --index-url https://download.pytorch.org/whl/cu126  # 12.6 – 12.7
```
Driver cũ hơn 12.6: nhờ quản trị server cập nhật driver.

**Bước 4: thư viện còn lại, LLVM cho Sionna RT trên CPU, và code.**
```bash
pip install numpy matplotlib sionna-rt
conda install -c conda-forge libllvm18 -y
conda env config vars set DRJIT_LIBLLVM_PATH=$(ls $CONDA_PREFIX/lib/libLLVM*.so* | head -1)
conda deactivate && conda activate smact          # nạp biến môi trường vừa đặt
git clone https://github.com/nminhquang380/sense-map-act && cd sense-map-act
python check_env.py
```
`check_env.py` in PASS / WARN / FAIL cho từng mục. Chỉ cần sửa các dòng **FAIL**. Dòng WARN của Sionna RT không chặn gì: benchmark Sionna là tuỳ chọn. Nếu thiếu `gcc` (chế độ `torch.compile` sẽ bị bỏ qua): `conda install -c conda-forge gxx -y`.

**Bước 5: chạy thử nhanh (khoảng 1 phút), rồi chạy toàn bộ trong tmux.**
```bash
python bench_marl_env.py --check
python bench_model.py --name nsadm --small --batch 4 --infer-batch 4 --out results/test.json

tmux new -s bench                        # mất SSH thì job vẫn chạy
conda activate smact && bash run_all.sh 2>&1 | tee logs_run_all.txt
# Ctrl-b rồi d để thoát tmux; quay lại bằng: tmux attach -t bench
```
Server dùng chung: xem `nvidia-smi` để chọn GPU rảnh, ví dụ `export CUDA_VISIBLE_DEVICES=0,1,2,3` trước khi chạy.

**Bước 6: gửi kết quả.**
```bash
zip -r gtc_results.zip results figs logs logs_run_all.txt
```

## Từng khối đo gì

| Script | Công trình | Đo gì | Hình cho poster |
|---|---|---|---|
| `bench_marl_env.py` | CTDE-MAPPO (ICCC'26) | Env vectorised trên GPU vs env Gym kiểu cũ; quét số swarm song song E và số UAV N; thời gian dự kiến cho 2M bước | `marl_throughput`, `marl_projection`, `marl_nsweep` |
| `bench_model.py --name nsadm` | NSADM (WCL'26) | Train step + toàn bộ T bước khử nhiễu: CPU, GPU FP32, TF32, BF16/FP16 AMP, torch.compile | `model_nsadm` |
| `bench_model.py --name drm` | 3D-DRM (ICC'26) | Như trên; latency dự báo 5 frame so với ngưỡng thời gian thực | `model_drm` |
| `bench_ddp.py` | 3D-DRM | Mở rộng 1/2/4 GPU (DDP + NCCL), hiệu suất so với tuyến tính | `ddp_drm` |
| `bench_sionna_rt.py` | Mở rộng 3D-DRM | Ray tracing bản đồ gain 3D (cảnh Munich) CPU vs GPU | `sionna_time`, `sionna_map` |

## Ba điều cần làm đúng để số liệu đứng vững

1. **Kiểm tra tương đương MARL.** `--check` cho thấy env vectorised và env gốc trả về cùng reward (sai số ~1e-14). Vì đây là cùng một MDP, đường học theo số bước môi trường giữ nguyên; chỉ thời gian mỗi bước thay đổi. Đó là cơ sở để **dự báo** thời gian huấn luyện từ throughput đo được, thay vì train lại toàn bộ.
2. **Model thật hay proxy.** Mặc định là proxy cùng kích cỡ, chỉ để kiểm tra luồng. Cho poster, nên gắn model thật qua `--plugin` (xem docstring đầu `bench_model.py`; chỉ cần 4 hàm). Nếu chỉ dùng proxy, ghi rõ trên poster là "size-matched proxy".
3. **Cấu hình.** Sửa `--dataset`, `--epochs`, `--batch`; với NSADM sửa W, H, T trong `build_nsadm`; với 3D-DRM sửa patch P trong `build_drm` cho đúng code bạn đã chạy. Hằng số vật lý đánh dấu `PLACEHOLDER` trong env MARL không ảnh hưởng tốc độ.

## Phương pháp đo (ghi vào ô "Benchmarking methodology" trên poster)

- Bỏ các vòng warm-up; gọi `torch.cuda.synchronize()` trước khi đọc đồng hồ; báo cáo **median** của 20 lần (CPU: 3 lần).
- CPU baseline dùng toàn bộ lõi (`torch.set_num_threads`); cùng batch, cùng trọng số, cùng input cho mọi chế độ.
- Độ chính xác: `rel_err_vs_cpu_fp32` trong JSON cho thấy AMP không làm đổi kết quả đáng kể.
- Thời gian huấn luyện dự kiến = epochs × ⌈dataset/batch⌉ × thời gian một step đo được.
- Phần cứng và phiên bản phần mềm được ghi tự động vào khối `"hardware"` của mỗi JSON.
