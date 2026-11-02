# Bộ benchmark GPU cho poster GTC 2027

Mục tiêu: đo **cái GPU làm được** cho ba công trình đã công bố. Kết quả accuracy đã có sẵn trong paper, không cần chạy lại.

## Cài đặt (máy 4×RTX 3090)

```bash
pip install torch numpy matplotlib          # nếu chưa có
pip install sionna-rt                       # tuỳ chọn, cho khối 5
bash run_all.sh                             # hoặc chạy từng khối bên dưới
```

Chạy thử nhanh trước (khoảng 1 phút) để chắc không lỗi:

```bash
python bench_marl_env.py --check
python bench_model.py --name nsadm --small --batch 4 --infer-batch 4 --out results/test.json
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
