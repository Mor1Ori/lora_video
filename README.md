# lora_video

基于 Stable Diffusion v1.5 的小样本图像 LoRA 风格验证项目。  
当前目标是先完成本地可复现实验闭环，再迁移到云平台高性能显卡进行更长训练和后续视频阶段扩展。

## 1. 项目功能

- 数据预处理：扫描、清洗、裁剪缩放、训练/验证划分、导出元数据
- 训练验证：图像 LoRA 训练（支持小样本和 rank 消融）
- 结果评估：单 prompt 推理与 baseline/LoRA 对比图生成
- 过程记录：实验协议与周报记录

## 2. 目录结构

```text
.
|- scripts/
|  |- preprocess.py
|  |- train_smoke_test.py
|  |- train_lora_image.py
|  |- infer_lora_image.py
|  `- generate_prompt_comparison.py
|- data/
|  |- raw/
|  `- processed/
|- artifacts/
|- reports/
`- docs/
```

## 3. 环境准备（本地/云通用）

建议 Python 3.10+，训练建议使用 CUDA GPU。

1. 创建环境

```bash
conda create -n lora_video python=3.10 -y
conda activate lora_video
```

2. 安装依赖

```bash
# torch/torchvision 请根据你的 CUDA 版本使用官方安装命令
pip install torch torchvision
pip install diffusers transformers peft accelerate safetensors pillow tqdm numpy
```

## 4. 数据格式

将原始图片放入 `data/raw/`。  
支持可选的同名侧边标注文本（`.txt` 后缀）。

示例：

```text
data/raw/
|- img_0001.jpg
|- img_0001.txt
|- img_0002.png
`- ...
```

## 5. 运行流程（本地/云使用同一套命令）

以下命令均在项目根目录执行。

### 5.1 数据预处理

```bash
python scripts/preprocess.py --input-dir data/raw --output-dir data/processed --size 512 --val-ratio 0.2
```

输出：

- `data/processed/train`, `data/processed/val`
- `data/processed/metadata_train.jsonl`
- `data/processed/metadata_val.jsonl`
- `data/processed/stats.json`

### 5.2 环境冒烟测试

```bash
python scripts/train_smoke_test.py --steps 50 --batch-size 8 --out-dir artifacts/smoke
```

### 5.3 图像 LoRA 训练

```bash
python scripts/train_lora_image.py --train-data-dir data/processed --metadata-file data/processed/metadata_train.jsonl --output-dir artifacts/lora_image_ghibli_r8_s200 --resolution 512 --train-batch-size 1 --num-train-epochs 2 --max-train-steps 200 --save-steps 100 --learning-rate 1e-4 --rank 8 --mixed-precision fp16
```

### 5.4 推理生成

```bash
python scripts/infer_lora_image.py --lora-path artifacts/lora_image_ghibli_r8_s200 --output-dir artifacts/infer_lora_ghibli --prompt "a cozy countryside street in hand-drawn animation style, warm light, cinematic composition" --num-images 4 --steps 30
```

### 5.5 baseline vs LoRA 对比

```bash
python scripts/generate_prompt_comparison.py --lora-path artifacts/lora_image_ghibli_r8_s200 --output-dir artifacts/compare_lora_r8_s200 --steps 25 --guidance-scale 7.5 --seed 42
```

## 6. 消融实验建议

固定其他参数，仅修改 `--rank`：

- `rank=4`
- `rank=8`
- `rank=16`

建议记录：

- `train_state.json` 里的 `best_loss`
- 对比图目录（`baseline/` 与 `lora/`）
- 风格强度与细节稳定性主观观察

## 7. 云平台迁移说明

流程保持不变，只需要做三件事：

1. 上传代码和数据，保持相同目录结构
2. 重建环境并安装依赖
3. 在云端项目根目录执行同样命令

Linux 云主机示例：

```bash
cd /workspace/lora_video
conda activate lora_video
python scripts/train_lora_image.py --train-data-dir data/processed --metadata-file data/processed/metadata_train.jsonl --output-dir artifacts/lora_image_cloud_r8_s400 --max-train-steps 400 --rank 8 --mixed-precision fp16
```

可选：若模型缓存已存在，可离线运行

```bash
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
```

## 8. 常见问题

- 首次运行较慢：需要下载模型，后续会命中缓存
- 离线 LoRA 加载报错：请使用当前脚本版本（已显式处理 `weight_name`）
- 显存不足：优先降低 `--train-batch-size`，其次减少 `--resolution` 或 `--max-train-steps`

## 9. 当前阶段建议

先完成本地图像阶段消融与结论，再切到云平台做长步数训练和视频阶段实验。

## 10. 视频阶段入口

视频阶段的数据与评测设计文档：

- `docs/video_stage_protocol.md`

快速开始（示例）：

```bash
python scripts/preprocess_video.py --input-dir data/video_raw --output-dir data/video_processed --size 512 --val-ratio 0.2 --min-frames 16 --max-frames 32 --frame-stride 1 --caption-file clip.txt
python scripts/generate_video_style.py --output-dir artifacts/video_gen_baseline --clip-id clip_0001 --prompt "a cozy riverside town at sunset, hand-drawn animation style, gentle camera movement" --num-frames 12 --steps 20 --guidance-scale 7.5 --strength 0.30 --seed 42 --fps 8
python scripts/generate_video_style.py --output-dir artifacts/video_gen_lora_r8_s400 --clip-id clip_0001 --prompt "a cozy riverside town at sunset, hand-drawn animation style, gentle camera movement" --lora-path artifacts/lora_image_ghibli_r8_s400 --num-frames 12 --steps 20 --guidance-scale 7.5 --strength 0.30 --seed 42 --fps 8
python scripts/eval_video_consistency.py --clips-dir artifacts/video_gen_baseline --output artifacts/video_eval/consistency_baseline_generated.json --min-frames 8
python scripts/eval_video_consistency.py --clips-dir artifacts/video_gen_lora_r8_s400 --output artifacts/video_eval/consistency_lora_generated.json --min-frames 8
python scripts/generate_video_style_batch.py --baseline-output-dir artifacts/video_gen_baseline_batch --lora-output-dir artifacts/video_gen_lora_r8_s400_batch --lora-path artifacts/lora_image_ghibli_r8_s400 --num-frames 12 --steps 20 --guidance-scale 7.5 --strength 0.30 --seed 42 --fps 8
python scripts/eval_video_consistency.py --clips-dir artifacts/video_gen_baseline_batch --output artifacts/video_eval/consistency_baseline_batch.json --min-frames 8
python scripts/eval_video_consistency.py --clips-dir artifacts/video_gen_lora_r8_s400_batch --output artifacts/video_eval/consistency_lora_batch.json --min-frames 8
```
