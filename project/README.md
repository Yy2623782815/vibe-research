# ImageNet Validation Subset Benchmark

一个可直接运行的 Python 项目，用于评估多个图像分类模型在 ImageNet validation subset 上的性能，并通过 MLflow 记录实验。

## 对比模型

- `resnet50`
- `mobilenet_v3_large`
- `convnext_tiny`
- `swin_t`

## 项目结构

```text
project/
  data/
  scripts/
  models/
  results/
  analysis/
  main.py
  requirements.txt
  README.md
```

## 环境安装

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r project/requirements.txt
```

## 数据格式要求

默认使用 `torchvision.datasets.ImageFolder` 读取验证集目录，即：

```text
imagenet_val/
  n01440764/
    *.JPEG
  n01443537/
    *.JPEG
  ...
```

## 运行说明

### 1) Smoke Test（100 张）

```bash
python project/main.py \
  --imagenet-val-dir /path/to/imagenet_val \
  --mode smoke \
  --smoke-size 100 \
  --batch-size 32 \
  --device cuda
```

### 2) Full Run（1000+）

```bash
python project/main.py \
  --imagenet-val-dir /path/to/imagenet_val \
  --mode full \
  --full-size 1200 \
  --batch-size 64 \
  --device cuda
```

## 产出文件

- `project/data/subset.csv`：分层采样子集（每类随机 20 张）
- `project/data/run_subset.csv`：本次运行使用的样本（smoke/full）
- `project/results/predictions_<model>.csv`：逐图预测结果
- `project/results/summary.csv`：模型对比汇总（Top-1/Top-5/延迟/参数量）
- `project/results/per_class_accuracy.csv`：每模型、每类别准确率
- `project/analysis/accuracy_bar.png`：准确率柱状图
- `project/analysis/accuracy_latency_scatter.png`：准确率-延迟散点图
- `project/analysis/per_class_summary.csv`：每类最佳模型汇总

## MLflow

运行后会在本地创建 MLflow 实验 `imagenet-model-benchmark`，并记录：

- 参数：模式、样本量、batch size、设备、模型列表
- 指标：top-1、top-5、平均推理时延、参数量
- Artifacts：CSV 结果与分析图表

可通过以下命令查看：

```bash
mlflow ui
```

默认地址：`http://127.0.0.1:5000`

## 脚本拆分

- `project/scripts/data_prep.py`：分层采样与 run 子集生成
- `project/scripts/evaluate.py`：模型评测与指标统计
- `project/scripts/analyze.py`：可视化与 per-class 汇总

