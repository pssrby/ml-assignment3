# CIFAR-10 卷积神经网络分类

本目录是机器学习第三次作业的完整实现：从零训练 Wide ResNet（WRN-28-10）完成
CIFAR-10 十类图像分类，并用多个独立训练模型的概率平均集成提高测试集准确率。
不使用预训练权重或额外数据。最终 5 模型集成在 10,000 张测试图像上取得
**97.46%** 的 Top-1 准确率。

## 环境

本项目在 Python 3.10、PyTorch 2.5.1（CUDA 12.1）、torchvision 0.20.1 下测试，
GPU 为 NVIDIA RTX 3090，训练使用 BF16 自动混合精度。

首次运行时，在仓库根目录用 conda 创建环境并安装依赖：

```bash
conda create -n ml-assignment3 python=3.10 pip -y
conda activate ml-assignment3
pip install -r requirements.txt
```

`requirements.txt` 已指定 PyTorch cu121 软件源，PyTorch 会自带所需的 CUDA 12.1
运行时；宿主机只需安装可支持 CUDA 12.1 的 NVIDIA 驱动。安装后可检查 GPU 是否可用：

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

预期输出包含 `2.5.1+cu121` 和 `True`。环境已配置好时，只需激活：

```bash
conda activate ml-assignment3
```

## 数据

数据位于 `data/CIFAR10/`，按类别文件夹组织：

```
data/CIFAR10/
├── train/{airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck}/
└── test/{airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck}/
```

压缩包中的训练集实际为 **49,995** 张（缺少 5 张），测试集为 **10,000** 张。
类别顺序与标准 CIFAR-10 一致，程序直接使用文件夹名称作为标签。

## 训练

单个模型：

```bash
CUDA_VISIBLE_DEVICES=0 python src/train.py \
  --data-dir data/CIFAR10 --output-dir outputs/runs/my_run \
  --epochs 200 --batch-size 128 --lr 0.1 --weight-decay 5e-4 \
  --mix cutmix --label-smoothing 0.1 --cutout 8 --ema-decay 0.9998
```

训练 5 个集成成员并自动评估集成（需要 5 张 GPU，约 2 小时）：

```bash
conda activate ml-assignment3
bash scripts/run_experiments.sh
```

每个运行目录包含：

- `checkpoint_last.pt`：模型、EMA、优化器和最后一个 epoch；
- `history.csv` / `train.log`：逐 epoch 损失、训练准确率和耗时；
- `metrics.json`：参数量、测试准确率、逐类准确率等；
- `test_probs.npy` / `test_labels.npy`：测试集概率和标签，用于集成。

## 测试集集成

```bash
python src/ensemble.py --run-dirs \
  outputs/runs/A_wrn28_10_cutmix \
  outputs/runs/B_wrn28_10_se_cutmix \
  outputs/runs/C_wrn28_10_se_mixup \
  outputs/runs/D_wrn28_10_se_ra \
  outputs/runs/E_wrn28_10_ra_cutmix \
  --output-dir outputs/ensemble
```

集成方式为各模型（水平翻转 TTA 后）的 softmax 概率等权平均。

## 文件

- `src/models.py`：WRN-28-10 与可选 SE 模块；
- `src/train.py`：数据增广、训练、EMA、TTA 测试；
- `src/ensemble.py`：多模型概率平均与混淆矩阵统计；
- `scripts/run_experiments.sh`：复现实验用的 5 模型训练脚本；
- `report.md`：实验报告（网络结构、参数、结果分析）。

最终测试结果和详细分析见 [`report.md`](report.md)。
