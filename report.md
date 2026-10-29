# 基于 Wide ResNet 与多模型集成的 CIFAR-10 图像分类

## 摘要

本实验从零训练卷积神经网络完成 CIFAR-10 十分类任务。主干网络采用适合 32×32
图像的 Wide ResNet-28-10（WRN-28-10），并配合随机裁剪、水平翻转、Cutout、
RandAugment、CutMix/MixUp、标签平滑和权重指数滑动平均（EMA）训练。最后将 5 个
独立训练模型的 softmax 概率做等权平均，并使用水平翻转测试时增强（TTA）。在给定的
10,000 张测试图像上，集成模型取得了 **97.46%** 的 Top-1 识别率。

## 1. 数据与实验设置

数据集位于 `data/CIFAR10/`，按类别文件夹存放，共 10 类：airplane、automobile、
bird、cat、deer、dog、frog、horse、ship、truck。压缩包中训练集实际包含
**49,995** 张图像（5 张缺失），测试集为 **10,000** 张，每类 1,000 张。所有图像均为
32×32 RGB。测试集不参与梯度更新，也不用于集成成员筛选或超参数选择；所有成员按
预定配方独立训练后统一评估，训练集与测试集的类别映射由文件夹名称自动生成并校验
一致。

实验环境为 Python 3.10.22、PyTorch 2.5.1+cu121、torchvision 0.20.1+cu121、
NumPy 2.2.6，GPU 为 NVIDIA GeForce RTX 3090（24 GB），驱动版本 550.107.02。
训练使用 BF16 自动混合精度和 channels-last 内存格式。在仓库根目录可按以下命令
配置环境：

```bash
conda create -n ml-assignment3 python=3.10 pip -y
conda activate ml-assignment3
pip install -r requirements.txt
```

`requirements.txt` 指定了 PyTorch cu121 源，PyTorch 自带 CUDA 12.1 运行时，因此
无需在 conda 中单独安装 CUDA 工具包。安装完成后可用
`python -c "import torch; print(torch.cuda.is_available())"` 检查 GPU 是否可用。

## 2. 网络结构

CIFAR-10 图像分辨率较低，过深的网络容易增加优化难度，因此选择宽度较大的
Wide ResNet。WRN-28-10 由 1 个 3×3 卷积 stem、3 个 stage 和分类头组成，每个 stage
包含 4 个残差块，通道数依次为 160、320、640；stage 2 和 stage 3 的首个残差块使用
步长 2 下采样。网络总参数量为 **36,480,442**（不使用 SE 时）。

| 阶段 | 操作 | 输出尺寸 |
|---|---|---|
| stem | 3×3 conv, 16 channels | 32×32×16 |
| stage 1 | 4×BasicBlock, 160 channels, stride 1 | 32×32×160 |
| stage 2 | 4×BasicBlock, 320 channels, stride 2 | 16×16×320 |
| stage 3 | 4×BasicBlock, 640 channels, stride 2 | 8×8×640 |
| head | BN → ReLU → global average pooling → FC | 10 |

每个 BasicBlock 采用“卷积-BN-ReLU-卷积-BN-残差相加-ReLU”的结构。为改善梯度
传播，每个残差块最后一个 BN 的缩放参数初始化为 0，使残差分支初始输出为 0，网络
初始阶段以 shortcut 为主；当通道数或空间尺寸变化时，shortcut 使用 1×1 卷积进行
匹配。部分集成成员在残差
分支上加入 Squeeze-and-Excitation（SE）通道注意力：先全局平均池化，再经 reduction
为 16 的两层全连接网络得到通道权重，最后逐通道缩放特征。带 SE 的模型参数量为
36,754,002，增加不到 1%。

## 3. 训练策略与参数

### 3.1 数据增广与正则化

训练输入先做 padding=4 的反射填充随机裁剪到 32×32，再以 0.5 概率水平翻转，并按
通道使用 CIFAR-10 均值 (0.4914, 0.4822, 0.4465) 和标准差
(0.2470, 0.2435, 0.2616) 归一化。不同集成成员使用以下增广组合：

- **Cutout**：以 0.5 概率将 8×8 的方形区域置零；
- **RandAugment**：2 个操作、magnitude 为 9；
- **CutMix**：以 0.5 概率按 Beta(1,1) 采样裁剪框并混合图像与标签；
- **MixUp**：以 0.5 概率按 Beta(1,1) 混合两幅图像及标签；
- **标签平滑**：平滑系数 0.1。

混合样本的损失为软标签交叉熵：

$$
L=-\frac{1}{N}\sum_i\sum_c t_{i,c}\log p_{i,c}.
$$

### 3.2 优化参数

| 参数 | 设置 |
|---|---|
| 优化器 | SGD + Nesterov momentum |
| 初始学习率 | 0.1 |
| 动量 / 权重衰减 | 0.9 / 5×10⁻⁴ |
| 学习率调度 | 5 epoch 线性 warmup + cosine annealing |
| 批大小 | 128 |
| 训练轮数 | 200 |
| 精度 | BF16 自动混合精度 |
| EMA 衰减 | 0.9998 |
| 随机种子 | 每个成员不同（101–505） |

权重衰减不作用于 BN 参数和偏置。每个 epoch 遍历 390 个 batch（丢弃不足一个 batch
的 75 张图像）。训练结束后使用 EMA 参数，而不是最后一个 batch 的瞬时参数进行测试。
测试时对原图和水平翻转图分别计算 softmax，并取平均概率作为 TTA 结果。

### 3.3 集成

5 个成员使用相同训练预算，但在网络结构（是否使用 SE）和增广策略上有所区别，以
降低预测相关性。最终对 5 个模型的 TTA softmax 概率做等权平均：

$$
p(y\mid x)=\frac{1}{5}\sum_{m=1}^{5}p_m(y\mid x),
$$

取最大概率类别作为预测。

## 4. 测试结果

5 个成员在 10,000 张测试图像上的结果如下。Raw 表示训练结束时的普通权重，EMA 表示
参数滑动平均权重，二者均使用水平翻转 TTA。

| 成员 | SE | 混合增广 | RandAugment | Raw (%) | EMA (%) |
|---|---|---|---|---|---|
| A | 否 | CutMix | 否 | 96.80 | 96.91 |
| B | 是 | CutMix | 否 | 96.88 | 96.94 |
| C | 是 | MixUp | 否 | 96.40 | 96.57 |
| D | 是 | 无 | 是 | 96.25 | 96.27 |
| E | 否 | CutMix | 是 | 96.88 | **97.00** |
| **等权集成** | — | — | — | — | **97.46** |

单模型 EMA 准确率在 96.27%–97.00% 之间，平均为 96.74%；等权集成达到 **97.46%**
（10,000 张中正确 9,746 张），比最好的单模型高 0.46 个百分点，说明不同结构和增广
训练的成员具有互补性。5 个模型在 5 张 RTX 3090 上并行训练，单模型耗时约
1.9–2.2 小时，总墙钟时间约 2.2 小时。

集成模型的逐类准确率如下：

| 类别 | airplane | automobile | bird | cat | deer | dog | frog | horse | ship | truck |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 准确率 (%) | 98.3 | 99.2 | 96.4 | 93.4 | 97.9 | 95.9 | 98.4 | 98.5 | 98.6 | 98.0 |

汽车（99.2%）、船（98.6%）、马（98.5%）和青蛙（98.4%）最容易识别；猫的准确率
最低（93.4%），主要被误判为狗（45 张），反向的狗→猫错误为 26 张。其余主要混淆为
truck→automobile（14 张）、airplane→ship（10 张）和 bird→deer（9 张），均符合
类别间外观相似的直觉。总体测试误差为 2.54%。

## 5. 复现方式

```bash
conda activate ml-assignment3
bash scripts/run_experiments.sh
```

详细的复现方式见[`README.md`](README.md)
脚本并行启动 5 个训练任务，结束后自动调用 `src/ensemble.py` 生成集成结果。单模型
训练命令和输出文件说明见 `README.md`。所有测试结果由
`outputs/ensemble/ensemble_metrics.json` 记录，每个成员的完整日志、`metrics.json`
和测试概率位于 `outputs/runs/`；为控制提交体积，只保留单模型最优成员 E 的
checkpoint（`outputs/runs/E_wrn28_10_ra_cutmix/checkpoint_last.pt`），可用
`python src/predict.py` 快速复算其测试准确率。
## 6. 结论

实验表明，WRN-28-10 的宽度换来了较强的 CIFAR-10 表征能力；CutMix/MixUp、
Cutout、标签平滑和 RandAugment 缓解了过拟合，EMA 与水平翻转 TTA 进一步稳定了
预测。多个模型概率平均后，集成在测试集上达到 **97.46%** 的 Top-1
准确率。主要误差仍集中在猫/狗、卡车/汽车、飞机/船等视觉相似的类别上。
