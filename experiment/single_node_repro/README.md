# 单机 8 卡复现：RoboTwin 2.0 clean 后训练（Distributed Muon / Muon；H20 与 RTX PRO 5000）

用官方代码和官方 `configs/vla/robotwin/robotwin_dist_muon.yaml`，在单台 8 卡机器（8×H20、8×RTX PRO 5000）上复现 RoboTwin 2.0 clean 数据的后训练速度和 loss。
训练代码没有任何修改：本目录的脚本只负责下载权重、准备数据、填写配置里的路径，以及统计耗时。

## 结果

### 8×H20（micro 16 × 8 卡 × 梯度累积 2，不重算激活）

| | Distributed Muon | Muon |
|---|---:|---:|
| 预热后 step 耗时（step 31–130 平均） | **4.620 s** | **6.288 s** |
| P50 / P95 | 4.615 / 4.644 s | 6.283 / 6.315 s |
| 墙钟 / step（含取数据） | 4.630 s | 6.300 s |
| 整卡显存峰值（nvidia-smi，step 31–130） | 83.82 GiB | 82.85 GiB |
| 5,000 步实际用时（进程启动到结束，含编译和 2 次存 checkpoint） | 6.55 h | 8.85 h |
| loss（日志单步值，step 1 → 5000） | 0.378 → 0.0215 | 0.378 → 0.0216 |

以上数字都可以用 `summarize_run.py` 从 `train.log` 和 `smi.csv` 重新算出来，只有"5,000 步实际用时"不行，它按进程启动到结束计算。

- Distributed Muon 比 Muon 快 1.36×；官方 README 给的是 4.31 / 5.93 s，即 1.38×。比值一致，绝对值慢一些，原因见文末的"与官方的差异"。
- 测试环境：8×H20 96GB，NVLink 全互联；驱动 560.35.03；PyTorch 2.8.0+cu126，NCCL 2.27.3，Transformers 4.57.3，flash-attn 2.8.3，Triton 3.4.0。

### 8×RTX PRO 5000 72GB（micro 32 × 8 卡 × 梯度累积 1，开激活重算）

| | Distributed Muon | Muon |
|---|---:|---:|
| 预热后 step 耗时（平均） | **4.767 s**（step 31–130） | **6.900 s**（step 31–120，降频前；见下方说明） |
| P50 / P95 | 4.767 / 4.794 s | 6.902 / 6.932 s |
| 墙钟 / step（含取数据） | 4.780 s | 6.911 s |
| PyTorch 张量分配峰值 / 缓存池峰值 | 53.909 / 69.154 GiB | 53.352 / 63.834 GiB（全程 67.924） |
| 整卡显存峰值（nvidia-smi） | 69.98 GiB | 64.62 GiB（全程 68.71） |
| 5k 步粗估（平均 step × 5000，不含降频） | 6.62 h | 9.58 h |

- 每组都跑了 2 次。DistMuon 两次分别是 4.758 s 和 4.767 s；Muon 降频前的稳态两次分别是 6.909 s 和 6.900 s。表中取的都是第 2 次，那一次带了温度记录。
- Distributed Muon 比 Muon 快 **1.45×**。已有的 PRO 5000 参考数字是 5.350 / 7.044 s，即 1.32×，两边条件的差异见第 3 节。
- 张量分配峰值 53.9 GiB 和缓存池峰值 69.154 GiB，与参考测量（53.872 / 69.154 GiB）几乎相同，说明显存行为一致。
- **降频说明**：长时间满载时，最热的卡会到温度上限并降频。这是预期行为，不是单卡故障：
  - RTX PRO 5000 是 300 W 涡轮散热的工作站卡。`nvidia-smi -q -d TEMPERATURE` 显示温度上限约 92°C，到达后开始软件降频。
  - 8 卡装在一台机器里时，风道最差位置的卡最先到温度上限，测试机上是 GPU 6。
  - 跑 Muon 时，GPU 6 升到 92–93°C 后开始降频（第 1 次约 step 109 起，第 2 次约 step 126 起）。只要有一张卡降频，所有卡的 step 耗时都会被拖到 12–23 s。
  - 因此 Muon 只统计降频前的 step 31–120。DistMuon 每步功耗较低，130 步内最高 88°C，没有降频。
  - `summarize_run.py` 会根据 `smi.csv` 自动标出发生过热降频的 GPU。
- **还需要测长时间运行的效率**：上表的 5k 步粗估只基于 130 步，而且是不降频时的速度。长时间训练时温度会持续处于上限附近，Muon 和 DistMuon 都可能反复降频。实际用时需要另跑 1,000 步以上，统计包含降频在内的平均 step 耗时。
- 测试环境：驱动 580.95.05；PyTorch 2.8.0+cu128，NCCL 2.27.3，Transformers 4.57.3，flash-attn 2.8.3，Triton 3.4.0。8 张卡之间只走 PCIe，没有 NVLink，GPU 0–3 和 GPU 4–7 分别挂在两个 CPU 下。数据与 H20 上的逐字节相同。

### 图

`figures/` 里共 8 张图：

| 图 | 内容 |
|---|---|
| `h20_distmuon_repro.png` / `h20_muon_repro.png` | 本复现：8×H20，DistMuon / Muon，5k 步 |
| `pro5000_distmuon_repro.png` / `pro5000_muon_repro.png` | 本复现：8×RTX PRO 5000，DistMuon / Muon，130 步速度测试 |
| `h20_distmuon.png` | 已有的 H20 DistMuon 测量（参考） |
| `pro5000_distmuon.png` / `pro5000_muon.png` | 已有的 RTX PRO 5000 测量（参考） |
| `h20_muon_vs_distributed_muon.png` | 官方 README 的成功率与 H20 速度表 |

## 0. 代码和环境

```bash
git clone https://github.com/Robbyant/lingbot-vla-v2.git
cd lingbot-vla-v2
git checkout be969b8fd117fb70550c5d4bf4bc328211b5b1b6   # 复现用的官方提交
bash tools/create_train_env.sh                          # 官方环境脚本，创建 conda 环境 lingbotvla
```

把本目录（`experiment/single_node_repro/`）放进上面的仓库，或者直接 clone 本 fork 的 `single-node-repro` 分支，二者等价。

## 1. 下载权重（约 36 GB）

```bash
MODEL_ROOT=/data/lingbot bash experiment/single_node_repro/download_models.sh
```

会下载以下三个仓库：

- `robbyant/lingbot-vla-v2-6b`：预训练权重，自带 LingBot-Depth 和 DINO-Video 两个 teacher。
- `Qwen/Qwen3-VL-4B-Instruct`
- `Ruicheng/moge-2-vitb-normal`

## 2. 准备数据（约 60 GB；在 128 核机器上 JOBS=16 时，转换约 17 分钟）

```bash
DATA_ROOT=/data/robotwin2 bash experiment/single_node_repro/prepare_robotwin_clean.sh
```

依赖：`git`、`unzip`、`uv`（`pip install uv`）、`hf`（`pip install -U huggingface_hub`）。

脚本依次完成下面几步，所有输入都固定了版本：

1. 从 HF 数据集 `TianxingChen/RoboTwin2.0`（revision `a967b85`）下载 50 个任务的 `demo_clean.zip`，然后解压。
2. 新建一个 Python 3.12 的 venv，安装 `lerobot[dataset]==0.6.0` 和 `h5py==3.14.0`。这个环境只用来转换数据，和训练环境分开。
3. clone RoboTwin，切到提交 `266f3aa`，其中 XPolicyLab 子模块在提交 `c37109c`。
4. 打一个公开 patch（[来源](https://github.com/ZiguanWang/Robotwin-radeon-cloud/blob/27b2f6fcb6c715e2cbb6db4431381f87c94b86ed/docker/patches/xpolicylab-lerobot-v30.patch)，脚本会校验 sha256）。它做两件事：
   - 让 XPolicyLab 的 LeRobot v3.0 转换脚本能在 LeRobot 0.6.0 下运行；
   - 改用 CPU 的 H.264 编码，CRF 18。
5. 以关节动作模式转换（state 和 action 都是 14 维），每个任务 50 条，共 2,500 条轨迹、549,787 帧，15 fps，3 个相机，320×240。
6. 生成数据列表 `$DATA_ROOT/robotwin_demo_clean_joint_v30.txt`，每行格式为 `robotwin <lerobot 数据集目录>`，共 50 行。

转换并发数用 `JOBS` 调，默认 16。脚本可以重复运行，已下载的文件不会重新下载。

归一化统计直接用仓库自带的 `assets/norm_stats/robotwin.json`（`configs/robot_configs/robotwin.yaml` 会引用它），不需要重新计算。

## 3. 训练

```bash
conda activate lingbotvla
export MODEL_ROOT=/data/lingbot DATA_LIST=/data/robotwin2/robotwin_demo_clean_joint_v30.txt

# 速度测试：130 步，约 20 分钟，跑完不存 checkpoint
MAX_STEPS=130 TIMING_ONLY=1 bash experiment/single_node_repro/train.sh speed_distmuon

# 完整复现：5,000 步，第 2,500 步和第 5,000 步各存一次 checkpoint
bash experiment/single_node_repro/train.sh distmuon_5k

# 对比：普通 Muon，只多一个参数
bash experiment/single_node_repro/train.sh muon_5k --train.optimizer muon
```

- 输出目录是 `$OUT_ROOT/<run_name>/`，`OUT_ROOT` 默认为 `./output`。目录里有：
  - `config.yaml`：实际使用的配置；
  - `train.log`；
  - `smi.csv`：每秒采样的显存、温度、SM 频率和过热降频状态；
  - `mem_rank*.csv`：每卡每秒的 PyTorch 张量分配峰值和缓存池峰值（由 `mem_probe.py` 记录，`MEM_PROBE=0` 可关闭）；
  - `checkpoints/global_step_*`：每个约 74 GB，其中 `hf_ckpt` 约 24 GB。
- 训练结束后会自动打印统计结果。也可以随时手动运行：
  ```bash
  python experiment/single_node_repro/summarize_run.py output/<run_name>
  ```
- `train.sh` 默认的 batch 设置是每卡 micro 16 × 8 卡 × 梯度累积 2 = 全局 256，不重算激活。可以用环境变量 `MICRO`、`GA`、`NPROC` 修改，全局 batch 会自动按 `MICRO×GA×NPROC` 计算。
- 其他 trainer 参数直接写在命令后面，例如：
  ```bash
  --train.use_wandb true --train.wandb_project <project>
  ```

### 8×RTX PRO 5000 72GB

全局 batch 和 H20 一样是 256，但拆法不同：每卡 micro 32 × 8 卡 × 梯度累积 1，并打开激活重算（`enable_gradient_checkpointing=true`）。RTX PRO 5000 每卡只有 72 GB，不开激活重算放不下 micro 32。`train_pro5000.sh` 只是在 `train.sh` 上加了这几个参数：

```bash
export MODEL_ROOT=... DATA_LIST=...
MAX_STEPS=130 TIMING_ONLY=1 OPTIMIZER=dist_muon bash experiment/single_node_repro/train_pro5000.sh pro5000_distmuon
MAX_STEPS=130 TIMING_ONLY=1 OPTIMIZER=muon      bash experiment/single_node_repro/train_pro5000.sh pro5000_muon
```

环境上要注意：

- Blackwell 需要 CUDA 12.8 版的 PyTorch。`tools/create_train_env.sh` 默认从 PyPI 装 `torch==2.8.0`，那就是 cu128 版本，所以**不要**把 `PIP_INDEX_URL` 指到 cu126。
- 驱动版本至少 570。

本复现的结果见文首的"结果"一节。参考测量（`figures/pro5000_*.png`，不是本仓库测的）是：DistMuon 5.350 s，Muon 7.044 s。它和本复现有以下几处不同，对比时要注意：

- **数据**：不是 RoboTwin clean，而是另一套 13 个子集、649 条轨迹、643,977 帧的数据。
- **学习率**：base LR 用的是 5e-5，官方配置是 1e-4。学习率不影响速度。
- **Muon 有本地改动**：Muon 那组带了"BF16 通信"的修改，官方代码里没有这个开关。官方 Muon 的梯度 all-gather 用的是参数原本的 dtype，动作专家部分是 FP32。所以用官方代码测 Muon，可能比 7.04 s 略慢。
- **统计区间**：Muon 那组只跑了 60 步，统计的是 step 21–40 和 50–60。这里统一跑 130 步、统计 31–130。

## 4. 怎么看结果

- **训练速度**：看预热后的 step 耗时，默认取 step 31–130。前几步不能算，原因如下：
  - step 1 包含 `torch.compile` 编译，在 H20 上约 107 s，在 RTX PRO 5000 上约 20–60 s（取决于编译缓存）；
  - step 2–4 会重新编译，每步约 11 s。
- **StepTime**：trainer 自己记录的计时，不包含等待下一批数据的时间。
- **墙钟 / step**：相邻两条日志的时间差，包含取数据和写日志。
  - 两者只差约 0.01 s，说明数据加载基本被预取隐藏了。
  - 如果墙钟明显大于 StepTime，说明 GPU 在等数据（CPU 核数不够或磁盘太慢）。
- **P95**：100 个 step 里第 95 慢的耗时，用来看抖动。
- **5,000 步总用时**会比"平均 step 耗时 × 5000"多出：
  - 启动和编译的约 2 分钟；
  - 每次存 checkpoint 的约 60–75 s；
  - 每个 epoch 边界的约 10 s（每 epoch 约 2,148 步）。
- **显存**：分三层看：
  - PyTorch 张量分配峰值：真正被张量占用的显存。
  - 缓存池峰值：PyTorch 向驱动申请的总量，包含碎片。
  - nvidia-smi 整卡占用：在缓存池之外，还包括 CUDA context 和 NCCL 缓冲。
  - 前两项来自 `mem_rank*.csv`，最后一项来自 `smi.csv`。
- **过热降频**：只要有一张卡降频，所有卡的每一步都会被它拖慢。如果统计结果里出现 `thermal throttling on GPU ...`，说明这段 step 耗时混入了降频。测"不降频时的速度"要缩小统计区间（`--last`）；评估长时间训练的实际用时，则应该把降频算进去，跑更多步再统计。

### 抓 timeline（profiling）

两种方式都只抓 rank 0 的几个预热后的 step，训练代码不改。

**Nsight Systems（`.nsys-rep`，用 Nsight Systems 桌面端打开）**：设置 `NSYS_STEPS` 即可，录完最后一步后 nsys 会结束任务，报告在 `$OUT_ROOT/<run_name>/nsys_rank0.nsys-rep`。

```bash
MAX_STEPS=45 TIMING_ONLY=1 MEM_PROBE=0 NSYS_STEPS=31-35 OPTIMIZER=dist_muon \
  bash experiment/single_node_repro/train_pro5000.sh pro5000_distmuon_nsys
```

- `nsys_probe.py` 在 NVTX 行里标出 `step N` 和每次 `optimizer.step`（如 `CombinedOptimizer.step` 里嵌套 `DistMuon.step`、`AdamW.step`）。
- `NSYS_ARGS` 可加 nsys 参数，例如 `NSYS_ARGS="--pytorch=autograd-nvtx"` 给每个 PyTorch 算子打 NVTX（开销更大）。
- 命令行汇总：`nsys stats -r nvtx_sum,cuda_gpu_kern_sum nsys_rank0.nsys-rep`。

**PyTorch profiler（Chrome trace JSON，用 [Perfetto](https://ui.perfetto.dev) 或 `chrome://tracing` 打开）**：用 trainer 自带的开关。抓的是 `[profile_start_step, profile_end_step)`，下面是 step 31–32：

```bash
MAX_STEPS=36 TIMING_ONLY=1 MEM_PROBE=0 OPTIMIZER=dist_muon \
  bash experiment/single_node_repro/train_pro5000.sh pro5000_distmuon_torchprof \
  --train.enable_profiling true --train.profile_start_step 31 --train.profile_end_step 33 \
  --train.profile_trace_dir "$PWD/output/pro5000_distmuon_torchprof/trace" \
  --train.profile_record_shapes false --train.profile_profile_memory false --train.profile_with_stack false
```

- 2 步的 trace 约 250–290 MB（gzip 后约 20 MB）。`chrome://tracing` 打开这么大的文件很慢，建议用 Perfetto，它能直接打开 `.json.gz`。
- trace 里的 `ProfilerStep#30`、`#31` 对应训练的 step 31、32。
- trace 写完后，上游代码会因为调用不存在的 `helper.upload_trace` 退出，报 `AttributeError`。trace 文件已经完整写好，不影响使用。

**8×RTX PRO 5000 上的 timeline 结论**（nsys，rank 0，step 31–35 平均；录制时 step 耗时与不录时一致）：

| | Distributed Muon | Muon |
|---|---:|---:|
| 前向 + 反向（含梯度裁剪） | 4486 ms | 4484 ms |
| 其中 NCCL kernel / 未被计算重叠的部分 | 2482 / 857 ms | 2468 / 841 ms |
| `optimizer.step` | **299 ms** | **2419 ms** |
| 其中 GPU 计算（主要是 GEMM） | 210 ms | 1633 ms |
| 其中 NCCL / 未被计算重叠的部分 | 173 / 51 ms | 634 / 634 ms |

- 两者每步相差约 2.1 s，全部来自 `optimizer.step`：Muon 在每个 rank 上对所有矩阵算 Newton-Schulz，再做一次 all-gather；DistMuon 把这部分分摊到各卡。
- 前向 + 反向里，FSDP 的 all-gather 和 reduce-scatter 每步共约 2.5 s，其中约 0.85 s 没被计算重叠。NCCL 在这台只有 PCIe 的机器上选的是 `RING_LL` 协议。

## 5. 与官方的差异

| 项目 | 官方 | 本复现 | 说明 |
|---|---|---|---|
| batch 设置 | micro 32，全局 1024，按 32 卡写 | H20：micro 16 × 8 卡 × GA 2 = 256；RTX PRO 5000：micro 32 × 8 卡 × GA 1 = 256，开激活重算 | 在 8×H20 上不开激活重算时，micro 32 第 1 步前向就爆显存（已分配 91.9 GiB，整卡 95 GiB），所以 H20 用 micro 16 × GA 2。RTX PRO 5000 只有 72 GB，必须开激活重算 |
| 训练数据 | README 示例用 clean + randomized | 只用 clean | |
| 数据转换 | `experiment/robotwin/README.md` 用 RoboTwin `policy/pi0/generate.sh` 转成 LeRobot v2.1 | XPolicyLab 转成 LeRobot v3.0 | 训练代码两种格式都支持。速度基本不受影响，但 loss 和成功率不能保证与官方逐位一致 |
| step 耗时（H20） | 4.31 s（DistMuon）/ 5.93 s（Muon） | 4.620 s / 6.288 s | 官方没有公开 4.31 s 对应的 batch 设置和计时方式。在上面的 batch 设置下，8×H20 能跑通的最快配置就是这里的 4.62 s |

## 文件

| 文件 | 作用 |
|---|---|
| `download_models.sh` | 下载 3 个权重仓库，并检查 teacher 文件是否齐全 |
| `prepare_robotwin_clean.sh` | 下载、解压、转换 RoboTwin 2.0 clean 数据，生成数据列表 |
| `train.sh` | 用官方配置填好路径后启动 8 卡训练，同时记录显存和温度 |
| `train_pro5000.sh` | RTX PRO 5000 的设置：micro 32、GA 1、打开激活重算，用 `OPTIMIZER` 选 dist_muon 或 muon |
| `mem_probe.py` | 启动包装：训练代码不改，每秒记录各卡 PyTorch 显存峰值 |
| `nsys_probe.py` | 启动包装：训练代码不改，标出 step 和 optimizer.step 的 NVTX 区间，并在指定 step 开关 nsys 录制 |
| `nsys_rank0.sh` | `NSYS_STEPS` 打开时由 torchrun 调用：rank 0 在 nsys 下运行，其他 rank 正常运行 |
| `summarize_run.py` | 统计 step 耗时、墙钟、loss、显存峰值，并标出过热降频的 GPU |
| `figures/` | 8 张结果和参考图，见"结果 → 图" |
