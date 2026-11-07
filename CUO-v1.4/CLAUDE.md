# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

> 本仓库的注释与文档以中文为主，回复请使用中文。

## 项目概述

这是论文 **Continual Low-Rank Adaptation via Cumulative Unified Optimization (CUO)** 的实验代码，研究**类增量持续学习**（Class-Incremental Continual Learning）：在预训练 ViT 上用 LoRA 逐任务微调，并通过一个**闭式（ridge regression）解 Theta** 把至今所有任务的增量"统一"成单一权重，从而在不回放旧数据的前提下抵抗灾难性遗忘。

非 git 仓库；无测试、无 lint。这是单机单卡跑的研究脚本，唯一产物是 stdout 日志（含精度矩阵）。

## 环境与运行

```bash
pip install -r requirements.txt          # Python 3.12, torch 2.2.2, timm 1.0.9
```

- 额外依赖：`torchvision.set_image_backend('accimage')`（见 [utils/train_funcs.py](utils/train_funcs.py) `seed_etc_options`），**必须装 accimage**，否则数据加载报错。
- 强制使用 fused attention（`TIMM_FUSED_ATTN=1`，在 [train_eval.py](train_eval.py) 顶部设置）。非 fused 路径与 `torch.compile` 都是 `raise NotImplementedError`。
- 首次运行时 `find_not_pretrained_params` 会从 HuggingFace Hub **下载预训练权重做参数比对**，需要联网（已对 SSL 错误做容错）。

### 跑实验

每个实验占一张卡，通过 `CUDA_VISIBLE_DEVICES` 指定。完整批量见 [run_scripts.sh](run_scripts.sh)，单条示例：

```bash
CUDA_VISIBLE_DEVICES=0 bash script_ir.sh   -t 10 --data_root "/path/to/imageNet-R"   # 10-split ImageNet-R
CUDA_VISIBLE_DEVICES=0 bash script_c100.sh -t 10 --data_root "/path/to/cifar100"     # 10-split CIFAR-100
CUDA_VISIBLE_DEVICES=1 bash script_idn.sh  -t 10 --data_root "/path/to/DomainNet"    # 10-split DomainNet
CUDA_VISIBLE_DEVICES=2 bash script_ia.sh   -t 20 --data_root "/path/to/imageNet-A"   # 20-split ImageNet-A
```

`-t` 指定任务数（划分数），`--seed`（默认 2025）换随机种子复现多次实验。

### 脚本三层结构（真正的配置入口是脚本，不是 `get_args` 默认值）

`get_args` 里很多默认值（如 `lr=0.01`、`lora_rank=0`）**不是实际跑的配置**，会被脚本覆盖。要改超参先看脚本：

1. **[run_scripts.sh](run_scripts.sh)** —— 指定 GPU 与 `--data_root`，逐个调数据集脚本。
2. **`script_<dataset>.sh`**（[ir](script_ir.sh)/[c100](script_c100.sh)/[idn](script_idn.sh)/[ia](script_ia.sh)）—— 选数据集，开启 `--use_theta_bl/--use_theta_h/--use_ftinit`。
3. **[script_base.sh](script_base.sh)** —— 公共超参：`--training_string lora head --lora_rank 8 --lr 0.0001 -b 256 --lr_scale_patterns head --lr_scale 100`（head 用 100× 学习率）。

### 数据准备

四个数据集统一成 `DATA_ROOT/{train,val}/<class_folder>/<image>` 结构。用 [tools/](tools/) 下的 `split_*.py`（cifar100 / imagenet_r / sdomainnet）重排，运行前需**改脚本里的 `root_dir`**，并在 `tools/` 目录内执行（脚本用相对路径读 `*_train.txt`/`*_val.txt`，且用 symlink 建目录）。ImageNet-A 无 split 脚本，需手动整理。

## 架构

### 主流程：[train_eval.py](train_eval.py) `main_func`

外层按任务迭代 `GVM.cl_mngr`，每个任务：`timm.create_model`（仅 task 0 建模型）→ `train_one_task` → `evaluate_tasks_sofar`。`train_one_task` 内的关键顺序是：**先正常 LoRA 训练 → 存 checkpoint → 再算 Theta_bl（backbone）→ 再算 Theta_h（head）**。

### GVM：贯穿全局的状态对象（[utils/train_funcs.py](utils/train_funcs.py) `GlobalVarsManager`）

几乎每个函数的第一个参数都是 `GVM`。它是一个**上帝对象**，状态主要通过 `GVM.cache_dict` 在函数间**隐式传递**——读改这部分代码前务必清楚 cache_dict 里有什么：
- `args` / `cl_mngr`（任务划分）/ `acc_mat_dict`（精度矩阵）/ `param_dict`（各任务参数 `task_params_{t}` 与 `base_params`）/ `label_map_g2l`。
- `cache_dict` 关键键：`'Theta_bl_t-cums'`、`'Theta_h_t-cums'`（跨任务累积的 F/G/P 等求解分量）、`'param_id_dict'`、`'not_pretrained_params'`、`'training_string'`。

### CUO 核心算法（[utils/interm_tensor_funcs.py](utils/interm_tensor_funcs.py)）

这是论文的核心，分两套**闭式岭回归解**，每个任务训练完后增量更新：

- **Theta_bl_t（backbone）**：只作用于每个 block 的 `qkv.weight`（见 `get_param_id_dict(['qkv.weight'])`）。前向时通过 hook 收集输入 `Xi`、输出 `Yi=（W0+LoRA）x`、原权重 `W0`；累积 `Ft += XiᵀXi`、`Gt += Xiᵀ(Yi - Xi·W0)`，解 `Theta_bl_t = (Ft + λI)⁻¹ Gt`。
- **Theta_h_t（分类头）**：用 `theta_h_alpha` 衰减旧任务 logits、`theta_h_beta` 放大当前任务 one-hot 标签，累积后同样闭式求解。

数值稳定性：最终线性方程求解强制使用 **float64**。Backbone 的 X/Y 不再全量缓存，而是在 `--theta_device`（默认 CUDA）上按 batch 流式累计 F/G；跨任务累计量仍存为 CPU/float32，避免 50-task 时长期占用显存。

### 改造版 ViT（[utils/vit_builder.py](utils/vit_builder.py)，~2800 行）

基于 timm `VisionTransformer` 改写，三处关键改动：
- **`LoRALayer`**：LoRA 加在每个 block 的 `qkv`（及可选 `MLP.fc2`），但 backbone CUO 只对 `qkv.weight` 求解。
- **`IntermReader`**：空操作 `nn.Module`，靠 forward hook + `id(param)` 抓中间张量给 CUO 用。
- **Theta buffer 与推理开关**：`Theta_bl_t`/`Theta_h_t` 存为 buffer；推理时由 `set_model_flag` 翻转 `use_Theta_bl_t_for_infer`/`use_Theta_h_t_for_infer`，前向改用 `qkv.weight + Theta_bl_t.T` 和 `head.weight + Theta_h_t` 替代 LoRA/原头。这些开关在评估时开、评估后关。

### 其他模块

- **[utils/continual_manager.py](utils/continual_manager.py)** `ClassIncrementalManager`：按 seed 打乱类别并切成 `num_tasks` 份；提供 `current_task_classes` / `sofar_task_classes`，本身是迭代器。
- **[utils/dataset_builder.py](utils/dataset_builder.py)**：四个数据集各一个 `ImagePathDataset` 子类 + `define_dataset` 统一构造（含增广、`expand_times` 重复采样、全局→局部标签映射）。
- **[utils/misc.py](utils/misc.py)**：`ScalarMeter`（指标累计/格式化）、`calc_forgetting`、`set_logger`（仅 stdout）、`str2bool` 等。

## 关键约定 / 易错点

- **`seperate_head`（默认开）**：每个任务一个独立 head，训练时只看本任务类；评估时 `modify_head` 把所有任务 head 权重**拼接**成一个大头。
- 训练用 **temperature scaling**：loss 是 `CE(logits / temperature)`，`temperature=30`。
- 三个评估指标：**task-incremental**（只在当前任务类内 argmax）、**class-incremental**（在所有见过的类内 argmax）、以及对应的 **forgetting**；结果存 `acc_mat_dict` 的 `[train_taskid, eval_taskid]` 矩阵，由 `task_ending_info` 打印。
- **Forward Transfer Init（`use_ftinit`）**：新任务的 LoRA 参数用之前所有任务 LoRA 参数的**均值**初始化（[train_eval.py](train_eval.py) `train_one_task` 开头）。
- 改 backbone CUO 作用的层时，注意 `get_param_id_dict` 末尾有 `assert` 校验参数数量 == `lora_end_block - lora_start_block + 1`。
