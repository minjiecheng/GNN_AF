# GNN-AF

This repository contains the implementation accompanying the WWW 2026 paper:

**Revisiting and Enhancing Graph Neural Networks through the Lens of Amortized Flows**  

## Overview

This codebase studies graph neural networks from the perspective of amortized flows and provides a unified training pipeline under a common formulation. The current release focuses on node classification and includes the experimental setup used for the paper, together with a runnable example on the **Squirrel** dataset.

The implementation supports both the proposed amortized-flow variant and the corresponding original model setting, making it straightforward to reproduce comparisons reported in the paper.

## Repository Structure

```text
.
├── train.py             # main training and evaluation script
├── models.py            # model architectures
├── models_mp.py         # propagation and message-passing modules
├── dataset_loader.py    # dataset processing and loading
├── utils.py             # utility functions
└── data/
    └── squirrel/        # included example dataset; other supported datasets will be downloaded automatically
```

## Environment

The code has been verified with the following software stack:

- Python `3.8`
- PyTorch `1.12.1`
- PyTorch Geometric `2.0.3`
- `torch-scatter` `2.0.9`
- `torch-sparse` `0.6.18`
- `numpy`, `scipy`, `matplotlib`, `seaborn`, `tqdm`

## Running the Code

### Main Example: Reproducing the Squirrel Result

To reproduce the main `GCN` result on `Squirrel`, run:

```bash
python train.py \
  --Original_ot ot \
  --lr 0.05 \
  --weight_decay 0.0005 \
  --lambda_ 100.0 \
  --q_linear_lr 0.001 \
  --q_linear_delay 0.0 \
  --dataset Squirrel \
  --net GCN
```

Then you will get the experimental result for `Squirrel` of `GCN+AF` in Table 3, which is roughly as follows.

> GCN on dataset Squirrel, in 10 repeated experiment: test acc mean = 53.5735 ± 0.6535         val acc mean = 53.961

The default configuration runs `10` repeated experiments and reports the mean test performance.

### Changing the Architecture

The `--net` argument currently supports:

```text
GCN, GAT, GIN, GSAGE, APPNP, BernNet, ChebNetII
```

The `--dataset` argument currently supports:

```text
Cora, Citeseer, Pubmed, Computers, Photo, Chameleon, Squirrel, Actor, Texas, Cornell
```

## Important Arguments

| Argument | Description |
| --- | --- |
| `--lambda_` | Trade-off coefficient in the amortized-flow objective. |
| `--q_linear_lr` | Learning rate for the flow-related linear layer. |
| `--q_linear_delay` | Weight decay for the flow-related linear layer. |

## Citation

If you find this repository useful in your research, please cite:

```bibtex
@inproceedings{cheng2026revisiting,
  title={Revisiting and Enhancing Graph Neural Networks through the Lens of Amortized Flows},
  author={Cheng, Minjie and Yan, Bokai and Luo, Dixin and Xu, Hongteng},
  booktitle={Proceedings of the ACM Web Conference 2026},
  pages={1422--1432},
  year={2026}
}
```
