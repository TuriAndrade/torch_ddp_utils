# torch_ddp_utils

Utilities for **PyTorch Distributed Data Parallel (DDP)** workflows—focused on **data loading**, **training**, and **evaluation** in multi-GPU environments.

This repo provides small, composable helpers you can drop into existing projects to speed up DDP bring-up and keep training code clean and reproducible.

---

## What’s inside

- **DDP-friendly data loading**  
  Samplers, collators, and loader utilities designed for `torch.distributed` launches.

- **Training utilities**  
  Boilerplate-light training loops, callbacks/hooks, checkpoint/logging helpers (DDP-aware).

- **Evaluation helpers**  
  Distributed metric aggregation and synchronization primitives for validation/testing.

- **Config & structure**  
  Lightweight patterns for organizing experiments and making runs reproducible across GPUs/nodes.

---

## Installation

```bash
git clone https://github.com/TuriAndrade/torch_ddp_utils.git
cd torch_ddp_utils
pip install -r requirements.txt
