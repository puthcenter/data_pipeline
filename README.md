# PiPE

## 📖 Project Overview

This repository contains the implementation and benchmarking suite for **Profit Maximization in Data Pipelines with Externalities (PiPE)** — a combinatorial optimization problem over directed acyclic graphs (DAGs) where selecting a set of nodes yields value from their descendant "result" nodes but incurs pairwise conflict penalties.

> **⚠️ 数据说明 / Data Availability Notice**
>
> 本项目的算法设计和参数校准参考了阿里巴巴真实数据流日志中的图结构特征。**由于商业秘密保护，原始生产数据无法提供。** 仓库中 `data/pipeline_set.py` 是一个完整的合成数据生成器，它基于拓扑流形（Topology Manifold）方法生成与真实数据统计特性一致的 DAG 数据集，可直接用于算法复现和基准测试。
>
> The algorithm design and parameter calibration are informed by real-world data-flow graph characteristics from Alibaba's production environment. **The original production data cannot be disclosed due to trade secret protection.** Instead, `data/pipeline_set.py` provides a full synthetic data generator based on a topology manifold approach, producing DAG datasets with statistical properties consistent with real-world observations. De-identified data samples may be added in the future.

> **🏗️ 项目状态**: 持续更新中。算法和实验设计仍在积极迭代，欢迎关注。 / This project is under active development. Algorithms and experiment designs are being actively iterated. Stay tuned.

## 📂 Repository Structure

* **[`data/`](./data)**: 合成数据集生成器与配置 / Synthetic dataset generator and configuration.
    * `pipeline_set.py` — 完整的合成 DAG 生成管线，包含拓扑流形构建、三阶段经济学赋值（ROI / 闭包成本 / 外部性冲突）和敏感度实验设计（4 维度 × 3 水平 × 3 版本 = 36 个数据集）。
    * `data/output/` — 生成的数据集输出目录（**不含在仓库中**，需运行 `pipeline_set.py` 生成）。
* **[`code/`](./code)**: 算法实现与基准测试框架 / Algorithm implementations and benchmarking framework.
    * 启发式方法：Greedy, SGHC (Shapley-Guided Hill Climbing)
    * 元启发式方法：SA (Simulated Annealing), SASA (Structure-Aware SA)
    * 松弛/近似方法：QP (OSQP 二次规划松弛), QPBO (Roof Duality QPBO), MF (Max-Flow)
    * 精确方法：ILP (PuLP/SCIP, 用于 Ground Truth)
    * `main_runner.py` — 自动化基准测试运行器，批量执行各算法并汇总结果
    * `plot_summary.py` — 可视化工具（收敛曲线、Jaccard 相似度矩阵）
    * `gather.py` — 基于 Kneedle 算法的收敛拐点分析

## 🚀 Quick Start

### 1. Prerequisites

Python 3.8+ with the following dependencies:

```bash
pip install pandas numpy networkx scipy matplotlib seaborn pulp osqp
```

### 2. Generate Synthetic Datasets

```bash
cd data
python pipeline_set.py
```

This will populate `data/output/` with 36 benchmark datasets across four sensitivity dimensions (graph depth, density, product count, conflict intensity) at three levels each, with three random versions per configuration.

### 3. Run the Benchmark

```bash
cd code
python main_runner.py
```

### 4. Visualize Results

```bash
cd code
python plot_summary.py
```

## 🧠 Implemented Algorithms

| Algorithm | Type | Description |
| :--- | :--- | :--- |
| **Greedy** | Heuristic | One-pass greedy strategy based on initial marginal gain sorting with parent closure expansion. |
| **SGHC** | Heuristic | **Shapley-Guided Hill Climbing**. Uses Shapley value estimation over the externality graph to initialize the search, then performs local hill climbing. |
| **SA** | Meta-heuristic | Standard Simulated Annealing with adaptive temperature scheduling and random restarts. |
| **SASA** | Meta-heuristic | **Structure-Aware Simulated Annealing**. Leverages synergy/conflict structure from the externality graph to guide the neighborhood exploration. |
| **QP** | Relaxation | Quadratic Programming relaxation via OSQP solver with heuristic rounding to feasible integer solutions. |
| **QPBO** | Relaxation | Quadratic Pseudo-Boolean Optimization via Roof Duality, solved as a max-flow/min-cut problem on a specially constructed graph. |
| **MF** | Relaxation | Max-Flow-based formulation for solving a relaxed version of the selection problem. |
| **ILP** | Exact | Integer Linear Programming formulation solved via PuLP/SCIP, used for ground truth verification on small-to-medium instances. |

## 📊 Evaluation Metrics

* **Objective Score**: Total value of selected result nodes minus pairwise conflict penalties.
* **Convergence Time**: Wall-clock time to reach the best feasible solution.
* **Convergence Trend**: Evolution of the incumbent best-known solution over time / function evaluations.
* **Jaccard Similarity**: Overlap of selected node sets between different algorithms, measuring solution-structure agreement.
