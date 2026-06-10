# PiPE

## 📖 Project Overview

This repository contains the implementation and benchmarking suite for **Profit Maximization in Data Pipelines with Externalities (PiPE)** — a combinatorial optimization problem over directed acyclic graphs (DAGs) where selecting a set of nodes yields value from their descendant "result" nodes but incurs pairwise conflict penalties.

> **⚠️ 数据说明 / Data Availability Notice**
>
> 本项目的算法设计和参数校准参考了阿里巴巴真实数据流日志中的图结构特征。**由于商业秘密保护，原始生产数据无法提供。** 仓库中 `data/pipeline_set.py` 是一个完整的合成数据生成器，它基于拓扑流形（Topology Manifold）方法生成与真实数据统计特性一致的 DAG 数据集，可直接用于算法复现和基准测试。
>
> The algorithm design and parameter calibration are informed by real-world data-flow graph characteristics from Alibaba's production environment. **The original production data cannot be disclosed due to trade secret protection.** Instead, `data/pipeline_set.py` provides a full synthetic data generator based on a topology manifold approach, producing DAG datasets with statistical properties consistent with real-world observations. De-identified data samples may be added in the future.

> **🏗️ 项目状态 / Project Status**: 持续更新中。欢迎 Watch 本仓库以获取更新。 / Under active development. Watch this repo for updates.

## 📂 Repository Structure

* **[`data/`](./data)**: 合成数据集生成器 / Synthetic dataset generator.
    * `pipeline_set.py` — 完整的合成 DAG 生成管线：拓扑流形构建 → 三阶段经济学赋值（ROI 回报率、闭包成本、外部性冲突建模）→ 多维度敏感度实验设计。共生成 4 维度 × 3 水平 × 3 版本 = 36 个 benchmark 数据集。
    * `data/output/` — 生成的数据集输出目录（**不含在仓库中**，需运行生成器产生）。
* **[`code/`](./code)**: 算法实现与基准测试框架 / Algorithm implementations and benchmarking framework.
    * `main_runner.py` — 自动化基准测试运行器，批量执行各算法、汇总结果并调用绘图
    * `plot_summary.py` — 可视化工具（收敛曲线、Jaccard 相似度矩阵）
    * `gather.py` — 基于 Kneedle 算法的收敛拐点分析

## 🚀 一键运行 / One-Click Run

```bash
# 1. 生成合成数据集
cd data
python pipeline_set.py

# 2. 运行全部 benchmark（取消 main_runner.py 中 run_all_methods() 的注释后）
cd ../code
python main_runner.py
```

也可以单独运行某个算法，例如：

```bash
cd code
python greedy.py      # 仅运行贪心算法
python sasa.py        # 仅运行 SASA
```

### 依赖安装 / Prerequisites

Python 3.8+:

```bash
pip install pandas numpy networkx scipy matplotlib seaborn pulp osqp
```

## 🧠 算法列表 / Implemented Algorithms

| Algorithm | Type | Description |
| :--- | :--- | :--- |
| **Greedy** | Heuristic | One-pass greedy strategy based on initial marginal gain sorting with parent closure expansion. |
| **NGHC** | Heuristic | **Net-Value Greedy Hill Climbing**. Ablation variant of SGHC — uses simple net value (value − cost) for initialization instead of Shapley estimation. |
| **SGHC** | Heuristic | **Shapley-Guided Hill Climbing**. Uses Shapley value estimation over the externality graph to initialize the search, then performs local hill climbing. |
| **SA** | Meta-heuristic | Standard Simulated Annealing with adaptive temperature scheduling and random restarts. |
| **SASA** | Meta-heuristic | **Structure-Aware Simulated Annealing**. Leverages both synergy and conflict structure from the externality graph to guide neighborhood exploration. |
| **SASAX** | Meta-heuristic | Ablation variant of SASA — uses synergy-only or conflict-only guidance to isolate the contribution of each structural signal. |
| **QP** | Relaxation | Quadratic Programming relaxation via OSQP solver with heuristic rounding to feasible integer solutions. |
| **QPBO** | Relaxation | Quadratic Pseudo-Boolean Optimization via Roof Duality, solved as a max-flow/min-cut problem on a specially constructed graph. |
| **SCIP** | Exact | Integer Linear Programming solver using PuLP + SCIP backend, for ground truth verification on small-to-medium instances. |

## 📊 评估指标 / Evaluation Metrics

* **Objective Score**: Total value of selected result nodes minus pairwise conflict penalties.
* **Convergence Time**: Wall-clock time to reach the best feasible solution.
* **Convergence Trend**: Evolution of the incumbent best-known solution over time / function evaluations.
* **Jaccard Similarity**: Overlap of selected node sets between different algorithms, measuring solution-structure agreement.
