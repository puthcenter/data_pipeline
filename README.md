# PiPE

## Project Overview

This repository contains the implementation and benchmarking suite for **Profit Maximization in Data Pipelines with Externalities (PiPE)** — a combinatorial optimization problem over directed acyclic graphs (DAGs) where selecting a set of nodes yields value from their descendant "result" nodes but incurs pairwise conflict penalties.

> **Data Availability Notice**
>
> The algorithm design and parameter calibration are informed by real-world data-flow graph characteristics from Alibaba's production environment. **The original production data cannot be disclosed due to trade secret protection.** A single de-identified real-world sample (`dag121037_MEDIUM_0`, ~109K nodes, ~564K edges) is provided in `data/output/` for reference. Additional samples may be added in the future. For full-scale benchmarking, use the synthetic data generator `data/pipeline_set.py`.

> **Project Status**: Under active development. Watch this repo for updates.

## Repository Structure

* **[`data/`](./data)**: Synthetic dataset generator and sample data.
    * `pipeline_set.py` — End-to-end synthetic DAG generation pipeline: topology manifold construction → three-stage economic valuation (ROI, closure cost, externality conflict modeling) → multi-dimensional sensitivity experiment design. Produces 4 dimensions × 3 levels × 3 versions = 36 benchmark datasets.
    * `data/output/` — Generated dataset directory (**mostly excluded from the repo**). Currently contains one de-identified real-world sample: `dag121037_MEDIUM_0`.
* **[`code/`](./code)**: Algorithm implementations and benchmarking framework.
    * `main_runner.py` — Automated benchmark runner: executes all algorithms, aggregates results, and invokes plotting.
    * `plot_summary.py` — Visualization tools (convergence curves, Jaccard similarity matrices).
    * `gather.py` — Convergence knee-point analysis using the Kneedle algorithm.

## Quick Start

### Prerequisites

Python 3.8+:

```bash
pip install pandas numpy networkx scipy matplotlib seaborn pulp osqp
```

### One-Click Run

```bash
# 1. Generate synthetic datasets
cd data
python pipeline_set.py

# 2. Run all benchmarks (uncomment run_all_methods() in main_runner.py first)
cd ../code
python main_runner.py
```

You can also run individual algorithms:

```bash
cd code
python greedy.py
python sasa.py
```

## Implemented Algorithms

| Algorithm | Type | Description |
| :--- | :--- | :--- |
| **Greedy** | Heuristic | One-pass greedy strategy based on initial marginal gain sorting with parent closure expansion. |
| **NGHC** | Heuristic | Net-Value-Guided Hill Climbing. Ablation variant of SGHC — uses simple net value (value − cost) for initialization instead of Shapley estimation. |
| **SGHC** | Heuristic | **Shapley-Guided Hill Climbing**. Uses Shapley value estimation over the externality graph to initialize the search, then performs local hill climbing. |
| **SA** | Meta-heuristic | Standard Simulated Annealing with adaptive temperature scheduling and random restarts. |
| **SASA** | Meta-heuristic | **Structure-Aware Simulated Annealing**. Leverages both synergy and conflict structure from the externality graph to guide neighborhood exploration. |
| **SASAX** | Meta-heuristic | Ablation variant of SASA — uses synergy-only or conflict-only guidance to isolate the contribution of each structural signal. |
| **QP** | Relaxation | Quadratic Programming relaxation via OSQP solver with heuristic rounding to feasible integer solutions. |
| **QPBO** | Relaxation | Quadratic Pseudo-Boolean Optimization via Roof Duality, solved as a max-flow/min-cut problem on a specially constructed graph. |
| **SCIP** | Exact | Integer Linear Programming solver using PuLP + SCIP backend, for ground truth verification on small-to-medium instances. |

## Evaluation Metrics

* **Objective Score**: Total value of selected result nodes minus pairwise conflict penalties.
* **Convergence Time**: Wall-clock time to reach the best feasible solution.
* **Convergence Trend**: Evolution of the incumbent best-known solution over time / function evaluations.
* **Jaccard Similarity**: Overlap of selected node sets between different algorithms, measuring solution-structure agreement.
