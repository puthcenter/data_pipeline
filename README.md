# MwcpNe 

## 📖 Project Overview
This repository contains the implementation and benchmarking suite for the **Maximum Weight Parent Closure Problem with Negative Externalities (MwcpNe)**.

## 📂 Repository Structure

The project is organized into two main directories. Please refer to the specific README in each subdirectory for detailed documentation.

* **[`data/`](./data)**: Contains the dataset specifications and graph files.
    * *Processed real-world Alibaba data flow logs transformed into DAGs.*
    * *Details on file naming conventions and weight preprocessing.*
* **[`code/`](./code)**: Contains the source code for all algorithms and the benchmarking runner.
    * *Implementations of Greedy, SA, SASA, QPBO, ILP, SGHC, etc.*
    * *Automated benchmark runner and visualization tools.*

## 🚀 Quick Start

### 1. Prerequisites
Ensure you have a Python environment (Python 3.8+) set up. Install the required dependencies:

```bash
pip install pandas numpy networkx scipy matplotlib seaborn pulp osqp
```

### 2. Run the Benchmark
To execute the full benchmark suite across all available datasets and algorithms:

```bash
cd code
python main_runner.py
```

### 3. Visualize Results
After the benchmark completes, you can generate performance plots (convergence curves, similarity matrices) by running:

```bash
cd code
python plot_summary.py
```

## 🧠 Implemented Algorithms

| Algorithm | Type | Description |
| :--- | :--- | :--- |
| **Greedy** | Heuristic | A one-pass greedy strategy based on initial gain sorting. |
| **SGHC** | Heuristic | **Shapley-Guided Hill Climbing**. Uses Shapley value estimation for initialization. |
| **NGHC** | Heuristic | Net-Value Greedy Hill Climbing. Uses simple net value for initialization (Ablation for SGHC). |
| **SA** | Meta-heuristic | Standard Simulated Annealing with random restarts. |
| **SASA** | Meta-heuristic | **Structure-Aware Simulated Annealing**. Use synergy only and use conflict to guide simulated annealing.|
| **SASAX** | Meta-heuristic | Use synergy only or use conflict only. (Abaltion for SASA) |
| **QP / QPR** | Relaxation | Quadratic Programming Relaxation solved via OSQP with heuristic rounding. |
| **QPBO** | Relaxation | Quadratic Pseudo-Boolean Optimization using Roof Duality (Max-Flow/Min-Cut). |
| **ILP** | Exact | Integer Linear Programming solver (using CBC) for ground truth verification. |

## 📊 Evaluation Metrics
* **Objective Score:** The total weight of the selected parent closure minus negative externalities.
* **Convergence Time:** Time taken to reach the best solution (excluding I/O overhead).
* **Convergence Trend:** The evolution of the incumbent best-known solution (i.e., the best feasible solution found so far) for each algorithm over time or function evaluations.
* **Jaccard Similarity:** Measures the overlap of selected node sets between different algorithms.
