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
