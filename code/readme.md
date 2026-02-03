## 🚀 One-Click Execution

To run the full benchmark pipeline, simply execute the main runner script. This script handles the sequential execution of algorithms, data aggregation, and calls the visualization module automatically.

```bash
python main_runner.py
```

### 📊 Visualization & Configuration

The visualization module is located at `plot_summary.py`. 

**Note on Customization:** You should manually edit `plot_summary.py` to control the visualization output. The script does not take command-line arguments; instead, modify the configuration variables at the top of the file:

1.  **Select Algorithms (`TARGET_ALGOS`):**
    Modify this list to determine which algorithms appear in the plots. Comment out any algorithm you wish to exclude.
    
    ```python
    TARGET_ALGOS = [
        'sasa', 
        'sasa_synergy_only',
        # 'greedy',  <-- This line is commented out and will not be plotted
        'sa',
    ]
    ```
    
2.  **Time Limit (`MAX_PLOT_TIME`):**
    Set the maximum duration (in seconds) for the X-axis.
    ```python
    MAX_PLOT_TIME = 450  # Plots will cut off at 450 seconds
    ```

After modifying the configuration, you can regenerate just the plots without re-running the benchmarks by executing:

```bash
python plot_summary.py
```

## ⚙️ Algorithm Configuration

Each algorithm script (e.g., `greedy.py`, `sa.py`, `sasa.py`) is standalone and contains configurable parameters at the bottom of the file (typically within the `if __name__ == "__main__":` block or the `run_solver` function call).

* **Maximum Runtime (`MAX_RUNTIME`):** Controls the hard time limit (in seconds) for the optimization process. The algorithm will attempt to finalize and save the best result found once this limit is reached.
* **Logging Interval (`log_interval`):** Determines the frequency (in seconds) at which the current best solution is recorded to `convergence.csv`. Lower values provide higher resolution convergence plots but may introduce slight overhead.

### ⏱️ Note on Time Measurement

To ensure a fair benchmark of the optimization logic itself, the reported execution time **strictly excludes** the following overheads:
1.  **Data Loading:** Reading the graph structure and weights from disk.
2.  **Result Persistence:** The final generation of the ancestor closure (calculating the full set of dependent nodes) and writing result files (nodes, edges, CSVs) to the disk.

The recorded time reflects **only the core computational phase** of the algorithm.