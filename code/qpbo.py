import os
import time
import pandas as pd
import networkx as nx
import numpy as np
from collections import defaultdict
import argparse
from typing import Dict, List, Set, Tuple

# ==============================================================================
# --- 0. Convergence Tracker (With Verbosity Control) ---
# ==============================================================================
class ConvergenceTracker:
    """
    Tracks optimization progress.
    Supports a 'verbose' flag to suppress console output during high-frequency 
    inner-loop updates (e.g., during the greedy repair phase).
    """
    def __init__(self, algorithm_name, dataset_name, save_path="convergence.csv", log_interval=50.0):
        self.algorithm_name = algorithm_name
        self.dataset_name = dataset_name
        self.save_path = save_path
        self.log_interval = log_interval
        
        self.start_time = time.time()
        self.last_log_time = self.start_time
        
        self.history = []
        self.current_best_value = -float('inf')
        
        self._log(0.0, -float('inf'))

    def update(self, value, round_num=None, verbose=True):
        """
        Updates the best solution found so far.
        
        Args:
            verbose (bool): If False, the update is recorded to history/CSV 
                            but not printed to the console. Useful for reducing I/O noise.
        """
        # Update only if the new value is strictly better (considering float precision)
        if value > self.current_best_value + 1e-9:
            self.current_best_value = value
            elapsed = time.time() - self.start_time
            self._log(elapsed, value, round_num)
            
            if verbose:
                msg = f"      [Monitor] New Best: {value:,.2f} @ {elapsed:.2f}s"
                if round_num is not None:
                    msg += f" | Round: {round_num}"
                print(msg)
            
            # Persist to CSV immediately to prevent data loss on crash
            self.save_to_csv()
            self.last_log_time = time.time()
            return True
        return False

    def tick(self):
        """
        Time-driven logging to ensure continuous data points in the plot.
        """
        now = time.time()
        if now - self.last_log_time >= self.log_interval:
            elapsed = now - self.start_time
            self._log(elapsed, self.current_best_value)
            self.save_to_csv()
            self.last_log_time = now

    def finalize(self, total_runtime):
        """Final log entry at the end of execution."""
        elapsed = time.time() - self.start_time
        final_time = max(elapsed, total_runtime)
        self._log(final_time, self.current_best_value)
        self.save_to_csv()

    def _log(self, elapsed, value, round_num=None):
        entry = {
            "Algorithm": self.algorithm_name,
            "Dataset": self.dataset_name,
            "Time_Elapsed": elapsed,
            "Best_Value": value
        }
        if round_num is not None:
            entry["Round"] = round_num
        self.history.append(entry)

    def save_to_csv(self):
        try:
            pd.DataFrame(self.history).to_csv(self.save_path, index=False)
        except Exception:
            pass

# ==============================================================================
# 1. Data Loading & Result Saving
# ==============================================================================
def load_graph_data(folder):
    """
    Loads graph topology and externality data from CSV files.
    """
    node_df = pd.read_csv(os.path.join(folder, "nodes.csv"))
    edge_df = pd.read_csv(os.path.join(folder, "edges.csv"))
    
    G = nx.DiGraph()
    node_id_to_idx = {}
    idx_to_node_id = {}
    
    for idx, row in node_df.iterrows():
        nid = int(row['id'])
        G.add_node(nid,
                   value=float(row['value']),
                   cost=float(row['incoming_cost']),
                   is_result=bool(row['is_result']))
        node_id_to_idx[nid] = idx
        idx_to_node_id[idx] = nid

    for _, row in edge_df.iterrows():
        G.add_edge(int(row['source']), int(row['target']))

    externality_dict = {}
    possible_names = ["externality.csv", "externality_matrix.csv"]
    ext_path = next((os.path.join(folder, n) for n in possible_names if os.path.exists(os.path.join(folder, n))), None)
    
    if ext_path:
        df = pd.read_csv(ext_path)
        for _, row in df.iterrows():
            u = int(row['source'])
            v = int(row['target'])
            p = float(row.get('penalty', row.get('weight', 0.0)))
            if p != 0:
                externality_dict[(u, v)] = p

    return G, externality_dict, node_id_to_idx, idx_to_node_id

def save_result(output_dir, selected_nodes, selected_edges, score, duration, result_nodes):
    """
    Saves the final optimization results.
    """
    os.makedirs(output_dir, exist_ok=True)
    pd.DataFrame({'id': sorted(list(selected_nodes))}).to_csv(os.path.join(output_dir, 'sub_nodes.csv'), index=False)
    pd.DataFrame(selected_edges, columns=['source', 'target', 'cost']).to_csv(os.path.join(output_dir, 'sub_edges.csv'), index=False)
    with open(os.path.join(output_dir, 'score.txt'), 'w') as f: f.write(f"{score:.10f}")
    with open(os.path.join(output_dir, 'time.txt'), 'w') as f: f.write(f"{duration:.6f}")
    pd.DataFrame({'id': sorted(list(result_nodes))}).to_csv(os.path.join(output_dir, 'result_nodes.csv'), index=False)
    with open(os.path.join(output_dir, 'result_nodes_count.txt'), 'w') as f: f.write(str(len(result_nodes)))

# ==============================================================================
# 2. Incremental Score Calculator
# ==============================================================================
class IncrementalScoreCalculator:
    """
    Helper class for efficient objective function calculation.
    """
    def __init__(self, G, externality_dict, ancestor_cache):
        self.G = G
        self.ancestor_cache = ancestor_cache
        self.node_vals = {n: d['value'] for n, d in G.nodes(data=True)}
        self.node_costs = {n: d['cost'] for n, d in G.nodes(data=True)}
        
        self.current_set = set()
        self.ref_counts = defaultdict(int)
        self.current_penalties = defaultdict(float)
        self.current_score = 0.0
        
        self.incoming_ext = defaultdict(list) 
        self.outgoing_ext = defaultdict(list)

    def _hard_reset_logic(self, externality_dict):
        """Rebuilds the internal externality maps."""
        self.incoming_ext = defaultdict(list)
        self.outgoing_ext = defaultdict(list)
        for (u, v), p in externality_dict.items():
            self.incoming_ext[v].append((u, p))
            self.outgoing_ext[u].append((v, p))

    def full_recalc(self, nodes):
        """
        Calculates the score from scratch for a given set of nodes.
        """
        self.current_set = set(nodes)
        self.ref_counts = defaultdict(int)
        self.current_penalties = defaultdict(float)
        
        cost_sum = 0.0
        for n in self.current_set:
            for a in self.ancestor_cache[n]:
                if self.ref_counts[a] == 0:
                    cost_sum += self.node_costs[a]
                self.ref_counts[a] += 1
        
        for u in self.current_set:
            for v, p in self.outgoing_ext[u]:
                if v in self.current_set:
                    self.current_penalties[v] += p
        
        val_sum = 0.0
        for n in self.current_set:
            net_val = max(0.0, self.node_vals[n] - self.current_penalties[n])
            val_sum += net_val
            
        self.current_score = val_sum - cost_sum

# ==============================================================================
# 3. QPBO Solver (Roof Duality Implementation)
# ==============================================================================
class QPBOSolver:
    """
    Implements the Quadratic Pseudo-Boolean Optimization (QPBO) solver using Roof Duality.
    Phase 1: Transforms the problem into a Min-Cut/Max-Flow problem on a graph of size 2N.
    Phase 2: Performs a heuristic rounding and greedy repair on the partial assignment.
    """
    def __init__(self, G, externality_dict, node_id_to_idx, idx_to_node_id):
        self.G_orig = G
        self.ext = externality_dict
        self.n2i = node_id_to_idx
        self.i2n = idx_to_node_id
        self.N = len(G.nodes)

    def solve_roof_duality(self, time_limit):
        """
        Executes the Roof Duality algorithm.
        Constructs a flow network where nodes x_i and x_i_bar represent variables and their negations.
        The Minimum Cut on this graph provides a partial optimal assignment (labeled 0 or 1),
        leaving some variables unlabeled (0.5).
        """
        print(f"    [QPBO] Building Roof Duality Graph (MaxFlow)...")
        start_build = time.time()
        
        flow_G = nx.DiGraph()
        S_NODE = 'source'
        T_NODE = 'sink'
        
        # 1. Linear Terms
        for n, data in self.G_orig.nodes(data=True):
            i = self.n2i[n]
            i_bar = i + self.N
            w_i = data['cost']
            if data.get('is_result'):
                w_i -= data['value']
            
            if w_i > 0:
                flow_G.add_edge(i, T_NODE, capacity=w_i)
                flow_G.add_edge(S_NODE, i_bar, capacity=w_i)
            elif w_i < 0:
                flow_G.add_edge(i_bar, T_NODE, capacity=-w_i)
                flow_G.add_edge(S_NODE, i, capacity=-w_i)

        # 2. Quadratic Terms (Externalities)
        for (u, v), penalty in self.ext.items():
            if u not in self.n2i or v not in self.n2i: continue
            i, j = self.n2i[u], self.n2i[v]
            i_bar, j_bar = i + self.N, j + self.N
            cap = penalty / 2.0
            
            # Add symmetric edges for quadratic term interactions
            flow_G.add_edge(i, j_bar, capacity=cap)
            flow_G.add_edge(j, i_bar, capacity=cap)
            flow_G.add_edge(i_bar, j, capacity=cap)
            flow_G.add_edge(j_bar, i, capacity=cap)

        # 3. Topology Constraints (Dependencies)
        # If u -> v (u is parent, v is child), then x_v <= x_u is required.
        # This translates to infinite capacity edges enforcing the closure.
        INF = float('inf')
        for u, v in self.G_orig.edges:
            parent_idx = self.n2i[u]
            child_idx = self.n2i[v]
            flow_G.add_edge(child_idx, parent_idx, capacity=INF)
            
            parent_bar = parent_idx + self.N
            child_bar = child_idx + self.N
            flow_G.add_edge(parent_bar, child_bar, capacity=INF)

        # 4. Solve Min-Cut
        if time.time() - start_build > time_limit:
            print("    [QPBO] Time out during graph construction.")
            return np.zeros(self.N)
            
        try:
            cut_value, partition = nx.minimum_cut(flow_G, S_NODE, T_NODE)
            reachable, non_reachable = partition
        except Exception as e:
            print(f"    [QPBO] MaxFlow failed: {e}")
            return np.zeros(self.N)

        # 5. Parse Results
        # x[i] = 1 if i in S and i_bar not in S
        # x[i] = 0 if i not in S and i_bar in S
        # x[i] = 0.5 (Unlabeled) otherwise
        x_result = np.zeros(self.N)
        labeled_count = 0
        for i in range(self.N):
            i_bar = i + self.N
            in_S = i in reachable
            bar_in_S = i_bar in reachable
            if in_S and not bar_in_S:
                x_result[i] = 1.0
                labeled_count += 1
            elif not in_S and bar_in_S:
                x_result[i] = 0.0
                labeled_count += 1
            else:
                x_result[i] = 0.5
        
        print(f"    [QPBO] Solved. Labeled {labeled_count}/{self.N} variables ({(labeled_count/self.N)*100:.1f}%).")
        return x_result

    def smart_rounding(self, x_cont, ancestor_cache, all_result_nodes, tracker, deadline):
        """
        Rounding Phase:
        Takes the partial assignment from QPBO (where some variables are 0.5) and 
        repairs it using a heuristic greedy strategy.
        """
        if time.time() >= deadline:
            return set(), -float('inf'), set()

        print(f"    [Rounding] Starting QPBO-Guided Greedy Repair...")

        # --- Step 1: Classify Nodes ---
        # Fixed: Nodes strongly labeled as 1 by QPBO.
        # Uncertain: Nodes labeled 0.5 or weakly labeled, prioritized by heuristic.
        fixed_keep = set() 
        uncertain = []
        node_priority = {}
        
        for n in all_result_nodes:
            idx = self.n2i[n]
            val = x_cont[idx]
            if val > 0.9:
                fixed_keep.add(n)
            elif 0.4 < val < 0.6: 
                # Heuristic: Net Value / (1 + Conflict Degree)
                net_val = self.G_orig.nodes[n]['value'] - self.G_orig.nodes[n]['cost']
                conflict_deg = self.G_orig.degree[n] 
                priority = net_val / (1.0 + conflict_deg * 0.1)
                uncertain.append(n)
                node_priority[n] = priority

        uncertain.sort(key=lambda n: node_priority[n], reverse=True)

        # --- Step 2: Build Initial Solution ---
        current_selection = set(fixed_keep)
        inc_calc = IncrementalScoreCalculator(self.G_orig, self.ext, ancestor_cache)
        inc_calc._hard_reset_logic(self.ext)
        inc_calc.full_recalc(current_selection)
        
        best_score = inc_calc.current_score
        best_set = current_selection.copy() 
        
        # Log the baseline score established by the fixed nodes
        tracker.update(best_score, round_num="QPBO_Base", verbose=True)
        
        # --- Step 3: Greedy Attempt on Uncertain Nodes (Silent Mode) ---
        step_count = 0
        added_count = 0
        
        for cand in uncertain:
            if time.time() > deadline: break
            if cand in current_selection: continue
                
            test_set = current_selection.union({cand})
            inc_calc.full_recalc(test_set)
            new_score = inc_calc.current_score
            
            if new_score > best_score:
                best_score = new_score
                current_selection = test_set
                best_set = current_selection.copy()
                added_count += 1
                # Silent update to avoid flooding the console
                tracker.update(best_score, round_num=f"Greedy_Add_{step_count}", verbose=False)
            
            step_count += 1
            # Revert calculation state if not accepted (re-run full_recalc on current_selection)
            if new_score <= best_score:
                 inc_calc.full_recalc(current_selection)

        # --- Step 4: Finalize ---
        print(f"      [Rounding] Finished. Added {added_count} nodes. Final Score: {best_score:,.2f}")

        # Expand to full closure
        final_closure = set()
        for n in best_set:
            final_closure.update(ancestor_cache[n])

        return final_closure, best_score, best_set

# ==============================================================================
# 4. Execution Logic
# ==============================================================================
def run_solver(
    G: nx.DiGraph,
    externality_dict: Dict,
    node_id_to_idx: Dict, 
    idx_to_node_id: Dict,
    dataset_name: str,
    result_dir: str,
    algorithm_name: str = "qpbo",
    max_runtime: float = 2000.0
):
    os.makedirs(result_dir, exist_ok=True)
    
    tracker = ConvergenceTracker(
        algorithm_name, 
        dataset_name, 
        save_path=os.path.join(result_dir, "convergence.csv"),
        log_interval=10.0
    )
    
    start_time = time.time()
    deadline = start_time + max_runtime 
    
    result_nodes = [n for n, d in G.nodes(data=True) if d.get('is_result')]
    if not result_nodes:
        tracker.finalize(max_runtime)
        return set(), [], 0.0, set()

    ancestor_cache = {n: nx.ancestors(G, n) | {n} for n in result_nodes}
    
    solver = QPBOSolver(G, externality_dict, node_id_to_idx, idx_to_node_id)
    
    # --- Phase 1: QPBO (MaxFlow) ---
    remaining = deadline - time.time()
    if remaining > 1.0:
        x_cont = solver.solve_roof_duality(time_limit=remaining)
    else:
        x_cont = np.zeros(len(G.nodes))

    # --- Phase 2: Rounding & Repair ---
    closure, score, selected = solver.smart_rounding(
        x_cont, ancestor_cache, result_nodes, tracker, deadline=deadline
    )
    
    tracker.finalize(max_runtime)
    
    edges = [(u, v, G[u][v].get('weight', 0.0)) for u in closure for v in G.successors(u) if v in closure]
    
    return closure, edges, score, selected

def run_experiment(root: str, algo_name: str, max_runtime: float):
    if not os.path.exists(root):
        print(f"Error: Directory {root} does not exist.")
        return

    for folder in os.listdir(root):
        path = os.path.join(root, folder)
        if not os.path.isdir(path): continue

        try:
            print(f"--- Processing [{folder}] Algo: {algo_name} ---")

            G, ext_dict, n2i, i2n = load_graph_data(path)
            result_dir = os.path.join(path, f"{algo_name}_result")
            
            start = time.time()
            
            closure, edges, score, selected = run_solver(
                G, ext_dict, n2i, i2n,
                dataset_name=folder,
                result_dir=result_dir,
                algorithm_name=algo_name,
                max_runtime=max_runtime
            )
            
            duration = time.time() - start
            save_result(result_dir, closure, edges, score, duration, selected)
            print(f"[{folder}] Final Score={score:,.4f}, Time={duration:.2f}s")
            print("=" * 60)

        except Exception as e:
            import traceback
            print(f"[{folder}] Failed: {e}")
            traceback.print_exc()

if __name__ == "__main__":
    MAX_RUNTIME = 1000.0
    ALGO_NAME = "qpbo" 
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str, default="../data/output")
    args = parser.parse_args()
    
    run_experiment(root=args.root_dir, algo_name=ALGO_NAME, max_runtime=MAX_RUNTIME)