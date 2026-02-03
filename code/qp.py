import os
import time
import random
import pandas as pd
import networkx as nx
import numpy as np
from collections import defaultdict
import argparse
from scipy.sparse import csr_matrix, vstack, identity
import osqp
from typing import Dict, List, Set, Tuple

"""
NOTE: Algorithm Identification
------------------------------
Although this module and the output directories are named 'qp', 
this implementation corresponds to the **QPR (Quadratic Programming Relaxation)** algorithm.

It consists of two phases:
1. Relaxation: Solving a continuous Quadratic Programming problem using OSQP.
2. Rounding: Converting the continuous solution to discrete node selections via thresholding.
"""

# ==============================================================================
# --- 0. Enhanced Convergence Tracker (Event & Time Driven) ---
# ==============================================================================
class ConvergenceTracker:
    """
    Tracks optimization progress using a hybrid logging strategy.
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

    def update(self, value, round_num=None):
        """
        [Event-Driven] Logs immediately when a new best solution is found.
        """
        if value > self.current_best_value + 1e-9:
            self.current_best_value = value
            elapsed = time.time() - self.start_time
            self._log(elapsed, value, round_num)
            
            msg = f"      [Monitor] New Best: {value:,.2f} @ {elapsed:.2f}s"
            if round_num is not None:
                msg += f" | Round: {round_num}"
            print(msg)
            
            self.save_to_csv()
            self.last_log_time = time.time()
            return True
        return False

    def tick(self):
        """
        [Time-Driven] Logs periodically to ensure continuous data.
        """
        now = time.time()
        if now - self.last_log_time >= self.log_interval:
            elapsed = now - self.start_time
            self._log(elapsed, self.current_best_value)
            self.save_to_csv()
            self.last_log_time = now

    def finalize(self, total_runtime):
        """Finalizes the log at the end of execution."""
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
    Loads graph topology and externality data.
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
    Saves the final optimization result.
    """
    os.makedirs(output_dir, exist_ok=True)
    pd.DataFrame({'id': sorted(list(selected_nodes))}).to_csv(os.path.join(output_dir, 'sub_nodes.csv'), index=False)
    pd.DataFrame(selected_edges, columns=['source', 'target', 'cost']).to_csv(os.path.join(output_dir, 'sub_edges.csv'), index=False)
    with open(os.path.join(output_dir, 'score.txt'), 'w') as f: f.write(f"{score:.10f}")
    with open(os.path.join(output_dir, 'time.txt'), 'w') as f: f.write(f"{duration:.6f}")
    pd.DataFrame({'id': sorted(list(result_nodes))}).to_csv(os.path.join(output_dir, 'result_nodes.csv'), index=False)
    with open(os.path.join(output_dir, 'result_nodes_count.txt'), 'w') as f: f.write(str(len(result_nodes)))

# ==============================================================================
# 2. Incremental Calculator (For Rounding Evaluation)
# ==============================================================================
class IncrementalScoreCalculator:
    """
    Helper class to quickly calculate the objective score for a discrete set of nodes.
    Used extensively during the rounding phase.
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
        """Rebuilds internal externality maps."""
        self.incoming_ext = defaultdict(list)
        self.outgoing_ext = defaultdict(list)
        for (u, v), p in externality_dict.items():
            self.incoming_ext[v].append((u, p))
            self.outgoing_ext[u].append((v, p))

    def full_recalc(self, nodes):
        """
        Calculates score from scratch for a given set of result nodes.
        Includes Value, Ancestor Costs, and Externality Penalties.
        """
        self.current_set = set(nodes)
        self.ref_counts = defaultdict(int)
        self.current_penalties = defaultdict(float)
        
        # 1. Ancestor Costs (Closure)
        cost_sum = 0.0
        for n in self.current_set:
            for a in self.ancestor_cache[n]:
                if self.ref_counts[a] == 0:
                    cost_sum += self.node_costs[a]
                self.ref_counts[a] += 1
        
        # 2. Externality Penalties
        for u in self.current_set:
            for v, p in self.outgoing_ext[u]:
                if v in self.current_set:
                    self.current_penalties[v] += p
        
        # 3. Node Values (Net)
        val_sum = 0.0
        for n in self.current_set:
            net_val = max(0.0, self.node_vals[n] - self.current_penalties[n])
            val_sum += net_val
            
        self.current_score = val_sum - cost_sum

# ==============================================================================
# 3. QPR Solver Class
# ==============================================================================
class QPRelaxationSolver:
    """
    Implements the QPR algorithm:
    1. Relaxation: Maps the problem to a Quadratic Program.
    2. Rounding: Heuristic rounding of the continuous solution.
    """
    def __init__(self, G, externality_dict, node_id_to_idx, idx_to_node_id):
        self.G = G
        self.ext = externality_dict
        self.n2i = node_id_to_idx
        self.i2n = idx_to_node_id
        self.N = len(G.nodes)

    def solve_relaxation(self, time_limit):
        """
        Constructs and solves the QP relaxation using OSQP.
        Minimize: (1/2)x'Px + q'x
        Subject to: l <= Ax <= u
        """
        print(f"    [QP] Building Matrices & Solving (Budget={time_limit:.1f}s)...")
        
        if time_limit <= 0:
            return np.zeros(self.N)

        # --- Matrix Construction ---
        # P Matrix: Represents quadratic penalty terms (externalities).
        # We shift the diagonal to ensure positive semi-definiteness if required.
        PENALTY_MULTIPLIER = 1.0
        P_dict = defaultdict(float)
        diagonal_shift = defaultdict(float) 
        
        for (u, v), penalty in self.ext.items():
            if u in self.n2i and v in self.n2i:
                iu, iv = self.n2i[u], self.n2i[v]
                scaled_p = penalty * PENALTY_MULTIPLIER
                if iu != iv:
                    P_dict[(iu, iv)] += scaled_p
                    P_dict[(iv, iu)] += scaled_p
                    diagonal_shift[iu] += abs(scaled_p)
                    diagonal_shift[iv] += abs(scaled_p)
                else:
                    P_dict[(iu, iv)] += scaled_p

        rows_p, cols_p, data_p = [], [], []
        for (r, c), val in P_dict.items():
            rows_p.append(r); cols_p.append(c); data_p.append(val)
        
        for i in range(self.N):
            rows_p.append(i); cols_p.append(i); data_p.append(diagonal_shift[i] + 1e-5) 

        P = csr_matrix((data_p, (rows_p, cols_p)), shape=(self.N, self.N))

        # q Vector: Linear costs and values.
        q = np.zeros(self.N)
        for n, data in self.G.nodes(data=True):
            i = self.n2i[n]
            q[i] = data['cost']
            if data.get('is_result'): q[i] -= data['value']
            if i in diagonal_shift: q[i] -= 0.5 * diagonal_shift[i]

        # A Matrix: Constraints
        # 1. Topological constraints (Edge u->v implies x_v <= x_u)
        edge_rows, edge_cols, edge_data = [], [], []
        for idx, (u, v) in enumerate(self.G.edges):
            iu, iv = self.n2i[u], self.n2i[v]
            # Constraint: x_v - x_u <= 0
            edge_rows.extend([idx, idx]); edge_cols.extend([iv, iu]); edge_data.extend([1.0, -1.0])
            
        num_edges = len(self.G.edges)
        if num_edges > 0:
            A_tree = csr_matrix((edge_data, (edge_rows, edge_cols)), shape=(num_edges, self.N))
            l_tree = np.full(num_edges, -np.inf)
            u_tree = np.zeros(num_edges)
        else:
            A_tree = csr_matrix((0, self.N)); l_tree = np.array([]); u_tree = np.array([])

        # 2. Variable bounds (0 <= x <= 1)
        A_bound = identity(self.N, format='csr')
        l = np.concatenate([l_tree, np.zeros(self.N)])
        u = np.concatenate([u_tree, np.ones(self.N)])
        A = vstack([A_tree, A_bound], format='csc')

        # --- OSQP Solver Configuration ---
        prob = osqp.OSQP()
        
        # Configuration Explanation:
        # eps_abs=1e-3, eps_rel=1e-3: Relaxed tolerances for faster, approximate solutions.
        # max_iter=2000: Limited iterations to prevent hanging on large instances.
        # polish=False: Disabled polishing to save time, as we only need an approximate gradient.
        prob.setup(P, q, A, l, u, alpha=1.0, sigma=1e-6, verbose=False, 
                   eps_abs=1e-3, eps_rel=1e-3, max_iter=2000, polish=False,
                   time_limit=time_limit) 
        
        try:
            res = prob.solve()
            if res.info.status_val in [-3, -4, -10]:
                 print(f"    [QP] OSQP Error: {res.info.status}")
                 return np.zeros(self.N)
        except ValueError:
            return np.zeros(self.N)
            
        return np.clip(res.x, 0.0, 1.0)

    def smart_rounding(self, x_cont, ancestor_cache, all_result_nodes, tracker, deadline):
        """
        Rounding Phase:
        Iterates through various threshold levels to convert the continuous solution 'x_cont'
        into a valid discrete selection, picking the one that yields the best objective score.
        """
        if time.time() >= deadline:
            print("    [Rounding] Skipped due to time limit.")
            return set(), -float('inf'), set()

        print(f"    [Rounding] Starting Threshold Scan (Time remaining: {deadline - time.time():.1f}s)...")
        
        # Define Thresholds: Fixed points + Data-driven percentiles
        values = [x_cont[self.n2i[n]] for n in all_result_nodes]
        thresholds = [0.2, 0.4, 0.5, 0.6, 0.8]
        if len(values) > 5:
            thresholds.extend(np.percentile(values, [30, 70, 90]).tolist())
        thresholds = sorted(list(set(t for t in thresholds if 0.01 < t < 0.99)), reverse=True)

        inc_calc = IncrementalScoreCalculator(self.G, self.ext, ancestor_cache)
        inc_calc._hard_reset_logic(self.ext)
        
        best_score = -float('inf')
        best_set = set()

        for idx, t in enumerate(thresholds):
            if time.time() > deadline: break
            
            # Select nodes exceeding the threshold
            S = {n for n in all_result_nodes if x_cont[self.n2i[n]] >= t}
            if not S: continue
            
            # Evaluate
            inc_calc.full_recalc(S)
            score = inc_calc.current_score
            
            if tracker.update(score, round_num=f"Threshold_{t:.2f}"):
                best_score = score
                best_set = S.copy()

        if not best_set: best_set = set()
        
        print(f"    [Rounding] Finished. Best Score found: {best_score}")

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
    algorithm_name: str = "qp",
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
    solver = QPRelaxationSolver(G, externality_dict, node_id_to_idx, idx_to_node_id)
    
    # --- Phase 1: QP Relaxation ---
    remaining = deadline - time.time()
    if remaining > 1.0:
        x_cont = solver.solve_relaxation(time_limit=remaining)
    else:
        x_cont = np.zeros(len(G.nodes))

    # --- Phase 2: Smart Rounding ---
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
    ALGO_NAME = "qp" # Note: Internal logic implements QPR
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str, default="../data/output")
    args = parser.parse_args()
    
    run_experiment(root=args.root_dir, algo_name=ALGO_NAME, max_runtime=MAX_RUNTIME)