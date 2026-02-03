import os
import time
import pandas as pd
import networkx as nx
import random
import math
from collections import defaultdict
import argparse
from typing import List, Dict, Set, Tuple, Optional

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
        [Time-Driven] Logs periodically to ensure continuous data points.
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

def _safe_float(x):
    try: return float(x)
    except: return 0.0

# ==============================================================================
# --- 1. Data Loading ---
# ==============================================================================
def load_all_data(folder: str):
    """
    Loads nodes, edges, and externality constraints from CSV files.
    """
    node_df = pd.read_csv(os.path.join(folder, "nodes.csv"))
    edge_df = pd.read_csv(os.path.join(folder, "edges.csv"))
    G = nx.DiGraph()
    node_costs = {}
    
    for _, row in node_df.iterrows():
        nid = int(row["id"])
        cost = _safe_float(row.get("incoming_cost", row.get("cost", 0.0)))
        val = _safe_float(row.get("value", 0.0))
        is_res = str(row.get("is_result", False)).lower() == 'true'
        G.add_node(nid, value=val, is_result=is_res, cost=cost)
        node_costs[nid] = cost

    edges_data = []
    for _, r in edge_df.iterrows():
        u, v = int(r["source"]), int(r["target"])
        w = _safe_float(r.get("cost", 0.0)) 
        edges_data.append((u, v, w))
    G.add_weighted_edges_from(edges_data, weight="cost")
    
    externality_dict = {}
    p_names = ["externality.csv", "externality_matrix.csv"]
    f_path = next((os.path.join(folder, n) for n in p_names if os.path.exists(os.path.join(folder, n))), None)
    if f_path:
        df = pd.read_csv(f_path)
        for _, r in df.iterrows():
            try:
                s, t = int(r["source"]), int(r["target"])
                p = float(r.get("penalty", r.get("weight", 0.0)))
                if p > 0: 
                    externality_dict[(s,t)] = p
            except: pass
    return G, node_costs, edges_data, externality_dict

def save_result(output_dir, selected_nodes, selected_edges, score, duration, res_nodes):
    """Saves the optimization results."""
    os.makedirs(output_dir, exist_ok=True)
    pd.DataFrame({'id': sorted(list(selected_nodes))}).to_csv(os.path.join(output_dir, 'sub_nodes.csv'), index=False)
    
    if selected_edges:
        sorted_edges = sorted(list(selected_edges), key=lambda x: (x[0], x[1]))
        pd.DataFrame(sorted_edges, columns=['source', 'target', 'cost']).to_csv(os.path.join(output_dir, 'sub_edges.csv'), index=False)
    else:
        pd.DataFrame(columns=['source', 'target', 'cost']).to_csv(os.path.join(output_dir, 'sub_edges.csv'), index=False)
        
    with open(os.path.join(output_dir, 'score.txt'), 'w') as f: f.write(str(score))
    with open(os.path.join(output_dir, 'time.txt'), 'w') as f: f.write(f"{duration:.6f}")
    
    if not isinstance(res_nodes, set): res_nodes = set(res_nodes)
    pd.DataFrame({'id': sorted(list(res_nodes))}).to_csv(os.path.join(output_dir, 'result_nodes.csv'), index=False)
    with open(os.path.join(output_dir, 'result_nodes_count.txt'), 'w') as f: f.write(str(len(res_nodes)))

# ==============================================================================
# --- 2. Incremental Score State (with SA extensions) ---
# ==============================================================================
class IncrementalScoreState:
    """
    Manages the objective score incrementally.
    Supports 'flip' and 'rollback' operations specifically for Simulated Annealing.
    """
    def __init__(self, G, externality_dict, ancestor_cache):
        self.ancestor_cache = ancestor_cache
        self.ext_incoming = defaultdict(list) 
        self.ext_outgoing = defaultdict(list) 
        
        # Parse externalities (Strictly Directed)
        if externality_dict:
            for (u, v), pen in externality_dict.items():
                self.ext_outgoing[u].append((v, pen))
                self.ext_incoming[v].append((u, pen))
        
        self.current_result_set = set()       
        self.ref_counts = defaultdict(int)   
        self.current_score = 0.0             
        self.current_penalties = defaultdict(float) 
        
        self.node_vals = {n: d.get('value', 0.0) for n, d in G.nodes(data=True)}
        self.node_costs = {n: d.get('cost', 0.0) for n, d in G.nodes(data=True)}

    def _calc_effective_value(self, node):
        val = self.node_vals[node]
        pen = self.current_penalties[node]
        return val - pen

    def initialize(self, result_nodes: Set[int]):
        """Resets and batch initializes the state."""
        self.current_result_set = set()
        self.ref_counts = defaultdict(int)
        self.current_score = 0.0
        self.current_penalties = defaultdict(float)
        for node in result_nodes: self.add_node(node)

    def add_node(self, node: int):
        if node in self.current_result_set: return
        
        # 1. Incoming Penalty: This node suffers from existing peers
        initial_pen = 0.0
        for peer, pen in self.ext_incoming[node]:
            if peer in self.current_result_set: initial_pen += pen
        self.current_penalties[node] = initial_pen
        
        self.current_score += self._calc_effective_value(node)
        
        # 2. Outgoing Penalty: This node penalizes existing peers
        for target, pen in self.ext_outgoing[node]:
            if target in self.current_result_set:
                old_val = self._calc_effective_value(target)
                self.current_score -= old_val
                self.current_penalties[target] += pen 
                new_val = self._calc_effective_value(target)
                self.current_score += new_val
        
        # 3. Ancestor Costs
        for anc in self.ancestor_cache[node]:
            if self.ref_counts[anc] == 0: self.current_score -= self.node_costs[anc]
            self.ref_counts[anc] += 1
        self.current_result_set.add(node)

    def remove_node(self, node: int):
        if node not in self.current_result_set: return
        
        # 1. Remove node value
        self.current_score -= self._calc_effective_value(node)
        self.current_penalties[node] = 0.0 
        
        # 2. Relieve outgoing penalties
        for target, pen in self.ext_outgoing[node]:
            if target in self.current_result_set:
                old_val = self._calc_effective_value(target)
                self.current_score -= old_val
                self.current_penalties[target] -= pen 
                new_val = self._calc_effective_value(target)
                self.current_score += new_val
        
        # 3. Update Ancestor Costs
        for anc in self.ancestor_cache[node]:
            self.ref_counts[anc] -= 1
            if self.ref_counts[anc] == 0: self.current_score += self.node_costs[anc]
        self.current_result_set.remove(node)

    # --- SA Specific Interfaces ---
    def flip(self, node: int) -> float:
        """
        Toggles the node's presence in the solution.
        Returns the delta (change) in the total score.
        """
        old_score = self.current_score
        if node in self.current_result_set:
            self.remove_node(node)
        else:
            self.add_node(node)
        return self.current_score - old_score

    def rollback(self, node: int):
        """
        Reverts the last flip operation.
        """
        self.flip(node)

    def get_full_closure_nodes(self) -> Set[int]:
        return {n for n, c in self.ref_counts.items() if c > 0}

# ==============================================================================
# --- 3. Simulated Annealing Solver ---
# ==============================================================================
def run_solver(
    G: nx.DiGraph, 
    node_costs: Dict, 
    edges_data: List, 
    externality_dict: Dict, 
    dataset_name: str,
    result_dir: str,
    algorithm_name: str = "sa",
    max_runtime: float = 2000.0,
    initial_accept_prob: float = 0.80,
    sampling_iterations: int = 200
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
    
    all_res = [n for n, d in G.nodes(data=True) if d.get('is_result')]
    if not all_res: 
        tracker.finalize(max_runtime)
        return set(), [], 0.0, set()
    
    ancestor_cache = {r: set(nx.ancestors(G, r)) | {r} for r in all_res}
    problem_size_N = len(all_res)
    
    global_best_score = -float('inf')
    global_best_sol = set()
    
    # Reusable state manager
    state_manager = IncrementalScoreState(G, externality_dict, ancestor_cache)
    
    cycle_idx = 0
    print(f"    [{algorithm_name.upper()}] Start Random Restart SA... Budget: {max_runtime}s")

    while time.time() < deadline:
        cycle_idx += 1
        
        # 1. Random Restart: Initialize with a random subset
        ratio = random.random()
        k = max(1, int(len(all_res) * ratio))
        init_set = set(random.sample(all_res, k))
        
        # 2. Initialize State
        state_manager.initialize(init_set)
        curr_score = state_manager.current_score
        
        local_best = curr_score
        local_best_sol = state_manager.current_result_set.copy()
        
        # Check if the random start is a record
        if tracker.update(local_best, round_num=cycle_idx):
            global_best_score = local_best
            global_best_sol = local_best_sol.copy()
        
        # 3. Calculate Initial Temperature (Warm-up phase)
        # We sample random moves to estimate the scale of "bad" deltas.
        bad_deltas = []
        for _ in range(min(sampling_iterations, 100)):
            if time.time() > deadline: break
            node = random.choice(all_res)
            d = state_manager.flip(node)
            state_manager.rollback(node)
            if d < 0: bad_deltas.append(d)
        
        avg_bad = sum(bad_deltas)/len(bad_deltas) if bad_deltas else -1.0
        # Determine T s.t. exp(avg_bad / T) ~= initial_accept_prob
        T = max(abs(avg_bad) / math.log(initial_accept_prob) if avg_bad != 0 else 1.0, 10.0)
        
        T_min = 1e-3
        alpha = 0.95
        iter_per_temp = int(max(1, problem_size_N * 0.2))
        stagnation = 0
        
        # --- Annealing Loop ---
        while T > T_min and time.time() < deadline:
            improved_at_this_temp = False
            tracker.tick()
            
            for i in range(iter_per_temp):
                if i % 100 == 0:
                    tracker.tick()
                    if time.time() > deadline: break
                
                node = random.choice(all_res)
                # Flip and get delta
                delta = state_manager.flip(node)
                
                accept = False
                if delta > 0: accept = True
                else:
                    try: p = math.exp(delta/T)
                    except: p = 0.0
                    if random.random() < p: accept = True
                
                if accept:
                    # state_manager is already updated via flip()
                    curr_score += delta 
                    
                    if curr_score > local_best + 1e-9:
                        local_best = curr_score
                        local_best_sol = state_manager.current_result_set.copy()
                        improved_at_this_temp = True
                else:
                    # Reject: Rollback
                    state_manager.rollback(node)
            
            if not improved_at_this_temp: stagnation += 1
            else: stagnation = 0
            
            if stagnation > 20: break 
            T *= alpha
            
        # Check local best against global best
        if tracker.update(local_best, round_num=cycle_idx):
            global_best_score = local_best
            global_best_sol = local_best_sol.copy()

    tracker.finalize(max_runtime)
    
    # Construct Final Result
    state_manager.initialize(global_best_sol)
    final_nodes = state_manager.get_full_closure_nodes()
    final_edges = [(u, v, c) for u, v, c in edges_data if u in final_nodes and v in final_nodes]
            
    return final_nodes, final_edges, global_best_score, global_best_sol

# ==============================================================================
# --- Execution Entry Point ---
# ==============================================================================
def run_experiment(root_dir, algo_name, max_runtime):
    if not os.path.exists(root_dir): return
    
    for folder in os.listdir(root_dir):
        path = os.path.join(root_dir, folder)
        if not os.path.isdir(path): continue
        if "dag" not in folder: continue 

        try:
            print(f"--- Processing [{folder}] (Algo: {algo_name}) ---")
            
            G, nc, ed, ext = load_all_data(path)
            
            result_dir = os.path.join(path, f"{algo_name}_result")
            start_real = time.time()
            
            sn, se, score, res_nodes = run_solver(
                G, nc, ed, ext,
                dataset_name=folder, 
                result_dir=result_dir,
                algorithm_name=algo_name,
                max_runtime=max_runtime
            )
            
            duration = time.time() - start_real
            save_result(result_dir, sn, se, score, duration, res_nodes)
            print(f"[{folder}] Final: {score:,.2f}, Time: {duration:.2f}s")
            print("="*60)
            
        except Exception as e:
            import traceback; traceback.print_exc()

if __name__ == "__main__":
    MAX_RUNTIME = 1000.0
    ALGO_NAME = "sa"
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str, default="../data/output")
    args = parser.parse_args()
    
    run_experiment(args.root_dir, ALGO_NAME, MAX_RUNTIME)