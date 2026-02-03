import os
import time
import pandas as pd
import networkx as nx
from collections import defaultdict
import argparse
from typing import List, Dict, Set, Tuple

# ==============================================================================
# --- 0. Utility Classes ---
# ==============================================================================
class ConvergenceTracker:
    """
    Tracks the optimization progress, logging the best solution found over time
    to both the console and a CSV file.
    """
    def __init__(self, algorithm_name, dataset_name, save_path="convergence.csv", log_interval=1.0):
        self.algorithm_name = algorithm_name
        self.dataset_name = dataset_name
        self.save_path = save_path
        self.log_interval = log_interval
        
        self.start_time = time.time()
        self.last_log_time = self.start_time
        
        self.history = []
        self.current_best_value = -float('inf')
        
        # Log initial state
        self._log(0.0, -float('inf'))

    def update(self, value, round_num=None):
        """
        Updates the best solution if the new value is better.
        Returns True if updated, False otherwise.
        """
        # Use a small epsilon for float comparison
        if value > self.current_best_value + 1e-9:
            self.current_best_value = value
            elapsed = time.time() - self.start_time
            
            # Record data
            self._log(elapsed, value, round_num)
            
            # Print status to console
            msg = f"      [{self.algorithm_name.upper()}] Best: {value:,.2f} @ {elapsed:.2f}s"
            if round_num is not None:
                msg += f" | Step: {round_num}"
            print(msg)
            
            self.save_to_csv()
            self.last_log_time = time.time()
            return True
        return False

    def finalize(self, total_runtime):
        """Ensures the final state is recorded at the end of execution."""
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
    Loads nodes, edges, and externality constraints from CSV files into a NetworkX graph.
    """
    node_df = pd.read_csv(os.path.join(folder, "nodes.csv"))
    edge_df = pd.read_csv(os.path.join(folder, "edges.csv"))
    G = nx.DiGraph()
    node_costs = {}
    
    # Load Nodes
    for _, row in node_df.iterrows():
        nid = int(row["id"])
        cost = _safe_float(row.get("incoming_cost", row.get("cost", 0.0)))
        val = _safe_float(row.get("value", 0.0))
        is_res = str(row.get("is_result", False)).lower() == 'true'
        G.add_node(nid, value=val, is_result=is_res, cost=cost)
        node_costs[nid] = cost

    # Load Edges
    edges_data = []
    for _, r in edge_df.iterrows():
        u, v = int(r["source"]), int(r["target"])
        w = _safe_float(r.get("cost", 0.0)) 
        edges_data.append((u, v, w))
    G.add_weighted_edges_from(edges_data, weight="cost")
    
    # Load Externalities (Conflict/Penalty)
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
    """Saves the optimization results to the specified directory."""
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
# --- 2. Incremental State Manager ---
# ==============================================================================
class CachedGreedyState:
    """
    Manages the state of the graph solution efficiently.
    It supports incremental updates to the objective score and the "gain" (marginal benefit)
    of removing specific nodes.
    """
    def __init__(self, G, node_costs, externality_dict, all_result_nodes):
        self.node_costs = node_costs
        self.node_vals = {n: d.get('value', 0.0) for n, d in G.nodes(data=True)}
        
        # Adjacency list for externalities: ext_adj[u] = [(v, penalty), ...]
        self.ext_adj = defaultdict(list)
        for (u, v), p in externality_dict.items():
            self.ext_adj[u].append((v, p))
            self.ext_adj[v].append((u, p))
            
        self.ancestor_to_users = defaultdict(set)
        self.res_to_ancestors = {}
        
        # Precompute ancestors and usage maps
        for r in all_result_nodes:
            ancs = nx.ancestors(G, r)
            ancs.add(r)
            self.res_to_ancestors[r] = list(ancs)
            for anc in ancs:
                self.ancestor_to_users[anc].add(r)
                
        self.active_results = set(all_result_nodes)
        self.current_score = self._calculate_initial_total_score(externality_dict)
        
        # Initialize Gains for all active nodes
        self.current_gains = {}
        for r in self.active_results:
            self.current_gains[r] = self._calculate_single_gain(r)
            
    def _calculate_initial_total_score(self, externality_dict):
        """Calculates the objective function value from scratch."""
        val_sum = sum(self.node_vals[r] for r in self.active_results)
        cost_sum = sum(self.node_costs[anc] for anc, users in self.ancestor_to_users.items() if len(users) > 0)
        ext_sum = 0.0
        for (u, v), p in externality_dict.items():
            if u in self.active_results and v in self.active_results:
                ext_sum += p
        return val_sum - cost_sum - ext_sum

    def _calculate_single_gain(self, node):
        """
        Calculates the marginal gain of removing a specific node.
        Positive gain implies that removing the node improves the total score.
        Formula: Gain = Saved Penalty + Saved Shared Costs - Lost Value
        """
        gain = -self.node_vals[node]
        
        # 1. Externality Gain: If neighbors exist, removing 'node' saves the penalty.
        for peer, pen in self.ext_adj[node]:
            if peer in self.active_results and peer != node:
                gain += pen
        
        # 2. Cost Gain: If 'node' is the *only* user of an ancestor, removing 'node' saves that cost.
        for anc in self.res_to_ancestors[node]:
            if len(self.ancestor_to_users[anc]) == 1:
                gain += self.node_costs[anc]
        return gain

    def remove_node_incremental(self, node):
        """
        Removes a node from the active set and updates the gains of related nodes 
        (neighbors and siblings sharing ancestors) incrementally.
        """
        self.active_results.remove(node)
        gain_val = self.current_gains.pop(node)
        self.current_score += gain_val 
        
        # A. Update neighbors sharing externalities
        # Since 'node' is removed, its neighbors no longer save this penalty if they were to be removed.
        for peer, pen in self.ext_adj[node]:
            if peer in self.active_results:
                self.current_gains[peer] -= pen

        # B. Update siblings sharing ancestors
        # If 'node' removal leaves an ancestor with exactly one user, that user now bears the full weight.
        # Thus, the gain of removing that survivor increases (it now saves that cost).
        for anc in self.res_to_ancestors[node]:
            users = self.ancestor_to_users[anc]
            users.remove(node)
            if len(users) == 1:
                survivor = next(iter(users))
                if survivor in self.current_gains:
                    self.current_gains[survivor] += self.node_costs[anc]

    def get_full_closure_nodes(self):
        """Returns the set of all required nodes (results + dependencies)."""
        return {anc for anc, users in self.ancestor_to_users.items() if len(users) > 0}

# ==============================================================================
# --- 3. Solver Logic ---
# ==============================================================================
def run_solver(
    G: nx.DiGraph, 
    node_costs: Dict, 
    edges_data: List, 
    externality_dict: Dict, 
    dataset_name: str,
    result_dir: str,
    algorithm_name: str = "greedy",
    max_runtime: float = 300.0,
    **kwargs 
):
    os.makedirs(result_dir, exist_ok=True)
    
    # Logging interval
    log_interval = 2.0
    tracker = ConvergenceTracker(
        algorithm_name, 
        dataset_name, 
        save_path=os.path.join(result_dir, "convergence.csv"),
        log_interval=log_interval 
    )
    
    start_time = time.time()
    all_res = [n for n, d in G.nodes(data=True) if d.get('is_result')]
    
    if not all_res: 
        tracker.finalize(max_runtime)
        return set(), [], 0.0, set()
    
    print(f"    [{algorithm_name.upper()}] Initializing state with {len(all_res)} nodes...")
    
    # Initialize State Manager
    state = CachedGreedyState(G, node_costs, externality_dict, all_res)
    
    # Record baseline score (full graph)
    tracker.update(state.current_score, round_num=0)
    
    next_log_time = time.time() + log_interval
    
    # --- Greedy Strategy Main Logic ---
    
    # 1. Snapshot: Calculate initial gains for all active nodes.
    initial_candidates = []
    for r in state.active_results:
        gain = state.current_gains[r]
        initial_candidates.append((r, gain))
    
    # 2. Sort candidates based on initial gain (Descending).
    #    Nodes with higher gains (more beneficial to remove) are processed first.
    initial_candidates.sort(key=lambda x: x[1], reverse=True)
    
    print(f"    [{algorithm_name.upper()}] Sorted {len(initial_candidates)} candidates by initial gain.")
    
    step = 0
    # 3. Iterate through the sorted list (One-pass).
    for node_to_check, initial_gain in initial_candidates:
        step += 1
        
        # Safe Check: Even though the order is determined statically, 
        # we check the *current* real-time gain before removal.
        # This ensures we don't remove a node if previous removals made it valuable.
        current_real_gain = state.current_gains.get(node_to_check, -float('inf'))
        
        # Only remove if it strictly improves the score
        if current_real_gain > 0:
            state.remove_node_incremental(node_to_check)
        
        # Logging & Timeout Check
        now = time.time()
        if now >= next_log_time:
            tracker.update(state.current_score, round_num=step)
            next_log_time = now + log_interval
        
        if now - start_time > max_runtime:
            tracker.update(state.current_score, round_num=step)
            print(f"    [{algorithm_name.upper()}] Maximum runtime limit reached.")
            break

    # Force a final update to ensure the last state is recorded
    tracker.update(state.current_score, round_num=step)

    tracker.finalize(time.time() - start_time)
    
    # Construct Final Result
    final_result_nodes = state.active_results
    final_nodes = state.get_full_closure_nodes()
    final_edges = [(u, v, c) for u, v, c in edges_data if u in final_nodes and v in final_nodes]
            
    return final_nodes, final_edges, state.current_score, final_result_nodes

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
    MAX_RUNTIME = 100.0
    ALGO_NAME = "greedy" 
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str, default="../data/output")
    args = parser.parse_args()
    
    run_experiment(args.root_dir, ALGO_NAME, MAX_RUNTIME)