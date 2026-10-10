import os
import time
import random
import math
import pandas as pd
import networkx as nx
from collections import defaultdict
import argparse
from typing import List, Dict, Set, Tuple, Optional

# ==============================================================================
# --- 0. Enhanced tracker class (event-driven & time-driven) ---
# ==============================================================================
class ConvergenceTracker:
    def __init__(self, algorithm_name, dataset_name, save_path="convergence.csv", log_interval=50.0):
        self.algorithm_name = algorithm_name
        self.dataset_name = dataset_name
        self.save_path = save_path
        self.log_interval = log_interval  # Monitoring interval
        
        self.start_time = time.time()
        self.last_log_time = self.start_time
        
        self.history = []
        self.current_best_value = -float('inf')
        
        # Initial record
        self._log(0.0, -float('inf'))

    def update(self, value, round_num=None):
        """
        [Event-Driven]
        Call only when a better solution is found.
        Return True if a new best solution is found, otherwise False.
        """
        if value > self.current_best_value + 1e-9:
            self.current_best_value = value
            elapsed = time.time() - self.start_time
            
            # Record data, including the iteration number.
            self._log(elapsed, value, round_num)
            
            # Build the output message.
            msg = f"      [Monitor] New Best: {value:,.2f} @ {elapsed:.2f}s"
            if round_num is not None:
                msg += f" | Round: {round_num}"
            print(msg)
            
            self.save_to_csv()
            self.last_log_time = time.time() # Reset the timer.
            return True
        return False

    def tick(self):
        """
        [Time-Driven]
        Periodically record data using forward fill, without printing a New Best message.
        """
        now = time.time()
        if now - self.last_log_time >= self.log_interval:
            elapsed = now - self.start_time
            self._log(elapsed, self.current_best_value)
            self.save_to_csv()
            self.last_log_time = now

    def finalize(self, total_runtime):
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
# --- 1. Data loading and result saving ---
# ==============================================================================
def load_all_data(folder: str):
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
# --- 2. Incremental state manager ---
# ==============================================================================
class IncrementalScoreState:
    def __init__(self, G, externality_dict, ancestor_cache):
        self.ancestor_cache = ancestor_cache
        self.ext_incoming = defaultdict(list) 
        self.ext_outgoing = defaultdict(list) 
        
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
        self.current_result_set = set()
        self.ref_counts = defaultdict(int)
        self.current_score = 0.0
        self.current_penalties = defaultdict(float)
        for node in result_nodes: self.add_node(node)

    def add_node(self, node: int):
        if node in self.current_result_set: return
        initial_pen = 0.0
        for peer, pen in self.ext_incoming[node]:
            if peer in self.current_result_set: initial_pen += pen
        self.current_penalties[node] = initial_pen
        self.current_score += self._calc_effective_value(node)
        
        for target, pen in self.ext_outgoing[node]:
            if target in self.current_result_set:
                old_val = self._calc_effective_value(target)
                self.current_score -= old_val
                self.current_penalties[target] += pen 
                new_val = self._calc_effective_value(target)
                self.current_score += new_val
        
        for anc in self.ancestor_cache[node]:
            if self.ref_counts[anc] == 0: self.current_score -= self.node_costs[anc]
            self.ref_counts[anc] += 1
        self.current_result_set.add(node)

    def remove_node(self, node: int):
        if node not in self.current_result_set: return
        self.current_score -= self._calc_effective_value(node)
        self.current_penalties[node] = 0.0 
        
        for target, pen in self.ext_outgoing[node]:
            if target in self.current_result_set:
                old_val = self._calc_effective_value(target)
                self.current_score -= old_val
                self.current_penalties[target] -= pen 
                new_val = self._calc_effective_value(target)
                self.current_score += new_val
        
        for anc in self.ancestor_cache[node]:
            self.ref_counts[anc] -= 1
            if self.ref_counts[anc] == 0: self.current_score += self.node_costs[anc]
        self.current_result_set.remove(node)
    
    def get_full_closure_nodes(self) -> Set[int]:
        return {n for n, c in self.ref_counts.items() if c > 0}

# ==============================================================================
# --- 3. Heuristic computation: individual net value (ablation) ---
# ==============================================================================
def calculate_single_net_values(all_res, ancestor_cache, node_values, node_costs, externality_dict):
    """
    Ablation computation:
    Single Net Value = Node Value - Parent Closure Cost - (Total Externality / 2)
    Deduct the full parent-closure cost, assuming the node is selected alone with no cost sharing.
    """
    # 1. Precompute each node's externality burden (half its total conflict penalty).
    # For conflicting nodes u and v with penalty P, allocate P/2 to each node.
    ext_burden = defaultdict(float)
    if externality_dict:
        for (src, tgt), penalty in externality_dict.items():
            half_p = penalty / 2.0
            ext_burden[src] += half_p
            ext_burden[tgt] += half_p

    net_values = {}
    for node in all_res:
        # A. Base value
        base_val = node_values.get(node, 0.0)
        
        # B. Full parent-closure cost
        # Sum the costs of all ancestors, including the node itself.
        closure_cost = sum(node_costs.get(anc, 0.0) for anc in ancestor_cache[node])
        
        # C. Half the externality penalty
        ext_penalty = ext_burden[node]
        
        # Compute the net value.
        net_values[node] = base_val - closure_cost - ext_penalty
        
    return net_values

def perturb_solution(base_solution: Set[int], all_candidates: List[int], perturb_strength: float = 0.05) -> Set[int]:
    new_set = set(base_solution)
    num_to_flip = max(1, int(len(all_candidates) * perturb_strength))
    targets = random.sample(all_candidates, num_to_flip)
    for node in targets:
        if node in new_set: new_set.remove(node)
        else: new_set.add(node)
    return new_set

# ==============================================================================
# --- 4. Hill-climbing logic ---
# ==============================================================================
def _greedy_descent_pass_incremental(
    state: IncrementalScoreState, 
    all_result_nodes: List[int], 
    tracker: ConvergenceTracker,
    deadline: float
):
    """
    Perform a single hill-climbing pass.
    """
    improvement_found = True
    candidates = list(all_result_nodes)
    random.shuffle(candidates) 
    
    local_best_score = state.current_score
    steps = 0
    
    while improvement_found:
        if time.time() > deadline: break
        
        improvement_found = False
        for i, node in enumerate(candidates):
            # Every 100 nodes, check whether a time-driven log entry is due.
            if i % 100 == 0:
                tracker.tick()
                if time.time() > deadline: break
            
            is_in = node in state.current_result_set
            
            if is_in: state.remove_node(node)
            else: state.add_node(node)
            
            new_score = state.current_score
            
            # Greedy Accept
            if new_score > local_best_score + 1e-9:
                local_best_score = new_score
                improvement_found = True
                steps += 1
            else:
                # Rollback
                if is_in: state.add_node(node)
                else: state.remove_node(node)
    
    return local_best_score, steps

# ==============================================================================
# --- 5. Unified solver interface (ablation logic) ---
# ==============================================================================
def run_solver(
    G: nx.DiGraph, 
    node_costs: Dict, 
    edges_data: List, 
    externality_dict: Dict, 
    dataset_name: str,
    result_dir: str,
    algorithm_name: str = "ablation_net_val",
    max_runtime: float = 2000.0
):
    os.makedirs(result_dir, exist_ok=True)
    
    # 1. Initialize the tracker.
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
    
    # Precomputation
    node_values = {n: G.nodes[n]['value'] for n in all_res}
    ancestor_cache = {r: set(nx.ancestors(G, r)) | {r} for r in all_res}
    
    # === A: Compute individual net values instead of Shapley estimates ===
    print(f"    [{algorithm_name.upper()}] Computing Single Net Values (Ablation)...")
    heuristic_values = calculate_single_net_values(
        all_res, ancestor_cache, node_values, node_costs, externality_dict
    )
    
    # Sort candidates.
    sorted_candidates = sorted(all_res, key=lambda n: heuristic_values.get(n, -float('inf')), reverse=True)
    n_candidates = len(sorted_candidates)
    
    # === B: Construct stratified proportions without a separate search over positive values ===
    # Use only proportions from 10% to 50%.
    ratios_to_try = [i / 100.0 for i in range(10, 51, 5)]
            
    # Init Variables
    state = IncrementalScoreState(G, externality_dict, ancestor_cache)
    best_global_score = -float('inf')
    best_global_solution = set()
    
    ratio_idx = 0
    restart_count = 0
    
    # Search-control variables
    perturbation_mode = False 
    stop_search = False       

    print(f"    [{algorithm_name.upper()}] Start Search Loop... Budget: {max_runtime}s")

    while time.time() < deadline:
        if stop_search:
            break

        restart_count += 1
        is_new_best = False
        strategy = ""
        
        # A. Select a strategy.
        if ratio_idx < len(ratios_to_try):
            # --- Phase 1: Stratified proportion search ---
            curr_ratio = ratios_to_try[ratio_idx]
            count = int(n_candidates * curr_ratio)
            # Use the top K nodes as the initial solution.
            init_set = set(sorted_candidates[:count])
            strategy = f"Stratified-{curr_ratio:.2f}"
            ratio_idx += 1
            perturbation_mode = False 
        else:
            # --- Phase 2: Iterative perturbation search ---
            perturbation_mode = True
            strategy = "Perturb-Best"
            init_set = perturb_solution(best_global_solution, all_res, perturb_strength=0.05)
            
        # B. Initialize.
        state.initialize(init_set)
        
        # [Check 1] 
        improved_at_start = False
        if tracker.update(state.current_score, round_num=restart_count):
            best_global_score = state.current_score
            best_global_solution = set(state.current_result_set)
            is_new_best = True 
            improved_at_start = True
            
        # C. Hill climbing
        local_opt, steps = _greedy_descent_pass_incremental(
            state, all_res, tracker, deadline
        )
        
        # [Check 2] 
        improved_at_hc = False
        if tracker.update(local_opt, round_num=restart_count):
            best_global_score = local_opt
            best_global_solution = set(state.current_result_set)
            is_new_best = True 
            improved_at_hc = True
        
        # D. Stopping logic
        if perturbation_mode:
            if not (improved_at_start or improved_at_hc):
                print(f"    [Stop] No improvement in perturbation round. Convergence reached.")
                stop_search = True

        tracker.tick()
        
        # Print logs.
        if "Stratified" in strategy or is_new_best or strategy == "Perturb-Best":
            log_msg = f"    [Restart {restart_count}] {strategy:<20} | Best: {best_global_score:,.0f} | Steps: {steps}"
            if perturbation_mode and is_new_best:
                log_msg += " (Improved -> Continue)"
            print(log_msg)

    tracker.finalize(max_runtime)
    
    # Build the result.
    state.initialize(best_global_solution)
    final_full_nodes = state.get_full_closure_nodes()
    
    final_selected_edges = []
    edge_cost_map = {(u, v): c for u, v, c in edges_data}
    for u in final_full_nodes:
        for v in G.successors(u):
            if v in final_full_nodes:
                cost = edge_cost_map.get((u, v), 0.0)
                final_selected_edges.append((u, v, cost))
                
    return final_full_nodes, final_selected_edges, best_global_score, best_global_solution

# ==============================================================================
# --- 6. Entry point ---
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
            
            # === Call the unified interface ===
            sn, se, score, res_nodes = run_solver(
                G, nc, ed, ext,
                dataset_name=folder, 
                result_dir=result_dir,
                algorithm_name=algo_name,
                max_runtime=max_runtime
            )
            
            dur = time.time() - start_real
            save_result(result_dir, sn, se, score, dur, res_nodes)
            print(f"[{folder}] Final: {score:,.2f}, Time: {dur:.2f}s")
            print("="*60)
            
        except Exception as e:
            import traceback; traceback.print_exc()

if __name__ == "__main__":
    MAX_RUNTIME = 1000.0
    # Set the algorithm name to identify the ablation variant.
    ALGO_NAME = "nghc"
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str, default="../data/output")
    args = parser.parse_args()
    
    run_experiment(args.root_dir, ALGO_NAME, MAX_RUNTIME)