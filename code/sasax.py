import os
import time
import pandas as pd
import networkx as nx
import random
import math
import statistics
from collections import defaultdict
import argparse
from typing import List, Dict, Set, Tuple

"""
Module Identification:
This script implements the Ablation Study for Structure-Aware Simulated Annealing (SASA). 
It systematically isolates specific structural heuristics ("Synergy-Only" vs. "Conflict-Only") 
to quantify their individual contributions to the optimization process. It is designed to 
generate two distinct result sets (sasa_synergy_only_result and sasa_conflict_only_result) 
in a single execution pass.
"""

# ==============================================================================
# --- 0. Infrastructure (Logging & Data Loading) ---
# ==============================================================================
class ConvergenceTracker:
    """
    Tracks optimization progress using a hybrid logging strategy.
    """
    def __init__(self, algorithm_name, dataset_name, save_path="convergence.csv", log_interval=10.0):
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
            
            # Use flush=True to ensure real-time visibility in logs
            print(f"      [{self.algorithm_name}] New Best: {value:,.2f} @ {elapsed:.2f}s", flush=True)
            
            self.save_to_csv()
            self.last_log_time = time.time()
            return True
        return False

    def tick(self):
        """
        [Time-Driven] Periodic logging to ensure continuous data points.
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
        except: 
            pass

def _safe_float(x):
    try: return float(x)
    except: return 0.0

def load_all_data(folder: str):
    """Loads graph nodes, edges, and externality constraints."""
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
        
    edges_data = [(int(r["source"]), int(r["target"]), 0.0) for _, r in edge_df.iterrows()]
    G.add_weighted_edges_from(edges_data, weight="cost")
    
    externality_dict = {}
    possible = ["externality.csv", "externality_matrix.csv"]
    ext_path = next((os.path.join(folder, n) for n in possible if os.path.exists(os.path.join(folder, n))), None)
    
    if ext_path:
        df_ext = pd.read_csv(ext_path)
        for _, row in df_ext.iterrows():
            p = float(row.get("penalty", row.get("weight", 0.0)))
            if p > 0: 
                externality_dict[(int(row["source"]), int(row["target"]))] = p
                
    return G, node_costs, edges_data, externality_dict

def save_result(output_dir, selected_nodes, selected_edges, score, duration, res_nodes):
    """Saves the final optimization result."""
    os.makedirs(output_dir, exist_ok=True)
    pd.DataFrame({'id': sorted(list(selected_nodes))}).to_csv(os.path.join(output_dir, 'sub_nodes.csv'), index=False)
    
    if selected_edges:
        pd.DataFrame(selected_edges, columns=['source', 'target', 'cost']).to_csv(os.path.join(output_dir, 'sub_edges.csv'), index=False)
    else:
        pd.DataFrame(columns=['source', 'target', 'cost']).to_csv(os.path.join(output_dir, 'sub_edges.csv'), index=False)
        
    with open(os.path.join(output_dir, 'score.txt'), 'w') as f: f.write(str(score))
    with open(os.path.join(output_dir, 'time.txt'), 'w') as f: f.write(f"{duration:.6f}")
    
    if not isinstance(res_nodes, set): res_nodes = set(res_nodes)
    pd.DataFrame({'id': sorted(list(res_nodes))}).to_csv(os.path.join(output_dir, 'result_nodes.csv'), index=False)

# ==============================================================================
# --- 1. Incremental State Management ---
# ==============================================================================
class IncrementalState:
    """
    Manages the current solution state and efficiently calculates the objective score
    using batch updates (batch_flip).
    """
    def __init__(self, G, initial_selection, ancestor_cache, externality_dict):
        self.G = G
        self.selected_results = set(initial_selection)
        self.ancestor_cache = ancestor_cache
        self.node_usage_counts = defaultdict(int)
        self.current_total_cost = 0.0
        self.all_node_costs = {n: d['cost'] for n, d in G.nodes(data=True)}
        
        # Initialize Cost
        for r in self.selected_results:
            for a in self.ancestor_cache[r]:
                if self.node_usage_counts[a] == 0: self.current_total_cost += self.all_node_costs[a]
                self.node_usage_counts[a] += 1
        
        # Initialize Value & Penalty
        self.current_total_value = 0.0
        self.node_vals = {n: d.get('value', 0.0) for n, d in G.nodes(data=True)}
        self.curr_penalties = defaultdict(float)
        self.ext_incoming = defaultdict(list)
        self.ext_outgoing = defaultdict(list)
        
        if externality_dict:
            for (u, v), p in externality_dict.items():
                self.ext_outgoing[u].append((v, p))
                self.ext_incoming[v].append((u, p))
            for v in self.selected_results:
                p_sum = sum(p for u, p in self.ext_incoming[v] if u in self.selected_results)
                self.curr_penalties[v] = p_sum
        
        for r in self.selected_results:
            self.current_total_value += (self.node_vals[r] - self.curr_penalties[r])

    def get_score(self):
        return self.current_total_value - self.current_total_cost

    def batch_flip(self, nodes):
        """
        Updates the solution by toggling the presence of the specified nodes.
        Returns the delta (score change).
        """
        if not nodes: return 0.0
        old_score = self.get_score()
        to_add, to_remove = [], []
        ancestor_deltas = defaultdict(int)
        
        for n in nodes:
            if n in self.selected_results: to_remove.append(n)
            else: to_add.append(n)
            
        # Remove Phase
        for n in to_remove:
            self.selected_results.remove(n)
            for a in self.ancestor_cache[n]: ancestor_deltas[a] -= 1
            
            net_val_n = self.node_vals[n] - self.curr_penalties[n]
            self.current_total_value -= net_val_n
            self.curr_penalties[n] = 0.0
            
            for target, pen in self.ext_outgoing[n]:
                if target in self.selected_results:
                    self.curr_penalties[target] -= pen
                    self.current_total_value += pen

        # Add Phase
        for n in to_add:
            self.selected_results.add(n)
            for a in self.ancestor_cache[n]: ancestor_deltas[a] += 1
            
            p_n = sum(p for s, p in self.ext_incoming[n] if s in self.selected_results)
            self.curr_penalties[n] = p_n
            
            outgoing_damage = 0.0
            for target, pen in self.ext_outgoing[n]:
                if target in self.selected_results:
                    self.curr_penalties[target] += pen
                    outgoing_damage += pen
            
            self.current_total_value += (self.node_vals[n] - p_n) - outgoing_damage

        # Cost Update Phase
        for anc, delta in ancestor_deltas.items():
            if delta == 0: continue
            old_count = self.node_usage_counts[anc]
            new_count = old_count + delta
            self.node_usage_counts[anc] = new_count
            
            if old_count == 0 and new_count > 0: self.current_total_cost += self.all_node_costs[anc]
            elif old_count > 0 and new_count == 0: self.current_total_cost -= self.all_node_costs[anc]
        
        return self.get_score() - old_score

    def rollback(self, nodes):
        """Reverts the last batch_flip operation."""
        self.batch_flip(nodes)

# ==============================================================================
# --- 2. Structure Manager ---
# ==============================================================================
class DynamicStructureManager:
    """
    Analyzes graph topology to identify "smart moves" (Conflict Cliques & Synergy).
    """
    def __init__(self, G, result_nodes, externality_dict, ancestor_cache):
        self.G = G
        self.result_nodes = list(result_nodes)
        self.externality_dict = externality_dict
        self.ancestor_cache = ancestor_cache
        
        self.ancestor_cost_map = {r: sum(G.nodes[a]['cost'] for a in ancestor_cache[r]) for r in result_nodes}
        
        # Analyze structures once for consistency across experiments
        self.strong_conflict_cliques = self._analyze_conflicts()
        self.synergy_followers = self._analyze_synergy()

    def _analyze_conflicts(self):
        """Identifies cliques of nodes with strong mutual negative externalities."""
        if not self.externality_dict: return []
        penalties = list(self.externality_dict.values())
        threshold = statistics.mean(penalties) + 0.5 * (statistics.stdev(penalties) if len(penalties) > 1 else 0)
        
        strong_adj = defaultdict(set)
        for (u, v), p in self.externality_dict.items():
            if p >= threshold:
                strong_adj[u].add(v); strong_adj[v].add(u)
                
        cliques = []
        visited = set()
        for n in sorted(strong_adj.keys(), key=lambda k: len(strong_adj[k]), reverse=True):
            if n in visited: continue
            curr = {n}
            for cand in strong_adj[n]:
                if cand not in visited and all(c in strong_adj[cand] for c in curr): curr.add(cand)
            if len(curr) > 1:
                cliques.append(list(curr))
                visited.update(curr)
        return cliques

    def _analyze_synergy(self):
        """Identifies follower nodes that can free-ride on leaders."""
        followers = defaultdict(list)
        inv_idx = defaultdict(list)
        valid = [r for r in self.result_nodes if self.ancestor_cost_map[r] > 1e-4]
        for r in valid:
            for anc in self.ancestor_cache[r]:
                if self.G.nodes[anc].get('cost', 0) > 0: inv_idx[anc].append(r)
        
        for u in valid:
            cands = set()
            for anc in self.ancestor_cache[u]:
                if anc in inv_idx: cands.update(v for v in inv_idx[anc] if u != v)
            for v in cands:
                if self.ancestor_cost_map[v] < 1e-4: continue
                common = sum(self.G.nodes[a]['cost'] for a in self.ancestor_cache[u].intersection(self.ancestor_cache[v]))
                if common / self.ancestor_cost_map[v] >= 0.8: followers[u].append(v)
        return followers

# ==============================================================================
# --- 3. [Ablation Specific] Move Generator ---
# ==============================================================================
def get_ablation_move(state: IncrementalState, manager: DynamicStructureManager, progress: float, strategy: str):
    """
    Generates moves restricted to a specific heuristic strategy for ablation purposes.
    """
    r = random.random()
    
    # --- Strategy 1: Synergy Only ---
    # Enables only the "Free-rider/Synergy" mechanism.
    if strategy == 'synergy_only':
        # 50% chance to attempt a Synergy Move
        if r < 0.5 and manager.synergy_followers:
            seeds = list(manager.synergy_followers.keys())
            if seeds:
                leader = random.choice(seeds)
                moves = [leader]
                # If leader is inactive, try activating it along with followers
                if leader not in state.selected_results:
                    followers = manager.synergy_followers[leader]
                    num_to_take = random.randint(1, len(followers))
                    moves.extend(random.sample(followers, num_to_take))
                return moves
        # Fallback to random walk
        return [random.choice(manager.result_nodes)]

    # --- Strategy 2: Conflict Only ---
    # Enables only the "Conflict Resolution/Mutex" mechanism.
    elif strategy == 'conflict_only':
        # 60% chance to attempt a Conflict Move
        if r < 0.6: 
            # Sub-mode A: Victim Swap (Target high-penalty nodes)
            if state.selected_results and random.random() < 0.6:
                sample_size = min(len(state.selected_results), 20)
                candidates = random.sample(list(state.selected_results), sample_size)
                # Find the candidate suffering the most penalty
                victim = max(candidates, key=lambda c: state.curr_penalties[c], default=None)
                
                if victim and state.curr_penalties[victim] > 0:
                    enemies = state.ext_incoming[victim]
                    active_enemies = [u for u, p in enemies if u in state.selected_results]
                    if active_enemies:
                        # 50/50 chance to disable victim or disable the aggressor
                        if random.random() < 0.5: return [victim]
                        else: return [random.choice(active_enemies)]
            
            # Sub-mode B: Mutex Swap (Target strong conflict cliques)
            if manager.strong_conflict_cliques:
                clique = random.choice(manager.strong_conflict_cliques)
                active_in_clique = [n for n in clique if n in state.selected_results]
                
                if len(active_in_clique) > 1:
                    # Mutex Constraint: Keep one, disable others
                    moves = list(active_in_clique)
                    survivor = random.choice(active_in_clique)
                    moves.remove(survivor)
                    return moves
                elif len(active_in_clique) == 0:
                    # Try enabling one member
                    return [random.choice(clique)]
                else:
                    # Swap operation
                    current = active_in_clique[0]
                    target = random.choice(clique)
                    if current != target: return [current, target]
                    else: return [current] # Equivalent to toggle

        # Fallback to random walk
        return [random.choice(manager.result_nodes)]

    # Default Fallback (Pure Random)
    return [random.choice(manager.result_nodes)]

# ==============================================================================
# --- 4. Unified Solver ---
# ==============================================================================
def run_solver(
    G: nx.DiGraph, 
    node_costs: Dict, 
    edges_data: List, 
    externality_dict: Dict, 
    dataset_name: str,
    result_dir: str,
    algorithm_name: str,      
    strategy_mode: str,       
    max_runtime: float = 2000.0
):
    os.makedirs(result_dir, exist_ok=True)
    tracker = ConvergenceTracker(algorithm_name, dataset_name, os.path.join(result_dir, "convergence.csv"))
    
    start_time = time.time()
    deadline = start_time + max_runtime
    
    all_res = [n for n, d in G.nodes(data=True) if d.get('is_result')]
    if not all_res: 
        tracker.finalize(max_runtime)
        return set(), [], 0.0, set()
    
    # Initialize Cache & Manager
    ancestor_cache = {r: nx.ancestors(G, r) | {r} for r in all_res}
    move_mgr = DynamicStructureManager(G, all_res, externality_dict, ancestor_cache)
    
    global_best_score = -float('inf')
    global_best_sol = set()
    restart_idx = 0
    
    print(f"    [{algorithm_name.upper()}] Mode: {strategy_mode} | Start...", flush=True)

    while time.time() < deadline:
        restart_idx += 1
        init_set = set(all_res) # Initialize with all nodes active
        state = IncrementalState(G, init_set, ancestor_cache, externality_dict)
        
        curr_val = state.get_score()
        local_best = curr_val
        best_sol_in_restart = state.selected_results.copy()
        
        if tracker.update(curr_val, restart_idx):
            global_best_score = curr_val; global_best_sol = best_sol_in_restart.copy()
            
        # Initial Temperature Estimation
        deltas = []
        for _ in range(20):
            nodes = get_ablation_move(state, move_mgr, 0.0, strategy_mode)
            d = state.batch_flip(nodes)
            if d < 0: deltas.append(abs(d))
            state.rollback(nodes)
        
        T = max((sum(deltas)/len(deltas))/math.log(2), 10.0) if deltas else 100.0
        T_min, alpha = 0.1, 0.97
        stagnation = 0
        iter_max = int(len(all_res) * 0.8)
        
        while T > T_min and time.time() < deadline:
            improved = False
            progress = (time.time() - start_time) / max_runtime
            tracker.tick()
            
            for i in range(iter_max):
                if i % 100 == 0:
                    tracker.tick()
                    if time.time() > deadline: break

                # [Core Call] Get move based on ablation strategy
                nodes = get_ablation_move(state, move_mgr, progress, strategy_mode)
                if not nodes: continue
                
                delta = state.batch_flip(nodes)
                if delta > -1e-9 or random.random() < math.exp(delta / T):
                    val = state.get_score()
                    if val > local_best + 1e-6:
                        local_best = val; best_sol_in_restart = state.selected_results.copy()
                        improved = True; stagnation = 0
                else:
                    state.rollback(nodes)
            
            if not improved: stagnation += 1
            if stagnation > 50: break
            T *= alpha
            
        if tracker.update(local_best, restart_idx):
            global_best_score = local_best; global_best_sol = best_sol_in_restart.copy()
        
        print(f"    [{algorithm_name} R{restart_idx}] Best: {global_best_score:,.0f}", flush=True)

    tracker.finalize(max_runtime)
    
    final_nodes = set()
    for r in global_best_sol: final_nodes.update(ancestor_cache[r])
    final_edges = [(u, v, 0.0) for u in final_nodes for v in G.successors(u) if v in final_nodes]
            
    return final_nodes, final_edges, global_best_score, global_best_sol

# ==============================================================================
# --- 5. Main Execution Entry Point ---
# ==============================================================================
def run_ablation_experiment(root_dir, max_runtime):
    if not os.path.exists(root_dir): return
    
    # Define Ablation Experiments
    experiments = [
        ("sasa_synergy_only",  "synergy_only"),   # Algorithm Name -> Strategy Mode
        ("sasa_conflict_only", "conflict_only"),  # Algorithm Name -> Strategy Mode
    ]

    for folder in os.listdir(root_dir):
        path = os.path.join(root_dir, folder)
        if not os.path.isdir(path) or "dag" not in folder: continue
        
        print(f"\n{'='*60}\nProcessing Dataset: [{folder}]\n{'='*60}", flush=True)
        try:
            G, nc, ed, ext = load_all_data(path)
            
            for algo_name, strategy in experiments:
                print(f"\n--- Running Experiment: {algo_name} ---", flush=True)
                
                # Result directory parallel to standard results
                res_dir = os.path.join(path, f"{algo_name}_result") 
                
                start_t = time.time()
                sn, se, score, res_nodes = run_solver(
                    G, nc, ed, ext,
                    dataset_name=folder,
                    result_dir=res_dir,
                    algorithm_name=algo_name,     
                    strategy_mode=strategy,       
                    max_runtime=max_runtime
                )
                dur = time.time() - start_t
                save_result(res_dir, sn, se, score, dur, res_nodes)
                print(f" -> Finished {algo_name}: {score:,.2f} in {dur:.2f}s", flush=True)

        except Exception as e:
            import traceback; traceback.print_exc()

if __name__ == "__main__":
    # Configuration
    MAX_RUNTIME = 1000.0
    ROOT_DIR = "../data/output"
    
    run_ablation_experiment(ROOT_DIR, MAX_RUNTIME)