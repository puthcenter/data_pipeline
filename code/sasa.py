import os
import time
import pandas as pd
import networkx as nx
import random
import math
import statistics
from collections import defaultdict, deque
import argparse
from typing import List, Dict, Set, Tuple, Optional
import sys

# ==============================================================================
# --- 0. Convergence Tracker (Event & Time Driven) ---
# ==============================================================================
class ConvergenceTracker:
    """
    Tracks optimization progress using a hybrid logging strategy.
    Ensures that both new best solutions (Event-Driven) and periodic states (Time-Driven)
    are recorded for analysis.
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
            # flush=True ensures real-time visibility in logs
            print(msg, flush=True) 
            
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
        except Exception:
            pass

def _safe_float(x):
    try: return float(x)
    except: return 0.0

# ==============================================================================
# --- 1. Data Loading & Result Saving ---
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

    # For MwcpNe, edge costs are often transferred to nodes, so edge weights are 0
    edges_data = [(int(r["source"]), int(r["target"]), 0.0) for _, r in edge_df.iterrows()]
    G.add_weighted_edges_from(edges_data, weight="cost")

    externality_dict = {} 
    possible_names = ["externality.csv", "externality_matrix.csv"]
    ext_path = next((os.path.join(folder, n) for n in possible_names if os.path.exists(os.path.join(folder, n))), None)
    
    if ext_path:
        df_ext = pd.read_csv(ext_path)
        for _, row in df_ext.iterrows():
            s, t = int(row["source"]), int(row["target"])
            p = float(row.get("penalty", row.get("weight", 0.0)))
            if p > 0: 
                externality_dict[(s, t)] = p
                
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
    with open(os.path.join(output_dir, 'result_nodes_count.txt'), 'w') as f: f.write(str(len(res_nodes)))

# ==============================================================================
# --- 2. Incremental State Evaluator ---
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
        
        # --- Cost Initialization ---
        for r in self.selected_results:
            for a in self.ancestor_cache[r]:
                if self.node_usage_counts[a] == 0: self.current_total_cost += self.all_node_costs[a]
                self.node_usage_counts[a] += 1
        
        # --- Value & Penalty Initialization ---
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
                p_sum = 0.0
                for u, pen in self.ext_incoming[v]:
                    if u in self.selected_results:
                        p_sum += pen
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
        
        to_add = []
        to_remove = []
        ancestor_deltas = defaultdict(int)
        
        for n in nodes:
            if n in self.selected_results: to_remove.append(n)
            else: to_add.append(n)
        
        # --- Remove Phase ---
        for n in to_remove:
            self.selected_results.remove(n)
            for a in self.ancestor_cache[n]: ancestor_deltas[a] -= 1
            
            # Remove value and self-penalty
            net_val_n = self.node_vals[n] - self.curr_penalties[n]
            self.current_total_value -= net_val_n
            self.curr_penalties[n] = 0.0 
            
            # Reduce penalty on peers
            for target, pen in self.ext_outgoing[n]:
                if target in self.selected_results:
                    self.curr_penalties[target] -= pen
                    self.current_total_value += pen 

        # --- Add Phase ---
        for n in to_add:
            self.selected_results.add(n)
            for a in self.ancestor_cache[n]: ancestor_deltas[a] += 1
            
            # Calculate incoming penalty from existing peers
            p_n = 0.0
            for source, pen in self.ext_incoming[n]:
                if source in self.selected_results:
                    p_n += pen
            self.curr_penalties[n] = p_n
            
            # Apply penalty to existing peers
            outgoing_damage = 0.0
            for target, pen in self.ext_outgoing[n]:
                if target in self.selected_results:
                    self.curr_penalties[target] += pen
                    outgoing_damage += pen
            
            self.current_total_value += (self.node_vals[n] - p_n) - outgoing_damage

        # --- Cost Update Phase ---
        for anc, delta in ancestor_deltas.items():
            if delta == 0: continue
            old_count = self.node_usage_counts[anc]
            new_count = old_count + delta
            self.node_usage_counts[anc] = new_count
            
            if old_count == 0 and new_count > 0:
                self.current_total_cost += self.all_node_costs[anc]
            elif old_count > 0 and new_count == 0:
                self.current_total_cost -= self.all_node_costs[anc]
                
        return self.get_score() - old_score

    def rollback(self, nodes):
        """Reverts the last batch_flip operation."""
        self.batch_flip(nodes)

# ==============================================================================
# --- 3. Dynamic Structure Manager (SASA Core) ---
# ==============================================================================
class DynamicStructureManager:
    """
    Analyzes graph topology to identify "smart moves" for the annealing process.
    Detects conflict cliques and synergy relationships.
    """
    def __init__(self, G, result_nodes, externality_dict, ancestor_cache):
        self.G = G
        self.result_nodes = list(result_nodes)
        self.externality_dict = externality_dict
        self.ancestor_cache = ancestor_cache
        
        self.ancestor_cost_map = {}
        for r in result_nodes:
            self.ancestor_cost_map[r] = sum(G.nodes[a]['cost'] for a in ancestor_cache[r])

        print("    [SmartMove] Analyzing Graph Structures...", flush=True)
        
        # 1. Identify Strong Conflict Cliques (via Adaptive Thresholding)
        self.strong_conflict_cliques = self._analyze_conflicts()
        
        # 2. Identify Directed Synergy (Inverted Index Optimized)
        self.synergy_followers = self._analyze_synergy()
        
        self.static_net_values = {
            r: G.nodes[r]['value'] - self.ancestor_cost_map[r] 
            for r in result_nodes
        }
        self.high_value_nodes = sorted(
            [n for n in result_nodes if self.static_net_values[n] > 0],
            key=lambda x: self.static_net_values[x], reverse=True
        )

    def _analyze_conflicts(self):
        """Analyzes externality distribution to build strong conflict graphs."""
        if not self.externality_dict:
            return []
            
        penalties = list(self.externality_dict.values())
        if not penalties: return []
        
        mean_p = statistics.mean(penalties)
        std_p = statistics.stdev(penalties) if len(penalties) > 1 else 0.0
        
        threshold = mean_p + 0.5 * std_p
        
        strong_adj = defaultdict(set)
        count = 0
        for (u, v), p in self.externality_dict.items():
            if p >= threshold:
                strong_adj[u].add(v)
                strong_adj[v].add(u)
                count += 1
        
        print(f"      -> Conflict Analysis: Threshold={threshold:.4f}, Retained Edges={count}/{len(penalties)}", flush=True)
        
        # Greedy Clique Construction
        cliques = []
        visited = set()
        nodes_sorted = sorted(strong_adj.keys(), key=lambda k: len(strong_adj[k]), reverse=True)
        
        for n in nodes_sorted:
            if n in visited: continue
            current_clique = {n}
            candidates = strong_adj[n]
            for cand in candidates:
                if cand in visited: continue
                if all(c in strong_adj[cand] for c in current_clique):
                    current_clique.add(cand)
            if len(current_clique) > 1:
                cliques.append(list(current_clique))
                visited.update(current_clique)
        
        print(f"      -> Identified {len(cliques)} strong conflict cliques.", flush=True)
        return cliques

    def _analyze_synergy(self):
        """
        Identifies synergy/free-rider relationships using inverted indices.
        A 'follower' effectively 'free-rides' on the cost paid by a 'leader'.
        """
        followers = defaultdict(list)
        strong_links = 0
        SYNERGY_THRESHOLD = 0.8 
        
        # 1. Build Inverted Index
        ancestor_to_results = defaultdict(list)
        valid_results = []
        
        for r in self.result_nodes:
            if self.ancestor_cost_map.get(r, 0) > 1e-4:
                valid_results.append(r)
                for anc in self.ancestor_cache[r]:
                    if self.G.nodes[anc].get('cost', 0) > 0:
                        ancestor_to_results[anc].append(r)

        print(f"      [Synergy] Built index for {len(valid_results)} nodes.", flush=True)
        
        # 2. Iterate Potential Leaders
        for u in valid_results:
            cost_u = self.ancestor_cost_map[u]
            candidates = set()
            u_ancestors = self.ancestor_cache[u]
            
            for anc in u_ancestors:
                if anc in ancestor_to_results:
                    for v in ancestor_to_results[anc]:
                        if u != v:
                            candidates.add(v)
            
            for v in candidates:
                cost_v = self.ancestor_cost_map[v]
                if cost_v < 1e-4: continue

                anc_v = self.ancestor_cache[v]
                common = u_ancestors.intersection(anc_v)
                intersection_cost = sum(self.G.nodes[a]['cost'] for a in common)
                
                ratio_v_covered = intersection_cost / cost_v
                if ratio_v_covered >= SYNERGY_THRESHOLD:
                    followers[u].append(v)
                    strong_links += 1
        
        print(f"      -> Synergy Analysis (Optimized): Threshold={SYNERGY_THRESHOLD}, Directed Links={strong_links}", flush=True)
        return followers

# ==============================================================================
# --- 4. Move Generator (Strategy Routing v2) ---
# ==============================================================================
def get_smart_move(state: IncrementalState, manager: DynamicStructureManager, T: float, progress: float):
    """
    Selects a move strategy based on the current annealing progress.
    """
    r = random.random()
    
    # Adaptive Probability Probabilities
    p_synergy = 0.4 * (1 - progress)  
    p_conflict = 0.3 + 0.1 * progress 
    p_mutex = 0.2
    
    # --- Strategy A: Synergy Batch (Group Activation) ---
    if manager.synergy_followers and r < p_synergy:
        seeds = list(manager.synergy_followers.keys())
        if seeds:
            leader = random.choice(seeds)
            followers = manager.synergy_followers[leader]
            moves = [leader]
            if leader not in state.selected_results:
                num_to_take = random.randint(1, len(followers))
                moves.extend(random.sample(followers, num_to_take))
            return moves

    # --- Strategy B: Strong Conflict Resolution (Targeted Flipping) ---
    elif r < (p_synergy + p_conflict):
        if state.selected_results:
            # Limit sampling size for performance
            sample_size = min(len(state.selected_results), 20)
            candidates = random.sample(list(state.selected_results), sample_size)
            
            victim = None
            max_pen = -1
            
            for c in candidates:
                if state.curr_penalties[c] > max_pen:
                    max_pen = state.curr_penalties[c]
                    victim = c
            
            if victim and max_pen > 0:
                enemies = state.ext_incoming[victim]
                active_enemies = [u for u, p in enemies if u in state.selected_results]
                
                if active_enemies:
                    enemy = random.choice(active_enemies)
                    # Flip either the victim or the enemy
                    if random.random() < 0.5: return [victim]
                    else: return [enemy]

    # --- Strategy C: Mutex Swap (Clique Optimization) ---
    elif manager.strong_conflict_cliques and r < (p_synergy + p_conflict + p_mutex):
        clique = random.choice(manager.strong_conflict_cliques)
        active_in_clique = [n for n in clique if n in state.selected_results]
        
        if len(active_in_clique) > 1:
            moves = list(active_in_clique)
            survivor = random.choice(active_in_clique)
            moves.remove(survivor) 
            return moves 
        elif len(active_in_clique) == 0:
            return [random.choice(clique)]
        else:
            current = active_in_clique[0]
            target = random.choice(clique)
            if current != target:
                return [current, target]

    # --- Strategy D: Random Exploration ---
    target = random.choice(manager.result_nodes)
    return [target]

# ==============================================================================
# --- 5. Main Solver Interface ---
# ==============================================================================
def run_solver(
    G: nx.DiGraph, 
    node_costs: Dict, 
    edges_data: List, 
    externality_dict: Dict, 
    dataset_name: str,
    result_dir: str,
    algorithm_name: str = "sa_smart_v2",
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
    
    all_res = [n for n, d in G.nodes(data=True) if d.get('is_result')]
    if not all_res: 
        tracker.finalize(max_runtime)
        return set(), [], 0.0, set()
    
    ancestor_cache = {}
    for r in all_res:
        ancs = nx.ancestors(G, r); ancs.add(r)
        ancestor_cache[r] = ancs
    
    move_mgr = DynamicStructureManager(G, all_res, externality_dict, ancestor_cache)
    
    global_best_score = -float('inf')
    global_best_sol = set()
    restart_idx = 0
    
    print(f"    [{algorithm_name.upper()}] Start... Budget: {max_runtime}s", flush=True)

    while time.time() < deadline:
        restart_idx += 1
        
        # Initialization: Start with all result nodes active
        init_set = set(all_res) 
        
        state = IncrementalState(G, init_set, ancestor_cache, externality_dict)
        curr_val = state.get_score()
        
        local_best = curr_val
        best_sol_in_restart = state.selected_results.copy()
        
        if tracker.update(curr_val, round_num=restart_idx):
            global_best_score = curr_val
            global_best_sol = best_sol_in_restart.copy()
            
        # Automatic Temperature Initialization
        T_init = 100.0
        deltas = []
        for _ in range(20):
            if time.time() > deadline: break
            nodes = get_smart_move(state, move_mgr, 100, 0.0)
            d = state.batch_flip(nodes)
            if d < 0: deltas.append(abs(d))
            state.rollback(nodes)
        
        if deltas:
            avg_d = sum(deltas) / len(deltas)
            T_init = max(avg_d / math.log(2), 10.0)
            
        T = T_init
        T_min = 0.1
        alpha = 0.97
        
        stagnation = 0
        iter_max = int(len(all_res) * 0.8)
        
        # --- Annealing Loop ---
        while T > T_min and time.time() < deadline:
            improved = False
            progress = (time.time() - start_time) / max_runtime
            
            tracker.tick()
            
            for i in range(iter_max):
                if i % 100 == 0:
                    tracker.tick()
                    if time.time() > deadline: break

                nodes = get_smart_move(state, move_mgr, T, progress)
                if not nodes: continue
                
                delta = state.batch_flip(nodes)
                
                accept = False
                if delta > -1e-9: 
                    accept = True
                else:
                    if random.random() < math.exp(delta / T):
                        accept = True
                
                if accept:
                    val = state.get_score()
                    if val > local_best + 1e-6:
                        local_best = val
                        best_sol_in_restart = state.selected_results.copy()
                        improved = True
                        stagnation = 0
                else:
                    state.rollback(nodes)
            
            if not improved: stagnation += 1
            if stagnation > 50: break 
            
            T *= alpha
            
        if tracker.update(local_best, round_num=restart_idx):
            global_best_score = local_best
            global_best_sol = best_sol_in_restart.copy()
        
        print(f"    [Restart {restart_idx}] Best: {global_best_score:,.0f} | InitT: {T_init:.1f}", flush=True)

    tracker.finalize(max_runtime)
    
    final_nodes = set()
    for r in global_best_sol: final_nodes.update(ancestor_cache[r])
    
    final_edges = []
    for u in final_nodes:
        if G.has_node(u):
            for v in G.successors(u):
                if v in final_nodes:
                    final_edges.append((u, v, 0.0))
            
    return final_nodes, final_edges, global_best_score, global_best_sol

# ==============================================================================
# --- 6. Execution Entry Point ---
# ==============================================================================
def run_experiment(root_dir, algo_name, max_runtime):
    if not os.path.exists(root_dir): return
    
    for folder in os.listdir(root_dir):
        path = os.path.join(root_dir, folder)
        if not os.path.isdir(path): continue
        if "dag" not in folder: continue 

        try:
            print(f"--- Processing [{folder}] (Algo: {algo_name}) ---", flush=True)
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
            
            dur = time.time() - start_real
            save_result(result_dir, sn, se, score, dur, res_nodes)
            
            print(f"[{folder}] Final: {score:,.2f}, Time: {dur:.2f}s", flush=True)
            print("="*60, flush=True)
            
        except Exception as e:
            import traceback; traceback.print_exc()

if __name__ == "__main__":
    MAX_RUNTIME = 1000.0
    ALGO_NAME = "sasa"
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str, default="../data/output")
    args = parser.parse_args()
    
    run_experiment(args.root_dir, ALGO_NAME, MAX_RUNTIME)