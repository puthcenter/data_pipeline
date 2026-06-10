import os
import time
import pandas as pd
import networkx as nx
from collections import defaultdict
import argparse
from typing import List, Dict, Set, Tuple

# 新增并发库
from concurrent.futures import ProcessPoolExecutor, as_completed

# ==============================================================================
# --- 0. 基础工具类 ---
# ==============================================================================
class ConvergenceTracker:
    def __init__(self, algorithm_name, dataset_name, save_path="convergence.csv", log_interval=1.0):
        self.algorithm_name = algorithm_name
        self.dataset_name = dataset_name  # 保存当前数据集名称
        self.save_path = save_path
        self.log_interval = log_interval
        
        self.start_time = time.time()
        self.last_log_time = self.start_time
        
        self.history = []
        self.current_best_value = -float('inf')
        
        self._log(0.0, -float('inf'))

    def update(self, value, round_num=None):
        if value > self.current_best_value + 1e-9:
            self.current_best_value = value
            elapsed = time.time() - self.start_time
            
            self._log(elapsed, value, round_num)
            
            # 【改动 1】：在输出前面加上 [数据集名称] 作为 Tag，并强制刷新缓冲
            msg = f"[{self.dataset_name}] [{self.algorithm_name.upper()}] Best: {value:,.2f} @ {elapsed:.2f}s"
            if round_num is not None:
                msg += f" | Step: {round_num}"
            print(msg, flush=True) 
            
            self.save_to_csv()
            self.last_log_time = time.time()
            return True
        return False

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
# --- 1. 数据加载 & 2. 增量计算状态管理器 ---
# (这部分代码无需改动，保持你原来的逻辑)
# ==============================================================================
def load_all_data(folder: str):
    # ... (保持原样) ...
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
    # ... (保持原样) ...
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

class CachedGreedyState:
    # ... (保持原样) ...
    def __init__(self, G, node_costs, externality_dict, all_result_nodes):
        self.node_costs = node_costs
        self.node_vals = {n: d.get('value', 0.0) for n, d in G.nodes(data=True)}
        
        self.ext_adj = defaultdict(list)
        for (u, v), p in externality_dict.items():
            self.ext_adj[u].append((v, p))
            self.ext_adj[v].append((u, p))
            
        self.ancestor_to_users = defaultdict(set)
        self.res_to_ancestors = {}
        
        for r in all_result_nodes:
            ancs = nx.ancestors(G, r)
            ancs.add(r)
            self.res_to_ancestors[r] = list(ancs)
            for anc in ancs:
                self.ancestor_to_users[anc].add(r)
                
        self.active_results = set(all_result_nodes)
        self.current_score = self._calculate_initial_total_score(externality_dict)
        
        self.current_gains = {}
        for r in self.active_results:
            self.current_gains[r] = self._calculate_single_gain(r)
            
    def _calculate_initial_total_score(self, externality_dict):
        val_sum = sum(self.node_vals[r] for r in self.active_results)
        cost_sum = sum(self.node_costs[anc] for anc, users in self.ancestor_to_users.items() if len(users) > 0)
        ext_sum = 0.0
        for (u, v), p in externality_dict.items():
            if u in self.active_results and v in self.active_results:
                ext_sum += p
        return val_sum - cost_sum - ext_sum

    def _calculate_single_gain(self, node):
        gain = -self.node_vals[node]
        for peer, pen in self.ext_adj[node]:
            if peer in self.active_results and peer != node:
                gain += pen
        for anc in self.res_to_ancestors[node]:
            if len(self.ancestor_to_users[anc]) == 1:
                gain += self.node_costs[anc]
        return gain

    def remove_node_incremental(self, node):
        self.active_results.remove(node)
        gain_val = self.current_gains.pop(node)
        self.current_score += gain_val 
        
        for peer, pen in self.ext_adj[node]:
            if peer in self.active_results:
                self.current_gains[peer] -= pen

        for anc in self.res_to_ancestors[node]:
            users = self.ancestor_to_users[anc]
            users.remove(node)
            if len(users) == 1:
                survivor = next(iter(users))
                if survivor in self.current_gains:
                    self.current_gains[survivor] += self.node_costs[anc]

    def get_full_closure_nodes(self):
        return {anc for anc, users in self.ancestor_to_users.items() if len(users) > 0}

# ==============================================================================
# --- 3. 求解逻辑 ---
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
    
    # 【改动 2】：添加 dataset_name 前缀
    print(f"[{dataset_name}] [{algorithm_name.upper()}] Init State with {len(all_res)} nodes...", flush=True)
    
    state = CachedGreedyState(G, node_costs, externality_dict, all_res)
    tracker.update(state.current_score, round_num=0)
    
    next_log_time = time.time() + log_interval
    initial_candidates = [(r, state.current_gains[r]) for r in state.active_results]
    initial_candidates.sort(key=lambda x: x[1], reverse=True)
    
    # 【改动 2】：添加 dataset_name 前缀
    print(f"[{dataset_name}] [STATIC] Pre-calculated order for {len(initial_candidates)} nodes.", flush=True)
    
    step = 0
    for node_to_check, initial_gain in initial_candidates:
        step += 1
        current_real_gain = state.current_gains.get(node_to_check, -float('inf'))
        if current_real_gain > 0:
            state.remove_node_incremental(node_to_check)
        
        now = time.time()
        if now >= next_log_time:
            tracker.update(state.current_score, round_num=step)
            next_log_time = now + log_interval
        
        if now - start_time > max_runtime:
            tracker.update(state.current_score, round_num=step)
            print(f"[{dataset_name}] [{algorithm_name.upper()}] Timeout reached.", flush=True)
            break

    tracker.update(state.current_score, round_num=step)
    tracker.finalize(time.time() - start_time)
    
    final_result_nodes = state.active_results
    final_nodes = state.get_full_closure_nodes()
    final_edges = [(u, v, c) for u, v, c in edges_data if u in final_nodes and v in final_nodes]
            
    return final_nodes, final_edges, state.current_score, final_result_nodes

# ==============================================================================
# --- 执行入口 ---
# ==============================================================================

# 【改动 3】：将对单个文件夹的处理逻辑抽离为一个独立函数，以便被多进程调用
def process_single_folder(path: str, folder: str, algo_name: str, max_runtime: float):
    try:
        print(f"[{folder}] >>> Starting Processing (Algo: {algo_name})", flush=True)
        
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
        
        print(f"[{folder}] <<< Finished! Final: {score:,.2f}, Time: {duration:.2f}s", flush=True)
        return folder, True
    
    except Exception as e:
        # 多进程中异常追踪栈容易丢失，手动打印
        import traceback
        error_msg = traceback.format_exc()
        print(f"[{folder}] ERROR:\n{error_msg}", flush=True)
        return folder, False


# 【改动 4】：主执行函数使用进程池进行并发
def run_experiment(root_dir, algo_name, max_runtime, max_workers=None):
    if not os.path.exists(root_dir): return
    
    # 收集所有需要处理的任务
    tasks = []
    for folder in os.listdir(root_dir):
        path = os.path.join(root_dir, folder)
        if os.path.isdir(path) and "dag" in folder:
            tasks.append((path, folder, algo_name, max_runtime))
    
    if not tasks:
        print("No valid folders found.")
        return

    print(f"Total tasks found: {len(tasks)}. Starting parallel processing with max_workers={max_workers or 'Auto'}...")
    print("="*80)

    # 使用 ProcessPoolExecutor 进行并行处理
    # max_workers 默认为 None 时，会自动设置为机器的 CPU 核心数
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        futures = {
            executor.submit(process_single_folder, *task_args): task_args[1] 
            for task_args in tasks
        }
        
        # 等待结果（如果需要统计完成进度，可以在这里做）
        completed_count = 0
        for future in as_completed(futures):
            folder_name = futures[future]
            completed_count += 1
            # 获取函数的返回值 (folder, success_bool)
            _, success = future.result() 
            status = "SUCCESS" if success else "FAILED"
            print(f"--- System: Progress {completed_count}/{len(tasks)} ({folder_name} {status}) ---", flush=True)

if __name__ == "__main__":
    MAX_RUNTIME = 100.0
    ALGO_NAME = "greedy"
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str, default="../data/output")
    # 允许命令行指定并行进程数，如果不填则自动拉满CPU
    parser.add_argument('--workers', type=int, default=None, help="Number of parallel processes")
    args = parser.parse_args()
    
    # 开始实验
    run_experiment(args.root_dir, ALGO_NAME, MAX_RUNTIME, max_workers=args.workers)