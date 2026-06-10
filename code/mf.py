import os
import time
import random
import math
import argparse
import pandas as pd
import networkx as nx
import numpy as np
from collections import defaultdict
from typing import List, Dict, Set, Tuple, Optional

# ==============================================================================
# --- 0. 改进的记录器类 (保持不变) ---
# ==============================================================================
class ConvergenceTracker:
    def __init__(self, algorithm_name, dataset_name, save_path="convergence.csv", log_interval=20.0):
        self.algorithm_name = algorithm_name
        self.dataset_name = dataset_name
        self.save_path = save_path
        self.log_interval = log_interval
        
        self.start_time = time.time()
        self.last_log_time = self.start_time
        
        self.history = []
        self.current_best_value = -float('inf')
        
        # 初始记录
        self._log(0.0, -float('inf'))

    def update(self, value, round_num=None):
        """
        [Event-Driven] 发现新高时记录
        """
        if value > self.current_best_value + 1e-9:
            self.current_best_value = value
            elapsed = time.time() - self.start_time
            
            self._log(elapsed, value, round_num)
            
            msg = f"      [Monitor] New Best: {value:,.2f} @ {elapsed:.2f}s"
            if round_num is not None:
                msg += f" | Iter: {round_num}"
            print(msg)
            
            self.save_to_csv()
            self.last_log_time = time.time() # 重置定时器
            return True
        return False

    def tick(self):
        """
        [Time-Driven] 心跳记录
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

# ==============================================================================
# --- 1. 数据加载与保存工具 ---
# ==============================================================================
def load_graph_data(folder):
    node_df = pd.read_csv(os.path.join(folder, "nodes.csv"))
    edge_df = pd.read_csv(os.path.join(folder, "edges.csv"))
    
    G = nx.DiGraph()
    for _, row in node_df.iterrows():
        nid = int(row['id'])
        G.add_node(nid,
                   value=float(row['value']),
                   cost=float(row['incoming_cost']),
                   is_result=bool(row['is_result']))
    for _, row in edge_df.iterrows():
        G.add_edge(int(row.source), int(row.target))

    externality_dict = {}
    possible_files = ["externality.csv", "externality_matrix.csv"]
    ext_path = None
    for f in possible_files:
        p = os.path.join(folder, f)
        if os.path.exists(p):
            ext_path = p
            break
            
    if ext_path:
        df_ext = pd.read_csv(ext_path)
        for _, row in df_ext.iterrows():
            u, v = int(row['source']), int(row['target'])
            p = float(row.get('penalty', row.get('weight', 0.0)))
            if p != 0:
                # 保证顺序一致，方便字典查找
                k = tuple(sorted((u, v)))
                externality_dict[k] = p

    return G, externality_dict

def save_result(output_dir, selected_nodes, selected_edges, score, duration, selected_result_nodes):
    os.makedirs(output_dir, exist_ok=True)
    if not isinstance(selected_nodes, list): selected_nodes = list(selected_nodes)
    if not isinstance(selected_result_nodes, list): selected_result_nodes = list(selected_result_nodes)

    pd.DataFrame({'id': sorted(selected_nodes)}).to_csv(os.path.join(output_dir, 'sub_nodes.csv'), index=False)
    pd.DataFrame(selected_edges, columns=['source', 'target', 'cost']).to_csv(os.path.join(output_dir, 'sub_edges.csv'), index=False)
    with open(os.path.join(output_dir, 'score.txt'), 'w') as f: f.write(str(score))
    with open(os.path.join(output_dir, 'time.txt'), 'w') as f: f.write(f"{duration:.6f}")
    pd.DataFrame({'id': sorted(selected_result_nodes)}).to_csv(os.path.join(output_dir, 'result_nodes.csv'), index=False)
    with open(os.path.join(output_dir, 'result_nodes_count.txt'), 'w') as f: f.write(str(len(selected_result_nodes)))

# ==============================================================================
# --- 2. 核心解法：Lagrangian Iterative Min-Cut Solver (优化版) ---
# ==============================================================================

class IterativeMinCutSolver:
    def __init__(self, G, externality_dict, max_time_budget):
        self.G = G
        self.ext = externality_dict
        self.max_time_budget = max_time_budget
        
        self.nodes = list(G.nodes())
        self.result_nodes = [n for n in self.nodes if G.nodes[n].get('is_result')]
        
        # 预计算物理净值 (Value - Cost)
        self.base_weights = {}
        for n in self.nodes:
            val = G.nodes[n]['value'] if G.nodes[n].get('is_result') else 0.0
            cost = G.nodes[n]['cost']
            self.base_weights[n] = val - cost

        # --- 优化点1：预构建 Flow Graph ---
        self.source = 'SOURCE'
        self.sink = 'SINK'
        self.flow_G = nx.DiGraph()
        self._init_flow_graph_structure()

    def _init_flow_graph_structure(self):
        """
        初始化网络流图的静态结构：
        1. 节点与 Source/Sink 的连接（初始 Capacity=0）
        2. 原图中的依赖边（反向，Capacity=INF）
        """
        # 添加所有节点到 Source/Sink 的边 (占位)
        for n in self.nodes:
            self.flow_G.add_edge(self.source, n, capacity=0.0)
            self.flow_G.add_edge(n, self.sink, capacity=0.0)
            
        # 添加 DAG 依赖约束边 (如果选 v 必须选 u -> 割断 v 时 u 必须在 S 侧)
        # Max-Weight Closure 转化规则：原图 edge (u, v) -> 网络流 edge (v, u) cap=inf
        for u, v in self.G.edges():
            self.flow_G.add_edge(v, u, capacity=float('inf'))

    def _update_capacities(self, penalty_accumulator):
        """
        根据当前的惩罚项更新 Source/Sink 边的容量
        """
        current_weights = self.base_weights.copy()
        
        # 施加惩罚
        for n, pen in penalty_accumulator.items():
            current_weights[n] -= pen

        for n in self.nodes:
            w = current_weights[n]
            # 直接更新属性比重建图快得多
            if w > 0:
                self.flow_G[self.source][n]['capacity'] = w
                self.flow_G[n][self.sink]['capacity'] = 0.0
            else:
                self.flow_G[self.source][n]['capacity'] = 0.0
                self.flow_G[n][self.sink]['capacity'] = -w

    def _calculate_true_score(self, closure_nodes):
        if not closure_nodes: return 0.0
        closure_set = set(closure_nodes)
        
        # 基础分
        val = sum(self.G.nodes[n]['value'] for n in closure_set if self.G.nodes[n].get('is_result'))
        cost = sum(self.G.nodes[n]['cost'] for n in closure_set)
        
        # 外部性惩罚
        penalty = 0.0
        for (u, v), p in self.ext.items():
            if u in closure_set and v in closure_set:
                penalty += p
                
        return val - cost - penalty

    def solve(self, tracker: ConvergenceTracker):
        start_time = time.time()
        best_score = -float('inf')
        best_closure = []
        best_result_nodes = []
        
        node_penalty_accumulator = defaultdict(float)
        step_size = 1.0 
        iteration = 0
        
        # --- 优化点2：随机重启计数器 ---
        stagnation_counter = 0
        stagnation_limit = 100  # 如果100轮没提升，进行重启
        
        print(f"    [MinCut] Start Lagrangian Iteration (Budget={self.max_time_budget}s)...")
        
        while (time.time() - start_time) < self.max_time_budget:
            iteration += 1
            tracker.tick()
            
            # 1. 更新图权值 (O(V))
            self._update_capacities(node_penalty_accumulator)
            
            # 2. 最小割求解 (O(V^3) or faster)
            try:
                cut_val, partition = nx.minimum_cut(self.flow_G, self.source, self.sink)
                reachable, non_reachable = partition
                current_closure = list(reachable)
                if self.source in current_closure: 
                    current_closure.remove(self.source)
            except Exception as e:
                print(f"Solver Exception: {e}")
                break
                
            # 3. 评估真值
            true_score = self._calculate_true_score(current_closure)
            
            # 4. 更新最优解
            if true_score > best_score + 1e-9:
                best_score = true_score
                best_closure = current_closure
                
                # 找到更好解，衰减步长，重置停滞计数
                step_size = max(0.5, step_size * 0.9)
                stagnation_counter = 0
                
                tracker.update(best_score, round_num=iteration)
            else:
                stagnation_counter += 1
            
            # 5. 冲突反馈更新 (Lagrangian Update)
            closure_set = set(current_closure)
            conflicts_found = 0
            
            for (u, v), base_penalty in self.ext.items():
                if u in closure_set and v in closure_set:
                    conflicts_found += 1
                    increment = base_penalty * step_size
                    # 简单平分惩罚
                    node_penalty_accumulator[u] += increment / 2
                    node_penalty_accumulator[v] += increment / 2
            
            # 6. 步长与策略调整
            if conflicts_found == 0:
                # 处于可行域（无冲突），尝试贪婪地减少惩罚以吸纳更多点
                for k in node_penalty_accumulator:
                    node_penalty_accumulator[k] *= 0.8
                step_size = min(2.0, step_size * 1.2)
            else:
                # 处于冲突域，增加步长以加大惩罚力度
                step_size *= 0.95
                if step_size < 0.05: 
                    step_size = 1.5

            # 7. 随机重启 (防止陷入局部最优)
            if stagnation_counter >= stagnation_limit:
                # print(f"      [Restart] Stagnation detected at iter {iteration}. Resetting penalties.")
                node_penalty_accumulator.clear() # 清空惩罚
                step_size = 1.0 + random.uniform(-0.2, 0.2) # 随机步长
                stagnation_counter = 0
            
        # 结束处理
        tracker.finalize(self.max_time_budget)
        
        # 构造输出
        final_edges = []
        c_set = set(best_closure)
        for u in c_set:
            for v in self.G.successors(u):
                if v in c_set:
                    final_edges.append((u, v, 0.0))
        
        final_res_nodes = [n for n in best_closure if n in self.result_nodes]
        
        return best_closure, final_edges, best_score, final_res_nodes

# ==============================================================================
# --- 3. 批量执行入口 (统一接口) ---
# ==============================================================================

def run_experiment(root: str, algo_name: str, max_runtime: float):
    if not os.path.exists(root):
        print(f"Error: 根目录 '{root}' 未找到。")
        return

    for folder in os.listdir(root):
        path = os.path.join(root, folder)
        if not os.path.isdir(path): continue

        try:
            print(f"--- Processing [{folder}] Algorithm: {algo_name} ---")
            
            G, ext_dict = load_graph_data(path)
            
            result_dir = os.path.join(path, f"{algo_name}_result")
            os.makedirs(result_dir, exist_ok=True)
            
            tracker = ConvergenceTracker(
                algo_name, 
                folder, 
                save_path=os.path.join(result_dir, "convergence.csv"),
                log_interval=10.0
            )

            start = time.time()
            
            # 初始化 Solver
            solver = IterativeMinCutSolver(
                G, ext_dict, 
                max_time_budget=max_runtime
            )
            
            nodes, edges, score, res_nodes = solver.solve(tracker)
            
            elapsed = time.time() - start

            save_result(result_dir, nodes, edges, score, elapsed, res_nodes)
            print(f"[{folder}] Final Score={score:,.4f}, Results={len(res_nodes)}, Time={elapsed:.2f}s")
            print("=" * 60)

        except Exception as e:
            import traceback
            print(f"[{folder}] Failed: {e}")
            traceback.print_exc()

if __name__ == "__main__":
    # 参数设置
    MAX_RUNTIME = 3  # 根据数据规模调整，通常 100-300s 足够
    ALGO_NAME = "mf"   # "mf" represents MinFlow/MaxCut Iterative Relaxation
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str, default="../data/output")
    args = parser.parse_args()
    
    run_experiment(root=args.root_dir, algo_name=ALGO_NAME, max_runtime=MAX_RUNTIME)