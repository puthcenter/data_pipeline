import os
import time
import pandas as pd
import networkx as nx
import argparse
from typing import List, Dict, Set
import pulp

# ==============================================================================
# --- 1. 数据加载 (保持原样) ---
# ==============================================================================
def _safe_float(x):
    try: return float(x)
    except: return 0.0

def load_all_data(folder: str):
    node_df = pd.read_csv(os.path.join(folder, "nodes.csv"))
    edge_df = pd.read_csv(os.path.join(folder, "edges.csv"))
    G = nx.DiGraph()
    node_costs = {}
    node_vals = {}
    
    for _, row in node_df.iterrows():
        nid = int(row["id"])
        # 统一处理成本和价值
        cost = _safe_float(row.get("incoming_cost", row.get("cost", 0.0)))
        val = _safe_float(row.get("value", 0.0))
        is_res = str(row.get("is_result", False)).lower() == 'true'
        
        G.add_node(nid, value=val, is_result=is_res, cost=cost)
        node_costs[nid] = cost
        node_vals[nid] = val

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
    return G, node_costs, node_vals, edges_data, externality_dict

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
# --- 2. ILP 精确求解器 ---
# ==============================================================================
def run_ilp_solver(
    G: nx.DiGraph, 
    node_costs: Dict,
    node_vals: Dict,
    edges_data: List, 
    externality_dict: Dict, 
    dataset_name: str,
    result_dir: str
):
    os.makedirs(result_dir, exist_ok=True)
    start_time = time.time()
    
    # 识别所有节点和结果节点
    all_nodes = list(G.nodes())
    result_candidates = [n for n, d in G.nodes(data=True) if d.get('is_result')]
    
    print(f"    [ILP] Building Model for {dataset_name}...")
    print(f"          Nodes: {len(all_nodes)}, Result Candidates: {len(result_candidates)}")
    
    # --- 定义问题 ---
    prob = pulp.LpProblem("Node_Selection_Optimization", pulp.LpMaximize)
    
    # --- 决策变量 ---
    # y[j]: 节点 j 是否在闭包中 (被选中付费)
    y = pulp.LpVariable.dicts("y", all_nodes, cat='Binary')
    
    # x[i]: 结果节点 i 是否被“激活” (获得收益)
    # 只有 is_result=True 的节点才有对应的 x 变量
    x = pulp.LpVariable.dicts("x", result_candidates, cat='Binary')
    
    # z[(u,v)]: 外部性线性化变量
    # 仅针对 externality_dict 中存在的对
    z_keys = list(externality_dict.keys())
    z = pulp.LpVariable.dicts("z", z_keys, cat='Binary')
    
    # --- 目标函数 ---
    # Max: sum(Value * x) - sum(Cost * y) - sum(Penalty * z)
    obj_terms = []
    
    # 1. 收益 (仅针对 Result Nodes)
    for i in result_candidates:
        val = node_vals.get(i, 0.0)
        if val != 0:
            obj_terms.append(val * x[i])
            
    # 2. 成本 (针对所有节点)
    for j in all_nodes:
        cost = node_costs.get(j, 0.0)
        if cost != 0:
            obj_terms.append(-1 * cost * y[j]) # 减去成本
            
    # 3. 外部性惩罚
    for (u, v), penalty in externality_dict.items():
        if penalty > 0:
            obj_terms.append(-1 * penalty * z[(u,v)])
            
    prob += pulp.lpSum(obj_terms)
    
    # --- 约束条件 ---
    
    # 1. 结果节点依赖约束: 如果选中结果 x[i]，则节点必须存在于闭包 y[i]
    for i in result_candidates:
        prob += x[i] <= y[i], f"Res_Dependency_{i}"
        
    # 2. 图的父闭包约束 (Parent Closure):
    # 对于边 u -> v (u是父, v是子), 如果子 v 被选中(y[v]=1), 则父 u 必须被选中(y[u]=1)
    # 即: y[v] <= y[u]
    for u, v, _ in edges_data:
        # 注意: G.edges 是 (source, target)，即 source -> target
        # 在依赖关系中，target 依赖 source。
        if u in y and v in y:
            prob += y[v] <= y[u], f"Edge_{u}_{v}"
            
    # 3. 外部性线性化约束
    # z_uv = 1 当且仅当 x_u = 1 AND x_v = 1
    # 由于目标函数是减去 penalty * z (即希望 z 尽可能小),
    # 我们只需要约束 z 的下界: z >= x_u + x_v - 1
    for (u, v) in externality_dict:
        if u in x and v in x:
            prob += z[(u,v)] >= x[u] + x[v] - 1, f"Ext_{u}_{v}"
        else:
            # 如果 u 或 v 不是 result node (理论上不应发生，视数据清理情况)，强制 z=0
            prob += z[(u,v)] == 0

    
    # --- 求解 (优化版) ---
    print(f"    [ILP] Solving with SCIP... (Allowing 0.1% optimality gap for speed)")
    
    # 使用 SCIP 求解器
    solver = pulp.SCIP_CMD(
        msg=1, # 显示求解日志
        options=[
            'set limits gap 0.001',       # 等同于 gapRel=0.001: 0.1% 的误差容忍度
            'set parallel maxnthreads 8'  # 设置最大并行线程数为 8
        ]
    )
    
    prob.solve(solver)
    
    status = pulp.LpStatus[prob.status]
    print(f"    [ILP] Status: {status}")
    
    # --- 结果提取 ---
    end_time = time.time()
    duration = end_time - start_time
    final_score = pulp.value(prob.objective)
    
    # 提取被选中的 Result Nodes
    final_res_nodes = set()
    for i in result_candidates:
        if pulp.value(x[i]) and pulp.value(x[i]) > 0.5:
            final_res_nodes.add(i)
            
    # 提取闭包节点 (y=1)
    final_closure_nodes = set()
    for j in all_nodes:
        if pulp.value(y[j]) and pulp.value(y[j]) > 0.5:
            final_closure_nodes.add(j)
            
    # 提取边
    final_edges = [(u, v, c) for u, v, c in edges_data 
                   if u in final_closure_nodes and v in final_closure_nodes]
    
    # 记录单个收敛点 (即最终结果)
    # 因为是精确解，这里只记录最后的时间点
    convergence_record = [{
        "Algorithm": "ilp",
        "Dataset": dataset_name,
        "Time_Elapsed": duration,
        "Best_Value": final_score
    }]
    pd.DataFrame(convergence_record).to_csv(os.path.join(result_dir, "convergence.csv"), index=False)
    
    return final_closure_nodes, final_edges, final_score, final_res_nodes

# ==============================================================================
# --- 执行入口 ---
# ==============================================================================
def run_experiment(root_dir, algo_name="ilp"):
    if not os.path.exists(root_dir): return
    
    for folder in os.listdir(root_dir):
        path = os.path.join(root_dir, folder)
        if not os.path.isdir(path): continue
        if "dag" not in folder: continue 

        try:
            print(f"--- Processing [{folder}] (Algo: {algo_name}) ---")
            
            G, nc, nv, ed, ext = load_all_data(path)
            
            result_dir = os.path.join(path, f"{algo_name}_result")
            start_real = time.time()
            
            sn, se, score, res_nodes = run_ilp_solver(
                G, nc, nv, ed, ext,
                dataset_name=folder, 
                result_dir=result_dir
            )
            
            duration = time.time() - start_real
            save_result(result_dir, sn, se, score, duration, res_nodes)
            print(f"[{folder}] Final: {score:,.2f}, Time: {duration:.2f}s")
            print("="*60)
            
        except Exception as e:
            import traceback; traceback.print_exc()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str, default="../data/output")
    args = parser.parse_args()
    
    run_experiment(args.root_dir, "ilp")