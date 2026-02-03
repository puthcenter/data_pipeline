import os
import time
import pandas as pd
import networkx as nx
import argparse
from typing import List, Dict, Set
import pulp

# ==============================================================================
# --- 1. Data Loading ---
# ==============================================================================
def _safe_float(x):
    try: return float(x)
    except: return 0.0

def load_all_data(folder: str):
    """
    Loads graph structure (nodes, edges) and externality constraints from CSV files.
    """
    node_df = pd.read_csv(os.path.join(folder, "nodes.csv"))
    edge_df = pd.read_csv(os.path.join(folder, "edges.csv"))
    G = nx.DiGraph()
    node_costs = {}
    node_vals = {}
    
    # Load Nodes
    for _, row in node_df.iterrows():
        nid = int(row["id"])
        # Parse cost and value
        cost = _safe_float(row.get("incoming_cost", row.get("cost", 0.0)))
        val = _safe_float(row.get("value", 0.0))
        is_res = str(row.get("is_result", False)).lower() == 'true'
        
        G.add_node(nid, value=val, is_result=is_res, cost=cost)
        node_costs[nid] = cost
        node_vals[nid] = val

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
    return G, node_costs, node_vals, edges_data, externality_dict

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
# --- 2. Integer Linear Programming (ILP) Solver ---
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
    """
    Formulates and solves the MwcpNe problem using an exact Integer Linear Programming solver.
    """
    os.makedirs(result_dir, exist_ok=True)
    start_time = time.time()
    
    # Identify sets of nodes
    all_nodes = list(G.nodes())
    result_candidates = [n for n, d in G.nodes(data=True) if d.get('is_result')]
    
    print(f"    [ILP] Building Model for {dataset_name}...")
    print(f"          Nodes: {len(all_nodes)}, Result Candidates: {len(result_candidates)}")
    
    # --- Problem Definition ---
    prob = pulp.LpProblem("Node_Selection_Optimization", pulp.LpMaximize)
    
    # --- Decision Variables ---
    
    # y[j]: Binary, 1 if node j is included in the closure (incurs Cost), 0 otherwise.
    y = pulp.LpVariable.dicts("y", all_nodes, cat='Binary')
    
    # x[i]: Binary, 1 if result node i is activated (gains Value), 0 otherwise.
    # Defined only for 'result' nodes.
    x = pulp.LpVariable.dicts("x", result_candidates, cat='Binary')
    
    # z[(u,v)]: Binary, 1 if both u and v are activated (incurs Penalty), 0 otherwise.
    # Linearization variable for the interaction term x[u] * x[v].
    z_keys = list(externality_dict.keys())
    z = pulp.LpVariable.dicts("z", z_keys, cat='Binary')
    
    # --- Objective Function ---
    # Maximize: sum(Value * x) - sum(Cost * y) - sum(Penalty * z)
    obj_terms = []
    
    # 1. Rewards (Positive Value)
    for i in result_candidates:
        val = node_vals.get(i, 0.0)
        if val != 0:
            obj_terms.append(val * x[i])
            
    # 2. Costs (Negative Cost)
    for j in all_nodes:
        cost = node_costs.get(j, 0.0)
        if cost != 0:
            obj_terms.append(-1 * cost * y[j]) 
            
    # 3. Penalties (Negative Externality)
    for (u, v), penalty in externality_dict.items():
        if penalty > 0:
            obj_terms.append(-1 * penalty * z[(u,v)])
            
    prob += pulp.lpSum(obj_terms)
    
    # --- Constraints ---
    
    # 1. Dependency Constraint:
    # If a result node x[i] is activated, it must be present in the closure y[i].
    for i in result_candidates:
        prob += x[i] <= y[i], f"Res_Dependency_{i}"
        
    # 2. Topology Constraint (Parent Closure Property):
    # For edge u -> v (u is parent, v is child), if child v is selected, parent u must be selected.
    # Implication: y[v] <= y[u]
    for u, v, _ in edges_data:
        # Note: G.edges are source -> target. In dependency logic, target depends on source.
        if u in y and v in y:
            prob += y[v] <= y[u], f"Edge_{u}_{v}"
            
    # 3. Externality Linearization Constraint:
    # z[(u,v)] must be 1 if both x[u] and x[v] are 1.
    # Standard linearization: z >= x_u + x_v - 1
    # Note: We do not need z <= x_u or z <= x_v because the objective function minimizes z (via negative penalty).
    for (u, v) in externality_dict:
        if u in x and v in x:
            prob += z[(u,v)] >= x[u] + x[v] - 1, f"Ext_{u}_{v}"
        else:
            # Fallback: if u or v is not a result candidate (should not happen in cleaned data), z is 0.
            prob += z[(u,v)] == 0

    # --- Solving ---
    print(f"    [ILP] Solving... (Allowing 0.1% optimality gap for efficiency)")
    
    # Configure CBC Solver
    # msg=1: Enable solver logging
    # gapRel=0.001: Stop if within 0.1% of the optimal solution
    # threads=8: Parallel execution (adjust based on CPU cores)
    solver = pulp.PULP_CBC_CMD(
        msg=1, 
        gapRel=0.001, 
        threads=8      
    )
    
    prob.solve(solver) 
    
    status = pulp.LpStatus[prob.status]
    print(f"    [ILP] Status: {status}")
    
    # --- Result Extraction ---
    end_time = time.time()
    duration = end_time - start_time
    final_score = pulp.value(prob.objective)
    
    # Extract Activated Result Nodes
    final_res_nodes = set()
    for i in result_candidates:
        if pulp.value(x[i]) and pulp.value(x[i]) > 0.5:
            final_res_nodes.add(i)
            
    # Extract Closure Nodes
    final_closure_nodes = set()
    for j in all_nodes:
        if pulp.value(y[j]) and pulp.value(y[j]) > 0.5:
            final_closure_nodes.add(j)
            
    # Extract Edges within Closure
    final_edges = [(u, v, c) for u, v, c in edges_data 
                   if u in final_closure_nodes and v in final_closure_nodes]
    
    # Record Convergence (Single data point for exact methods)
    convergence_record = [{
        "Algorithm": "ilp",
        "Dataset": dataset_name,
        "Time_Elapsed": duration,
        "Best_Value": final_score
    }]
    pd.DataFrame(convergence_record).to_csv(os.path.join(result_dir, "convergence.csv"), index=False)
    
    return final_closure_nodes, final_edges, final_score, final_res_nodes

# ==============================================================================
# --- Execution Entry Point ---
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