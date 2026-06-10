import os
import csv
import json
import random
import logging
import re
import numpy as np
import networkx as nx
from collections import defaultdict, deque
from dataclasses import dataclass, asdict
from itertools import combinations

# ==========================================
# 全局处理配置 (ETL Configuration)
# ==========================================

DATASET_VERSIONS = 3  
ACTIVE_PROFILES = ["MEDIUM"]
DATA_OUTPUT_DIR = "output"

@dataclass
class TopologyParams:
    """拓扑标准化参数：确保 50w 节点规模下的逻辑自洽"""
    target_capacity: int = 500000   
    sample_size: int = 25000        
    max_hierarchy: int = 40         
    
    # 形状因子
    cost_width_factor: int = 300  
    attenuation_rate: float = 0.10       
    
    # 连通性参数
    link_noise_prob: float = 0.03   
    density_target: int = 6         
    seed: int = None         

BASE_ECONOMICS = {
    "UNIT_COST_MEAN": 5.0,
    "UNIT_COST_STD": 3.0,
    "UNIT_COST_FLOOR": 1.0,
    "SHALLOW_ROI": 0.90,     
    "DEEP_ROI": 0.20,        
    "API_COST_MEAN": 5.0,    
    "API_COST_STD": 15.0,    
    "API_MARGIN_MEAN": 0.75, 
    "API_MARGIN_STD": 0.35,  
    "VALUATION_NOISE": 0.10, 
}

SCENARIO_PROFILES = {
    "MEDIUM": { 
        "ROI_TARGET": 0.8,
        "RISK_PENALTY_RATIO": 0.8, 
        "STRUCTURAL_RISK_SHARE": 0.5,
        "CONFLICT_INTENSITY": 0.7,     
        "NOISE_INTENSITY": 0.8,
        "CROSS_TIER_CONFLICT_MULTIPLIER": 2.0
    }
}

# 敏感度分析实验设计
EXPERIMENTS = [
    ("depth", "low", 101, {"max_hierarchy": 15}, {}),
    ("depth", "medium", 102, {"max_hierarchy": 40}, {}),
    ("depth", "high", 103, {"max_hierarchy": 80}, {}), 
    
    ("density", "low", 201, {"density_target": 2}, {}),
    ("density", "medium", 202, {"density_target": 6}, {}),
    ("density", "high", 203, {"density_target": 12}, {}),
    
    ("products", "low", 301, {"sample_size": 5000}, {}),
    ("products", "medium", 302, {"sample_size": 25000}, {}), 
    ("products", "high", 303, {"sample_size": 100000}, {}),
    
    ("conflicts", "low", 401, {}, {"RISK_PENALTY_RATIO": 0.2}),
    ("conflicts", "medium", 402, {}, {"RISK_PENALTY_RATIO": 0.8}),
    ("conflicts", "high", 403, {}, {"RISK_PENALTY_RATIO": 3.0}),
]

def setup_logger():
    logging.basicConfig(
        level=logging.INFO, 
        format='[%(levelname)s] %(asctime)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("sensitivity_pipeline.log", mode='w', encoding='utf-8')
        ]
    )

# ==========================================
# 核心模块: 拓扑流形生成 (Manifold Generation)
# ==========================================

def _repair_connectivity(G):
    components = list(nx.weakly_connected_components(G))
    if len(components) <= 1: return
    components.sort(key=len, reverse=True)
    giant = list(components[0])
    giant_roots = [n for n in giant if G.nodes[n].get('layer') == 0] or giant[:10]
    
    for c in components[1:]:
        c_list = list(c)
        sub_roots = [n for n in c_list if G.nodes[n].get('layer') == 0] or [n for n in c_list if G.in_degree(n) == 0]
        if sub_roots:
            G.add_edge(random.choice(giant_roots), random.choice(sub_roots))

def build_reference_manifold(cfg: TopologyParams):
    if cfg.seed is not None:
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)

    G = nx.DiGraph()
    # 动态适应：确保信道宽度不低于密度目标，防止密度参数失效
    base_width = int(cfg.cost_width_factor / cfg.max_hierarchy)
    channel_width = max(cfg.density_target + 2, base_width) 
    
    r = 1.0 - cfg.attenuation_rate
    geom_sum = (1 - r**(cfg.max_hierarchy + 1)) / (1 - r) if abs(cfg.attenuation_rate) > 1e-6 else cfg.max_hierarchy + 1
    init_channels = max(1, int((cfg.target_capacity / channel_width) / geom_sum))
    
    current_channels = defaultdict(list)
    current_node_id = 0
    
    # Layer 0
    for _ in range(int(init_channels * channel_width)):
        cid = random.randint(0, init_channels - 1)
        G.add_node(current_node_id, is_result=False, layer=0, pos=(cid+0.5)/init_channels, channel=cid)
        current_channels[cid].append(current_node_id)
        current_node_id += 1
    
    prev_channels = current_channels
    available_endpoints = []
    
    for l in range(1, cfg.max_hierarchy + 1):
        next_channels = defaultdict(list)
        active_cids = [cid for cid in prev_channels if random.random() > cfg.attenuation_rate]
        dropped_cids = [cid for cid in prev_channels if cid not in active_cids]
        
        for cid in dropped_cids: available_endpoints.extend(prev_channels[cid])
            
        for cid in active_cids:
            width = max(1, int(np.random.normal(channel_width, channel_width*0.1)))
            for _ in range(width):
                nid = current_node_id
                G.add_node(nid, is_result=False, layer=l, pos=(cid+0.5)/init_channels, channel=cid)
                next_channels[cid].append(nid)
                current_node_id += 1
                
                # 连通性注入 (严格遵循信道隔离逻辑)
                parents = set()
                target_d = max(1, int(random.gauss(cfg.density_target, cfg.density_target*0.1)))
                target_d = min(target_d, len(prev_channels[cid]) + 2)
                
                attempts = 0
                while len(parents) < target_d and attempts < target_d * 3:
                    attempts += 1
                    if random.random() < cfg.link_noise_prob:
                        neighbor_cid = (cid + random.choice([-1, 1])) % init_channels
                        pool = prev_channels.get(neighbor_cid, [])
                    else:
                        pool = prev_channels[cid]
                    if pool: parents.add(random.choice(pool))
                
                if not parents and prev_channels[cid]: parents.add(random.choice(prev_channels[cid]))
                for p in parents: G.add_edge(p, nid, cost=0.0)
        
        prev_channels = next_channels
        if not prev_channels: break

    for nodes in prev_channels.values(): available_endpoints.extend(nodes)
    results = random.sample(available_endpoints, min(len(available_endpoints), cfg.sample_size))
    for n in results: G.nodes[n]['is_result'] = True

    _repair_connectivity(G)
    logging.info(f"    > Built Manifold: Nodes={G.number_of_nodes()}, Edges={G.number_of_edges()}")
    return G

# ==========================================
# 核心模块: 三阶段价值与外部性生成 (Economics)
# ==========================================

def enrich_node_attributes(G, config):
    """还原完全版三阶段 ROI 赋值逻辑"""
    result_nodes = [n for n in G.nodes() if G.nodes[n].get('is_result', False)]
    if not result_nodes: return G, [], {}, 0.0

    layers = {n: G.nodes[n].get('layer', 0) for n in result_nodes}
    max_layer, min_layer = max(layers.values()), min(layers.values())
    api_nodes_set = set([n for n in result_nodes if layers[n] >= max_layer])

    for n in G.nodes():
        G.nodes[n]['is_api'] = (n in api_nodes_set)
        if n in api_nodes_set:
            c = random.normalvariate(config["API_COST_MEAN"], config["API_COST_STD"])
        else:
            c = random.normalvariate(config["UNIT_COST_MEAN"], config["UNIT_COST_STD"])
        G.nodes[n]['incoming_cost'] = max(config["UNIT_COST_FLOOR"], c)

    total_physical_cost = sum(G.nodes[n]['incoming_cost'] for n in G.nodes())
    
    # 闭包成本计算 (由于信道隔离，此处计算量受控)
    closure_costs = {}
    for r in result_nodes:
        anc = nx.ancestors(G, r)
        closure_costs[r] = sum(G.nodes[a]['incoming_cost'] for a in anc) + G.nodes[r]['incoming_cost']

    target_revenue = total_physical_cost * (1 + config["ROI_TARGET"])
    weights = {}
    total_weight = 0.0
    
    for r in result_nodes:
        if r in api_nodes_set:
            roi_expectation = max(-0.6, random.normalvariate(config["API_MARGIN_MEAN"], config["API_MARGIN_STD"]))
        else:
            depth_ratio = (layers[r] - min_layer) / (max_layer - min_layer) if max_layer > min_layer else 0
            base_roi = config["SHALLOW_ROI"] - depth_ratio * (config["SHALLOW_ROI"] - config["DEEP_ROI"])
            roi_expectation = random.normalvariate(base_roi, config["VALUATION_NOISE"])
        
        w = closure_costs[r] * max(0.01, (1 + roi_expectation))
        weights[r] = w
        total_weight += w

    factor = target_revenue / total_weight if total_weight > 0 else 0
    for r in result_nodes: G.nodes[r]['value'] = weights[r] * factor

    return G, result_nodes, closure_costs, total_physical_cost

def compute_distribution_threshold(data):
    if not data: return 0
    data = np.array(data)
    c1, c2 = np.min(data), np.max(data)
    for _ in range(15):
        d1, d2 = np.abs(data - c1), np.abs(data - c2)
        g1, g2 = data[d1 <= d2], data[d1 > d2]
        if len(g1) > 0: c1 = np.mean(g1)
        if len(g2) > 0: c2 = np.mean(g2)
    return (c1 + c2) / 2

def extract_constraint_relationships(G, result_nodes, closure_costs, total_cost, config):
    """还原完全版三重外部性冲突逻辑"""
    constraints_log = []
    costs = list(closure_costs.values())
    if not costs: return []
    
    threshold = compute_distribution_threshold(costs)
    heavy_nodes = [n for n in result_nodes if closure_costs[n] >= threshold]
    light_nodes = [n for n in result_nodes if closure_costs[n] < threshold]
    if not heavy_nodes: heavy_nodes = result_nodes

    total_budget = total_cost * config["RISK_PENALTY_RATIO"]
    clique_budget = total_budget * config["STRUCTURAL_RISK_SHARE"]
    noise_budget = total_budget - clique_budget
    
    # 1. 跨阶层冲突 (API 节点对浅层节点的压制/冲突)
    api_nodes = [n for n in result_nodes if G.nodes[n]['is_api']]
    shallow_threshold = np.percentile([G.nodes[x]['layer'] for x in result_nodes], 30) if result_nodes else 0
    shallow_nodes = [n for n in result_nodes if not G.nodes[n]['is_api'] and G.nodes[n]['layer'] <= shallow_threshold]

    for a_node in api_nodes:
        if not shallow_nodes: break
        targets = random.sample(shallow_nodes, min(len(shallow_nodes), random.randint(2, 5)))
        for s_node in targets:
            penalty = 0.08 * (G.nodes[a_node]['value'] + G.nodes[s_node]['value']) * random.uniform(1.0, config["CROSS_TIER_CONFLICT_MULTIPLIER"])
            constraints_log.extend([{'source': a_node, 'target': s_node, 'penalty': round(penalty, 4)},
                                    {'source': s_node, 'target': a_node, 'penalty': round(penalty, 4)}])
            noise_budget -= penalty 

    # 2. 结构化同级冲突 (Cliques)
    curr_clique_spent = 0
    pool = list(result_nodes)
    random.shuffle(pool)
    while curr_clique_spent < clique_budget and len(pool) >= 3:
        size = random.randint(3, 5)
        clique_nodes = [pool.pop() for _ in range(min(size, len(pool)))]
        avg_val = np.mean([G.nodes[n]['value'] for n in clique_nodes])
        penalty = avg_val * config["CONFLICT_INTENSITY"]
        for u, v in combinations(clique_nodes, 2):
            constraints_log.extend([{'source': u, 'target': v, 'penalty': round(penalty, 4)},
                                    {'source': v, 'target': u, 'penalty': round(penalty, 4)}])
            curr_clique_spent += penalty

    # 3. 随机噪音风险
    curr_noise_spent = 0
    unit_penalty = (np.mean([G.nodes[n]['value'] for n in light_nodes]) if light_nodes else 10.0) * config["NOISE_INTENSITY"]
    iters = 0
    while curr_noise_spent < max(0, noise_budget) and iters < len(result_nodes) * 5:
        iters += 1
        h, l = random.choice(heavy_nodes), random.choice(light_nodes if light_nodes else heavy_nodes)
        if h == l: continue
        p = unit_penalty * random.uniform(0.8, 1.2)
        constraints_log.extend([{'source': h, 'target': l, 'penalty': round(p, 4)},
                                {'source': l, 'target': h, 'penalty': round(p, 4)}])
        curr_noise_spent += p
    
    return constraints_log

# ==========================================
# 序列化与主流程 (Serialization & Execution)
# ==========================================

def serialize_dataset(G, constraints_data, folder_path, metadata):
    os.makedirs(folder_path, exist_ok=True)
    
    # nodes.csv
    with open(os.path.join(folder_path, "nodes.csv"), 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(["id", "is_result", "value", "incoming_cost", "layer", "pos", "channel", "is_api"])
        for n, d in G.nodes(data=True):
            w.writerow([n, d.get('is_result', False), f"{d.get('value', 0):.4f}", f"{d.get('incoming_cost', 0):.4f}",
                        d.get('layer'), f"{d.get('pos', 0):.4f}", d.get('channel'), d.get('is_api', False)])
            
    # edges.csv
    with open(os.path.join(folder_path, "edges.csv"), 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(["source", "target", "cost"])
        for u, v, d in G.edges(data=True):
            w.writerow([u, v, f"{d.get('cost', 0):.4f}"])
            
    # externality.csv
    with open(os.path.join(folder_path, "externality.csv"), 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(["source", "target", "penalty"])
        for item in constraints_data:
            w.writerow([item['source'], item['target'], f"{item['penalty']:.4f}"])
            
    # meta.json (还原详细版参数记录)
    with open(os.path.join(folder_path, "meta.json"), 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=4)

def run_sensitivity_pipeline():
    setup_logger()
    logging.info("Starting Full-Scale Sensitivity Pipeline...")

    profile_name = ACTIVE_PROFILES[0]  
    base_eco = BASE_ECONOMICS.copy()
    base_eco.update(SCENARIO_PROFILES[profile_name])

    for dim, level, file_id, topo_over, eco_over in EXPERIMENTS:
        for v_idx in range(DATASET_VERSIONS):
            logging.info(f"Processing: {dim}-{level} | Version: {v_idx}")
            
            # 1. 参数准备
            tp = TopologyParams(seed=hash(f"{file_id}_{v_idx}") % (2**32))
            for k, v in topo_over.items(): setattr(tp, k, v)
                
            proc_config = base_eco.copy()
            proc_config.update(eco_over)

            try:
                # 2. 生成流程 (拓扑 -> 价值 -> 外部性)
                G = build_reference_manifold(tp)
                G, res_nodes, clos_costs, total_cost = enrich_node_attributes(G, proc_config)
                constraints = extract_constraint_relationships(G, res_nodes, clos_costs, total_cost, proc_config)
                
                # 3. 还原完整版 Meta 参数
                final_meta = {
                    "etl_info": {
                        "source_id": file_id,
                        "original_file": f"SYNTHETIC_REF_{file_id}.txt",
                        "raw_node_count": tp.target_capacity,
                        "mapped_node_count": G.number_of_nodes(),
                        "standardized_edge_count": G.number_of_edges(),
                        "topology_params": asdict(tp)
                    },
                    "processing_context": {
                        "profile": profile_name,
                        "version_index": v_idx,
                        "parameters": proc_config,
                        "sensitivity_dim": dim,
                        "sensitivity_level": level
                    }
                }
                
                # 4. 输出存储
                folder_name = f"dag{file_id}_{profile_name}_{v_idx}"
                serialize_dataset(G, constraints, os.path.join(DATA_OUTPUT_DIR, folder_name), final_meta)
                
            except Exception as e:
                logging.error(f"Error on {file_id}_{v_idx}: {e}")

if __name__ == "__main__":
    run_sensitivity_pipeline()