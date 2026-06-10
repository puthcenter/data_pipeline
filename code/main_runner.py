import os
import subprocess
import pandas as pd
import numpy as np
from collections import defaultdict
import sys
import io
import time
import itertools
import re

# --- 配置 ---

METHODS = {
    "greedy": "greedy.py",
    "sghc": "sghc.py",
    #"hc": "hc.py",
    "sa": "sa.py",
    "sasa": "sasa.py",
    #"sa_full": "sa_full.py",
    "qp": "qp.py",
    "qpbo": "qpbo.py",
    # [新增] 加入 ILP 作为精确解方法
    #"ilp": "ilp.py"
}

# 定义难度级别，用于过滤和排序
DIFFICULTY_ORDER = ["EASY", "MEDIUM", "HARD", "medium"]

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.abspath(os.path.join(CODE_DIR, "..", "data", "output"))

# --- 改进的Logger类 ---

class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, 'w', encoding='utf-8', buffering=1)
        self.log.write(f"--- Main Runner Log Started at {time.strftime('%Y-%m-%d %H:%M:%S')} ---\n")
        self.log.flush()
        os.fsync(self.log.fileno())

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()
        try:
            os.fsync(self.log.fileno())
        except:
            pass

    def close(self):
        self.log.close()

# --- 脚本执行函数 ---

def run_method(name, script_name):
    print(f"\n" + "="*50)
    print(f"🔧 Running {name} ({script_name})...")
    print(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*50)
    sys.stdout.flush()
    
    script_path = os.path.join(CODE_DIR, script_name)
    
    if not os.path.exists(script_path):
        error_msg = f"❌ 文件不存在: {script_path}"
        print(error_msg)
        return name, False, error_msg
    
    try:
        # 注意：这里假设子脚本会自己去遍历 output 文件夹。
        # 如果子脚本不区分 dag_0 和 dag_EASY_0，它们可能会全部跑一遍。
        # main_runner 主要控制的是“统计结果”的筛选。
        process = subprocess.Popen(
            [sys.executable, "-u", script_path],
            cwd=CODE_DIR,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding='utf-8',
            bufsize=1,
            universal_newlines=True
        )
        
        output_lines = []
        while True:
            line = process.stdout.readline()
            if not line and process.poll() is not None:
                break
            if line:
                print(line, end='')
                output_lines.append(line)
                sys.stdout.flush()
        
        returncode = process.wait()
        
        if returncode == 0:
            print(f"\n✅ {name} finished successfully.")
            return name, True, None
        else:
            error_output = ''.join(output_lines)
            print(f"\n❌ {name} failed with exit code: {returncode}")
            return name, False, error_output
            
    except Exception as e:
        error_msg = f"❌ {name} crashed: {e}"
        print(error_msg)
        return name, False, str(e)

def run_all_methods():
    results = {}
    for name, script_name in METHODS.items():
        results[name] = run_method(name, script_name)[1]
        sys.stdout.flush()
    return results

# --- 辅助函数 ---

def calculate_jaccard(set_a, set_b):
    if not set_a and not set_b: return 0.0
    intersection = len(set_a.intersection(set_b))
    union = len(set_a.union(set_b))
    if union == 0: return 0.0
    return intersection / union

def get_difficulty_rank(category_name):
    """
    辅助排序：将 dag_EASY 映射为 0, dag_MEDIUM 映射为 1...
    如果找不到关键字，放在最后。
    """
    for idx, level in enumerate(DIFFICULTY_ORDER):
        if level in category_name:
            return idx
    return 999

# --- 读取结果 ---

def read_results():
    """
    读取 output 文件夹结果。
    [新增功能]: 自动跳过不包含 EASY/MEDIUM/HARD/EXTREME 的基础图文件夹。
    """
    final_results = defaultdict(lambda: defaultdict(list))
    convergence_data_frames = [] 
    overlap_stats = defaultdict(lambda: defaultdict(list))

    if not os.path.exists(DATA_DIR):
        print(f"❌ 数据目录不存在: {DATA_DIR}")
        return final_results, pd.DataFrame(), {}

    print("🔍 Scanning output directories...")

    sorted_folders = sorted(os.listdir(DATA_DIR))
    
    skipped_count = 0
    processed_count = 0

    for folder in sorted_folders:
        folder_path = os.path.join(DATA_DIR, folder)
        if not os.path.isdir(folder_path):
            continue
        
        # --- [关键修改] 过滤逻辑 ---
        # 检查文件夹名中是否包含难度关键词。
        # 假设命名格式如: dag_EASY_0, graph_HARD_1
        is_benchmark_task = False
        found_level = None
        for level in DIFFICULTY_ORDER:
            # 检查 level 是否作为独立的词存在 (避免比如 "EASYGOING" 被误判，虽然这里概率不大)
            if f"_{level}_" in folder: 
                is_benchmark_task = True
                found_level = level
                break
        
        if not is_benchmark_task:
            # print(f"  -> Skipping basic/unknown task: {folder}")
            skipped_count += 1
            continue
            
        processed_count += 1

        # 解析 Category (Group Key)
        # 例如: folder = dag_EASY_0 -> parts = ['dag', 'EASY', '0']
        # 我们希望 Group Key 是 'dag_EASY'，这样所有同难度的图会聚合在一起
        parts = folder.split("_")
        
        # 简单的取前缀逻辑：去除最后的 ID 数字
        try:
            int(parts[-1]) # 尝试把最后一部分转数字
            group_key = "_".join(parts[:-1]) # dag_EASY
        except ValueError:
            group_key = folder # 无法解析，保留原名

        # --- 以下逻辑保持不变 ---

        current_dataset_nodes = {} 

        for method in METHODS.keys():
            method_result_folder = f"{method}_result"
            method_path = os.path.join(folder_path, method_result_folder)
            
            # 1. Final Stats
            score_file = os.path.join(method_path, "score.txt")
            time_file = os.path.join(method_path, "time.txt")
            count_file = os.path.join(method_path, "result_nodes_count.txt")

            if os.path.exists(score_file) and os.path.exists(time_file) and os.path.exists(count_file):
                try:
                    with open(score_file) as f: score = float(f.read().strip())
                    with open(time_file) as f: time_used = float(f.read().strip())
                    with open(count_file) as f: count = int(f.read().strip())
                    final_results[group_key][method].append((score, time_used, count))
                except Exception:
                    pass
            
            # 2. Convergence
            # [关键修改] 如果是 ILP，则跳过读取收敛数据，因为我们只需要用它做 Baseline
            if method == "ilp":
                pass # Do nothing here, skip reading convergence.csv
            else:
                conv_file = os.path.join(method_path, "convergence.csv")
                if os.path.exists(conv_file):
                    try:
                        df = pd.read_csv(conv_file)
                        df['Category'] = group_key 
                        if 'Algorithm' not in df.columns:
                            df['Algorithm'] = method
                        convergence_data_frames.append(df)
                    except Exception:
                        pass

            # 3. Node Overlap
            nodes_file = os.path.join(method_path, "result_nodes.csv")
            if os.path.exists(nodes_file):
                try:
                    # 健壮读取
                    df_temp = pd.read_csv(nodes_file, header=None, nrows=5)
                    is_header = False
                    if not df_temp.empty:
                        try:
                            float(df_temp.iloc[0, 0])
                        except (ValueError, TypeError):
                            is_header = True
                    
                    if is_header:
                        nodes_df = pd.read_csv(nodes_file)
                        vals = nodes_df.iloc[:, 0]
                    else:
                        nodes_df = pd.read_csv(nodes_file, header=None)
                        vals = nodes_df.iloc[:, 0]
                        
                    nodes_list = pd.to_numeric(vals, errors='coerce').dropna().astype(int).tolist()
                    current_dataset_nodes[method] = set(nodes_list)
                except Exception:
                    pass

        # 计算 Jaccard
        if len(current_dataset_nodes) > 1:
            method_names = list(METHODS.keys())
            for m1, m2 in itertools.combinations_with_replacement(method_names, 2):
                if m1 in current_dataset_nodes and m2 in current_dataset_nodes:
                    jaccard_idx = calculate_jaccard(current_dataset_nodes[m1], current_dataset_nodes[m2])
                    overlap_stats[group_key][(m1, m2)].append(jaccard_idx)
                    if m1 != m2:
                        overlap_stats[group_key][(m2, m1)].append(jaccard_idx)

    print(f"✅ Processed {processed_count} benchmark folders. Skipped {skipped_count} basic folders.")

    if convergence_data_frames:
        all_convergence_df = pd.concat(convergence_data_frames, ignore_index=True)
    else:
        all_convergence_df = pd.DataFrame()

    return final_results, all_convergence_df, overlap_stats

def summarize_results(results):
    summary = []
    
    # --- [关键修改] 排序逻辑 ---
    # 我们希望按照 EASY -> MEDIUM -> HARD -> EXTREME 的顺序输出 CSV
    sorted_keys = sorted(results.keys(), key=lambda x: (get_difficulty_rank(x), x))

    for group_key in sorted_keys:
        for method in METHODS.keys():
            entries = results[group_key].get(method, [])
            if entries:
                avg_score = sum(x[0] for x in entries) / len(entries)
                avg_time = sum(x[1] for x in entries) / len(entries)
                avg_count = sum(x[2] for x in entries) / len(entries)
                num_runs = len(entries)
            else:
                avg_score, avg_time, avg_count, num_runs = None, None, None, 0
                
            summary.append({
                "Category": group_key,
                "Method": method,
                "Average_Score": avg_score,
                "Average_Time": avg_time,
                "Average_Result_Nodes": avg_count,
                "Num_Runs": num_runs
            })
    return pd.DataFrame(summary)

def summarize_overlaps(overlap_stats):
    overlap_summaries = []
    method_list = list(METHODS.keys())
    
    # 同样应用排序逻辑
    sorted_keys = sorted(overlap_stats.keys(), key=lambda x: (get_difficulty_rank(x), x))
    
    for category in sorted_keys:
        pairs = overlap_stats[category]
        matrix = pd.DataFrame(index=method_list, columns=method_list, dtype=float)
        
        for m1 in method_list:
            for m2 in method_list:
                scores = pairs.get((m1, m2), [])
                if scores:
                    avg_jaccard = sum(scores) / len(scores)
                    matrix.loc[m1, m2] = avg_jaccard
                else:
                    matrix.loc[m1, m2] = np.nan
        
        matrix_file = os.path.join(CODE_DIR, f"overlap_matrix_{category}.csv")
        matrix.to_csv(matrix_file)

        for m1 in method_list:
            for m2 in method_list:
                val = matrix.loc[m1, m2]
                if not pd.isna(val):
                    overlap_summaries.append({
                        "Category": category,
                        "Method_A": m1,
                        "Method_B": m2,
                        "Avg_Jaccard_Similarity": val
                    })
                    
    return pd.DataFrame(overlap_summaries)

# --- 主函数 ---

def main():
    print(f"--- Main Runner Started ---")
    sys.stdout.flush()
    
    # 1. 运行算法
    #run_all_methods()
    
    print("\n" + "="*50)
    print("📊 正在读取和汇总 Benchmark 结果...")
    print("   (只处理 EASY, MEDIUM, HARD, EXTREME 版本)")
    print("="*50)
    sys.stdout.flush()
    
    # 2. 读取
    final_results_dict, convergence_df, overlap_stats = read_results()
    
    # 3. Performance Summary
    summary_df = summarize_results(final_results_dict)
    print("\n📊 性能汇总表 (按难度排序):")
    if summary_df.empty:
        print("  (汇总表为空 - 未找到符合条件的 benchmark 结果)")
    else:
        # 只打印前几行预览
        print(summary_df.head(20).to_string())
    
    summary_path = os.path.join(CODE_DIR, "summary.csv")
    try:
        summary_df.to_csv(summary_path, index=False)
        print(f"\n✅ 性能汇总 CSV 已保存到: {summary_path}")
    except Exception as e:
        print(f"\n❌ 保存 CSV 失败: {e}")

    # 4. Convergence Summary
    conv_path = os.path.join(CODE_DIR, "convergence_summary.csv")
    if not convergence_df.empty:
        try:
            convergence_df.to_csv(conv_path, index=False)
            print(f"✅ 收敛历史总表 CSV 已保存到: {conv_path} (Rows: {len(convergence_df)})")
        except Exception as e:
            print(f"❌ 保存收敛表失败: {e}")
    else:
        print("⚠️ 未找到任何收敛数据 (忽略 ILP)。")

    # 5. Overlap Summary
    print("\n🔗 节点重合度分析:")
    overlap_df = summarize_overlaps(overlap_stats)
    overlap_path = os.path.join(CODE_DIR, "overlap_summary.csv")
    if not overlap_df.empty:
        try:
            overlap_df.to_csv(overlap_path, index=False)
            print(f"✅ 节点重合度汇总 CSV 已保存到: {overlap_path}")
        except Exception as e:
            print(f"❌ 保存重合度表失败: {e}")

    sys.stdout.flush()

if __name__ == "__main__":
    record_file_path = os.path.join(CODE_DIR, "record.txt")
    sys.stdout = Logger(record_file_path)
    sys.stderr = sys.stdout

    print(f"--- Main Runner (Benchmark Edition) ---")
    print(f"--- Time: {time.strftime('%Y-%m-%d %H:%M:%S')} ---")
    
    try:
        main()
        
        # 绘图
        plot_script_path = os.path.join(CODE_DIR, "plot_summary.py")
        if os.path.exists(plot_script_path):
            print("\n" + "="*50)
            print("📊 正在调用绘图脚本...")
            subprocess.run([sys.executable, plot_script_path], cwd=CODE_DIR)
            print("✅ 绘图脚本执行完毕")
        
    except Exception as e:
        print(f"\n❌ Fatal Error: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        print(f"\n--- End: {time.strftime('%Y-%m-%d %H:%M:%S')} ---")
        if isinstance(sys.stdout, Logger):
            sys.stdout.close()
            sys.stdout = sys.stdout.terminal