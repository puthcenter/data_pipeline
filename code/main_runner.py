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

# --- Configuration ---

# 1. Execution Configuration: Define scripts to be executed.
# Format: {"Task Name": "Script Filename"}
# Note: 'sasax_combined' runs a single script that generates multiple result sets.
SCRIPTS_TO_RUN = {
    "greedy": "greedy.py",
    "nghc": "nghc.py",
    "sghc": "sghc.py",
    "sa": "sa.py",
    "sasa": "sasa.py",
    "qp": "qp.py",
    "qpbo": "qpbo.py",
    "ilp": "ilp.py",
    "sasax_combined": "sasax.py" 
}

# 2. Analysis Configuration: Define method names for result statistics.
# These names must correspond to the output folders named in the format: {name}_result
# E.g., "sasa_synergy_only" corresponds to the folder "sasa_synergy_only_result"
METHODS_TO_ANALYZE = [
    "greedy",
    "sghc",
    "nghc",
    "sa",
    "sasa",
    "qp",
    "qpbo",
    "ilp",
    "sasa_synergy_only",  # Output 1 from sasax.py
    "sasa_conflict_only"  # Output 2 from sasax.py
]

# Define difficulty levels for filtering and sorting dataset categories.
DIFFICULTY_ORDER = ["EASY", "MEDIUM", "HARD", "medium"]

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.abspath(os.path.join(CODE_DIR, "..", "data", "output"))

# --- Logger Utility ---
class Logger(object):
    """
    Redirects stdout/stderr to both the terminal and a log file.
    """
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

# --- Script Execution Functions ---

def run_method(name, script_name):
    """
    Executes a single python script as a subprocess and captures its output.
    """
    print(f"\n" + "="*50)
    print(f"🔧 Running task: {name} ({script_name})...")
    print(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*50)
    sys.stdout.flush()
    
    script_path = os.path.join(CODE_DIR, script_name)
    
    if not os.path.exists(script_path):
        error_msg = f"❌ File not found: {script_path}"
        print(error_msg)
        return name, False, error_msg
    
    try:
        # Run subprocess with unbuffered output
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
            print(f"\n✅ Task {name} finished successfully.")
            return name, True, None
        else:
            error_output = ''.join(output_lines)
            print(f"\n❌ Task {name} failed with exit code: {returncode}")
            return name, False, error_output
            
    except Exception as e:
        error_msg = f"❌ Task {name} crashed: {e}"
        print(error_msg)
        return name, False, str(e)

def run_all_methods():
    """
    Iterates through the SCRIPTS_TO_RUN dictionary and executes them sequentially.
    """
    results = {}
    for task_name, script_name in SCRIPTS_TO_RUN.items():
        results[task_name] = run_method(task_name, script_name)[1]
        sys.stdout.flush()
    return results

# --- Helper Functions ---

def calculate_jaccard(set_a, set_b):
    if not set_a and not set_b: return 0.0
    intersection = len(set_a.intersection(set_b))
    union = len(set_a.union(set_b))
    if union == 0: return 0.0
    return intersection / union

def get_difficulty_rank(category_name):
    """
    Returns a numeric rank based on the difficulty keyword in the category name.
    Used for sorting output tables.
    """
    for idx, level in enumerate(DIFFICULTY_ORDER):
        if level in category_name:
            return idx
    return 999

# --- Result Processing ---

def read_results():
    """
    Scans the output directory, parses benchmark folders, and extracts:
    1. Scores and Execution Times
    2. Convergence history
    3. Node overlap statistics
    """
    final_results = defaultdict(lambda: defaultdict(list))
    convergence_data_frames = [] 
    overlap_stats = defaultdict(lambda: defaultdict(list))

    if not os.path.exists(DATA_DIR):
        print(f"❌ Data directory not found: {DATA_DIR}")
        return final_results, pd.DataFrame(), {}

    print("🔍 Scanning output directories...")

    sorted_folders = sorted(os.listdir(DATA_DIR))
    processed_count = 0
    skipped_count = 0

    for folder in sorted_folders:
        folder_path = os.path.join(DATA_DIR, folder)
        if not os.path.isdir(folder_path):
            continue
        
        # 1. Verify if the folder represents a benchmark task
        is_benchmark_task = False
        for level in DIFFICULTY_ORDER:
            if f"_{level}_" in folder: 
                is_benchmark_task = True
                break
        
        if not is_benchmark_task:
            skipped_count += 1
            continue
            
        processed_count += 1
        
        # Parse Group Key from folder name
        parts = folder.split("_")
        try:
            int(parts[-1]) 
            group_key = "_".join(parts[:-1]) 
        except ValueError:
            group_key = folder

        current_dataset_nodes = {} 

        # 2. Iterate through defined methods to extract data
        for method in METHODS_TO_ANALYZE:
            method_result_folder = f"{method}_result"
            method_path = os.path.join(folder_path, method_result_folder)
            
            # Robust file existence check
            score_file = os.path.join(method_path, "score.txt")
            time_file = os.path.join(method_path, "time.txt")
            count_file = os.path.join(method_path, "result_nodes_count.txt")
            nodes_csv_file = os.path.join(method_path, "result_nodes.csv")

            # Proceed if basic score/time files exist
            if os.path.exists(score_file) and os.path.exists(time_file):
                try:
                    with open(score_file, 'r') as f: 
                        score_text = f.read().strip()
                        if not score_text: raise ValueError("Score file is empty")
                        score = float(score_text)
                        
                    with open(time_file, 'r') as f: 
                        time_text = f.read().strip()
                        time_used = float(time_text) if time_text else 0.0

                    # Attempt to read node count; fallback to CSV calculation if count file is missing
                    count = 0
                    if os.path.exists(count_file):
                        with open(count_file, 'r') as f:
                            try:
                                count = int(float(f.read().strip()))
                            except:
                                count = 0
                    elif os.path.exists(nodes_csv_file):
                        try:
                            # Calculate line count from CSV
                            with open(nodes_csv_file, 'r') as f:
                                count = sum(1 for _ in f)
                            # Simple header check correction
                            df_temp = pd.read_csv(nodes_csv_file, header=None, nrows=1)
                            if not df_temp.empty and isinstance(df_temp.iloc[0,0], str):
                                try:
                                    float(df_temp.iloc[0,0])
                                except:
                                    count = max(0, count - 1)
                        except:
                            pass
                    
                    final_results[group_key][method].append((score, time_used, count))
                
                except Exception as e:
                    print(f"⚠️ Read failed [{folder}/{method}]: {e}")
            else:
                # Optional: Log missing files if debugging is needed
                pass

            # 3. Read Convergence History
            if method != "ilp":
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

            # 4. Read Node Data for Overlap Analysis
            if os.path.exists(nodes_csv_file):
                try:
                    # Robust CSV reading handling headers
                    df_temp = pd.read_csv(nodes_csv_file, header=None, nrows=5)
                    is_header = False
                    if not df_temp.empty:
                        try:
                            float(df_temp.iloc[0, 0])
                        except (ValueError, TypeError):
                            is_header = True
                    
                    if is_header:
                        nodes_df = pd.read_csv(nodes_csv_file)
                        if not nodes_df.empty:
                            vals = nodes_df.iloc[:, 0]
                        else:
                            vals = []
                    else:
                        nodes_df = pd.read_csv(nodes_csv_file, header=None)
                        if not nodes_df.empty:
                            vals = nodes_df.iloc[:, 0]
                        else:
                            vals = []
                        
                    if len(vals) > 0:
                        nodes_list = pd.to_numeric(vals, errors='coerce').dropna().astype(int).tolist()
                        current_dataset_nodes[method] = set(nodes_list)
                except Exception as e:
                    pass

        # Calculate Jaccard Similarity between methods
        if len(current_dataset_nodes) > 1:
            valid_methods = [m for m in METHODS_TO_ANALYZE if m in current_dataset_nodes]
            for m1, m2 in itertools.combinations_with_replacement(valid_methods, 2):
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
    """
    Aggregates results (Score, Time, Node Count) into a DataFrame.
    """
    summary = []
    sorted_keys = sorted(results.keys(), key=lambda x: (get_difficulty_rank(x), x))

    for group_key in sorted_keys:
        # Iterate through METHODS_TO_ANALYZE to ensure all columns are present
        for method in METHODS_TO_ANALYZE:
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
    """
    Aggregates Jaccard similarity statistics into a summary DataFrame and matrix CSVs.
    """
    overlap_summaries = []
    method_list = METHODS_TO_ANALYZE
    
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
        try:
            matrix.to_csv(matrix_file)
        except Exception:
            pass 

        for m1 in method_list:
            for m2 in method_list:
                if m1 == m2: continue 
                val = matrix.loc[m1, m2]
                if not pd.isna(val):
                    overlap_summaries.append({
                        "Category": category,
                        "Method_A": m1,
                        "Method_B": m2,
                        "Avg_Jaccard_Similarity": val
                    })
                    
    return pd.DataFrame(overlap_summaries)

# --- Main Entry Point ---

def main():
    print(f"--- Main Runner Started ---")
    sys.stdout.flush()
    
    # 1. Run Algorithms (Optional/Commented out by default)
    # run_all_methods()
    
    print("\n" + "="*50)
    print("📊 Reading and summarizing benchmark results...")
    print("   (Filters: EASY, MEDIUM, HARD, EXTREME)")
    print("="*50)
    sys.stdout.flush()
    
    # 2. Read Results
    final_results_dict, convergence_df, overlap_stats = read_results()
    
    # 3. Performance Summary
    summary_df = summarize_results(final_results_dict)
    print("\n📊 Performance Summary (Sorted by difficulty):")
    if summary_df.empty:
        print("  (Summary table is empty - No matching benchmark results found)")
    else:
        print(summary_df.head(20).to_string())
    
    summary_path = os.path.join(CODE_DIR, "summary.csv")
    try:
        summary_df.to_csv(summary_path, index=False)
        print(f"\n✅ Performance summary CSV saved to: {summary_path}")
    except Exception as e:
        print(f"\n❌ Failed to save CSV: {e}")

    # 4. Convergence Summary
    conv_path = os.path.join(CODE_DIR, "convergence_summary.csv")
    if not convergence_df.empty:
        try:
            convergence_df.to_csv(conv_path, index=False)
            print(f"✅ Convergence summary CSV saved to: {conv_path} (Rows: {len(convergence_df)})")
        except Exception as e:
            print(f"❌ Failed to save convergence summary: {e}")
    else:
        print("⚠️ No convergence data found (ignoring ILP).")

    # 5. Overlap Summary
    print("\n🔗 Node Overlap Analysis:")
    overlap_df = summarize_overlaps(overlap_stats)
    overlap_path = os.path.join(CODE_DIR, "overlap_summary.csv")
    if not overlap_df.empty:
        try:
            overlap_df.to_csv(overlap_path, index=False)
            print(f"✅ Overlap summary CSV saved to: {overlap_path}")
        except Exception as e:
            print(f"❌ Failed to save overlap summary: {e}")

    sys.stdout.flush()

if __name__ == "__main__":
    record_file_path = os.path.join(CODE_DIR, "record.txt")
    sys.stdout = Logger(record_file_path)
    sys.stderr = sys.stdout

    print(f"--- Main Runner (Benchmark Edition) ---")
    print(f"--- Time: {time.strftime('%Y-%m-%d %H:%M:%S')} ---")
    
    try:
        main()
        
        plot_script_path = os.path.join(CODE_DIR, "plot_summary.py")
        if os.path.exists(plot_script_path):
            print("\n" + "="*50)
            print("📊 Invoking plotting script...")
            subprocess.run([sys.executable, plot_script_path], cwd=CODE_DIR)
            print("✅ Plotting script executed successfully.")
        
    except Exception as e:
        print(f"\n❌ Fatal Error: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        print(f"\n--- End: {time.strftime('%Y-%m-%d %H:%M:%S')} ---")
        if isinstance(sys.stdout, Logger):
            sys.stdout.close()
            sys.stdout = sys.stdout.terminal