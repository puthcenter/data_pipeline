import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy import interpolate
import matplotlib.ticker as ticker

# --- Configuration ---
CODE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_BASE_DIR = os.path.abspath(os.path.join(CODE_DIR, "..", "data", "output"))

CONV_FILE = os.path.join(CODE_DIR, "convergence_summary.csv")
SUMMARY_FILE = os.path.join(CODE_DIR, "summary.csv")
OVERLAP_FILE = os.path.join(CODE_DIR, "overlap_summary.csv")

OUTPUT_DIR = os.path.join(CODE_DIR, "plots") 
DETAIL_DIR = os.path.join(OUTPUT_DIR, "details")   
AVG_DIR = os.path.join(OUTPUT_DIR, "average")      
MATRIX_DIR = os.path.join(OUTPUT_DIR, "similarity")

for d in [DETAIL_DIR, AVG_DIR, MATRIX_DIR]:
    os.makedirs(d, exist_ok=True)

sns.set_theme(style="white", context="paper", font_scale=1.4)

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'axes.unicode_minus': False, 
    'axes.edgecolor': 'black',     
    'axes.linewidth': 1.2,         
    'axes.grid': False,            
    'xtick.bottom': True,          
    'xtick.top': True,             
    'ytick.left': True,            
    'ytick.right': True,           
    'xtick.direction': 'in',       
    'ytick.direction': 'in',       
    'xtick.color': 'black',        
    'ytick.color': 'black',        
    'legend.frameon': True,        
    'legend.framealpha': 1.0,      
    'legend.edgecolor': 'black',   
    'legend.facecolor': 'white',   
    'legend.fancybox': False,      
    'legend.handlelength': 2.5, 
})

COMMON_LINE_WIDTH = 1.8 

# --- User Configuration Section ---

# List of algorithms to be included in the visualization.
# Comment out algorithms to hide them from the plots.
TARGET_ALGOS = [
    'sasa', 
    'sasa_synergy_only', 
    'sasa_conflict_only',
    ‘sghc', 
    'nghc',
    'greedy',
    'sa',
    'qpr',
    'qpbo'
]

# Set the maximum time (seconds) for the X-axis. 
# Set to None to use the maximum time found in the data.
MAX_PLOT_TIME = 450 

# --- Visual Configuration ---
ALGO_CONFIG = {
    # --- SASA Variants ---
    'sasa':               {'color': '#D62728', 'ls': '-',  'marker': 'o', 'zorder': 10}, 
    'sasa_synergy_only':  {'color': '#FF3333', 'ls': '--', 'marker': 'o', 'zorder': 9},  
    'sasa_conflict_only': {'color': '#FF8888', 'ls': ':',  'marker': 'o', 'zorder': 9},  
    
    # --- SGHC Variants ---
    'sghc':               {'color': '#FF7F0E', 'ls': '-',  'marker': 's', 'zorder': 10}, 
    'nghc':               {'color': '#FF7F0E', 'ls': '--', 'marker': 's', 'zorder': 9},  

    # --- Baselines ---
    'greedy': {'color': '#1F77B4', 'ls': '--', 'marker': '^', 'zorder': 2}, 
    'sa':     {'color': '#2CA02C', 'ls': '--', 'marker': 'v', 'zorder': 2}, 
    'qpr':    {'color': '#17BECF', 'ls': '--', 'marker': 'D', 'zorder': 2}, 
    'qpbo':   {'color': '#5F6A6A', 'ls': '--', 'marker': 'X', 'zorder': 2}, 
}

DEFAULT_CONFIG = {'color': 'gray', 'ls': '--', 'marker': '.', 'zorder': 1}

# --- Helper Functions ---

def get_ilp_score_for_instance(dataset_name):
    """Retrieves the ILP (Optimal) score if available for the given dataset."""
    try:
        score_path = os.path.join(DATA_BASE_DIR, dataset_name, "ilp_result", "score.txt")
        if os.path.exists(score_path):
            with open(score_path, 'r') as f:
                return max(float(f.read().strip()), 0.0)
    except: pass
    return None

def clean_data(df):
    """
    Preprocesses the dataframe to handle edge cases like infinite values.
    
    Note: Some algorithms may log -inf at t=0. We force this to 0.0 to ensure 
    the line starts correctly on the plot, avoiding dropna removal.
    """
    if 'Best_Value' in df.columns:
        df['Best_Value'] = df['Best_Value'].replace(-np.inf, 0.0)
        
    df.replace([np.inf], np.nan, inplace=True)
    df.dropna(subset=['Best_Value', 'Time_Elapsed'], inplace=True)
    
    if 'Best_Value' in df.columns:
        df['Best_Value'] = df['Best_Value'].clip(lower=0.0)
    return df

# --- Style Mapping ---
STR_TO_DASH_TUPLE = {
    '-':  "",             
    '--': (2.5, 1.25),    
    ':':  (1, 1),         
    '-.': (3, 1, 1, 1)    
}

# --- Plotting Functions ---

def plot_individual_datasets(df):
    print("\n🎨 Generating [Individual Dataset] detail plots...")
    datasets = df['Dataset'].unique()
    
    palette = {k: v['color'] for k, v in ALGO_CONFIG.items()}
    markers = {k: v['marker'] for k, v in ALGO_CONFIG.items()}
    
    dashes_map = {
        k: STR_TO_DASH_TUPLE.get(v['ls'], (2, 1)) 
        for k, v in ALGO_CONFIG.items()
    }
    
    hue_order = TARGET_ALGOS 
    
    for dataset in datasets:
        data_subset = df[df['Dataset'] == dataset].copy()
        if MAX_PLOT_TIME is not None:
            data_subset = data_subset[data_subset['Time_Elapsed'] <= MAX_PLOT_TIME]
        data_subset.sort_values(by=['Algorithm', 'Time_Elapsed'], inplace=True)
        if data_subset.empty: continue
            
        fig, ax = plt.subplots(figsize=(8, 5))
        
        # NOTE: drawstyle='steps-post' is used to strictly represent the optimization process.
        # The best value remains constant until a new, better solution is found (step function).
        sns.lineplot(data=data_subset, x='Time_Elapsed', y='Best_Value', hue='Algorithm',
                     palette=palette, hue_order=hue_order, 
                     style='Algorithm', markers=markers, dashes=dashes_map,
                     linewidth=COMMON_LINE_WIDTH, markersize=7, ax=ax,
                     drawstyle='steps-post') 
        
        ilp_score = get_ilp_score_for_instance(dataset)
        max_val = data_subset['Best_Value'].max()
        if ilp_score is not None:
            ax.axhline(y=ilp_score, color='#C00000', linestyle='--', linewidth=1.8, label='ILP (Optimal)')
            ax.text(x=MAX_PLOT_TIME*0.02 if MAX_PLOT_TIME else 10, y=ilp_score, 
                    s=f" Optimal: {ilp_score:.0f}", color='#C00000', va='bottom', fontweight='bold', fontsize=12)
            max_val = max(max_val, ilp_score)
        
        ax.set_xlabel("Time (s)", fontsize=14, fontweight='bold')
        ax.set_ylabel("Best Score", fontsize=14, fontweight='bold')
        ax.set_xlim(0, MAX_PLOT_TIME if MAX_PLOT_TIME else None)
        ax.set_ylim(bottom=0, top=max_val * 1.05) 
        ax.legend(loc='lower right', frameon=True, fontsize=10)
        plt.tight_layout(pad=0.5)
        plt.savefig(os.path.join(DETAIL_DIR, f"detail_{dataset}.pdf"), dpi=300, bbox_inches='tight')
        plt.close()

def plot_averaged_categories(conv_df, summary_df):
    print("\n🎨 Generating [Averaged] category plots...")
    
    categories = conv_df['Category'].unique()
    
    for category in categories:
        cat_df = conv_df[conv_df['Category'] == category]
        actual_max_time = cat_df['Time_Elapsed'].max()
        if pd.isna(actual_max_time) or actual_max_time == 0: continue
        
        target_max_time = MAX_PLOT_TIME if MAX_PLOT_TIME is not None else actual_max_time
        common_time = np.linspace(0, target_max_time, 200) 
        
        fig, ax = plt.subplots(figsize=(8, 5))
        
        algorithms = cat_df['Algorithm'].unique()
        sorted_algos = sorted(algorithms, key=lambda x: ALGO_CONFIG.get(x, DEFAULT_CONFIG)['zorder'])
        global_max_y = 0

        for algo in sorted_algos:
            algo_df = cat_df[cat_df['Algorithm'] == algo]
            interpolated_values = []
            
            for _, group in algo_df.groupby('Dataset'):
                group = group.sort_values('Time_Elapsed')
                x_vals = group['Time_Elapsed'].values
                y_vals = group['Best_Value'].values
                
                # Handle cases with insufficient data points (e.g., only t=0)
                if len(x_vals) < 1: continue
                if len(x_vals) == 1:
                    x_vals = np.append(x_vals, target_max_time)
                    y_vals = np.append(y_vals, y_vals[0])

                # Interpolation Strategy: 'previous'
                # This aligns with the step-function nature of 'steps-post'.
                # It prevents artificial slopes when averaging discrete events.
                f = interpolate.interp1d(x_vals, y_vals, kind='previous', 
                                         bounds_error=False, fill_value=(y_vals[0], y_vals[-1]))
                interpolated_values.append(f(common_time))
            
            if interpolated_values:
                stack = np.vstack(interpolated_values)
                mean_curve = np.mean(stack, axis=0)
                
                current_max = np.max(mean_curve)
                if current_max > global_max_y: global_max_y = current_max
                
                config = ALGO_CONFIG.get(algo, DEFAULT_CONFIG)
                
                ax.plot(
                    common_time, 
                    mean_curve, 
                    label=algo, 
                    color=config['color'],
                    linewidth=COMMON_LINE_WIDTH, 
                    linestyle=config['ls'], 
                    marker=config['marker'],
                    markersize=7,
                    markevery=20,
                    alpha=1.0, 
                    zorder=config['zorder']
                )
        
        if not summary_df.empty:
            ilp_row = summary_df[(summary_df['Category'] == category) & (summary_df['Method'] == 'ilp')]
            if not ilp_row.empty:
                ilp_avg = ilp_row.iloc[0]['Average_Score']
                if ilp_avg > global_max_y: global_max_y = ilp_avg
                ax.axhline(y=ilp_avg, color='#C00000', linestyle='--', linewidth=1.8, label='ILP Avg', zorder=1)
                ax.text(x=target_max_time*0.02, y=ilp_avg, s=f" Optimal: {ilp_avg:.0f}", 
                        color='#C00000', va='bottom', fontweight='bold', fontsize=12)

        ax.set_xlabel("Time (s)", fontsize=14, fontweight='bold')
        ax.set_ylabel("Average Score", fontsize=14, fontweight='bold')
        ax.set_xlim(0, target_max_time)
        ax.set_ylim(bottom=0, top=global_max_y * 1.05)

        ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
        ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

        handles, labels = ax.get_legend_handles_labels()
        
        def legend_sort_key(label):
            if label == 'ILP Avg': return 100 
            if label in TARGET_ALGOS:
                return TARGET_ALGOS.index(label)
            return 50

        hl = sorted(zip(handles, labels), key=lambda x: legend_sort_key(x[1]))
        handles_sorted, labels_sorted = zip(*hl)
        ax.legend(handles_sorted, labels_sorted, loc='lower right', frameon=True, fontsize=10)
        
        plt.tight_layout(pad=0.5)
        plt.savefig(os.path.join(AVG_DIR, f"average_{category}.pdf"), dpi=300, bbox_inches='tight')
        plt.close()

def plot_similarity_matrix(overlap_file):
    print("\n🎨 Generating [Similarity] heatmaps...")
    if not os.path.exists(overlap_file): return
    try:
        df = pd.read_csv(overlap_file)
    except: return
    if df.empty: return

    df.replace('qp', 'qpr', inplace=True)
    df = df[df['Method_A'].isin(TARGET_ALGOS) & df['Method_B'].isin(TARGET_ALGOS)]
    
    if df.empty:
        print("⚠️ No similarity data remaining after filtering.")
        return

    categories = df['Category'].unique()
    for cat in categories:
        cat_df = df[df['Category'] == cat]
        if cat_df.empty: continue
        
        pivot_table = cat_df.pivot(index="Method_A", columns="Method_B", values="Avg_Jaccard_Similarity")
        
        # Reorder index/columns to match TARGET_ALGOS
        existing_algos = [algo for algo in TARGET_ALGOS if algo in pivot_table.index]
        pivot_table = pivot_table.reindex(index=existing_algos, columns=existing_algos)

        fig, ax = plt.subplots(figsize=(8, 7)) 
        sns.heatmap(pivot_table, annot=True, cmap="Blues", vmin=0, vmax=1, fmt=".2f", 
                    linewidths=0.5, linecolor='black', ax=ax, cbar_kws={'label': 'Jaccard Similarity'})
        ax.set_xlabel("Method B", fontsize=12, fontweight='bold')
        ax.set_ylabel("Method A", fontsize=12, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(MATRIX_DIR, f"overlap_{cat}.pdf"), dpi=300, bbox_inches='tight')
        plt.close()

def process_and_plot():
    if os.path.exists(CONV_FILE):
        print("📖 Reading convergence data...")
        conv_df = pd.read_csv(CONV_FILE)
        conv_df['Algorithm'] = conv_df['Algorithm'].replace('qp', 'qpr')
        conv_df = clean_data(conv_df)
        
        print(f"🧹 Filtering data for algorithms: {TARGET_ALGOS}")
        conv_df = conv_df[conv_df['Algorithm'].isin(TARGET_ALGOS)]
        
        if conv_df.empty:
            print("❌ Error: Dataset is empty after filtering!")
            return
    else:
        print(f"⚠️ File not found: {CONV_FILE}")
        return

    if os.path.exists(SUMMARY_FILE):
        print("📖 Reading summary data...")
        summary_df = pd.read_csv(SUMMARY_FILE)
        if 'Average_Score' in summary_df.columns:
            summary_df['Average_Score'] = summary_df['Average_Score'].clip(lower=0.0)
    else:
        summary_df = pd.DataFrame()

    plot_individual_datasets(conv_df)
    plot_averaged_categories(conv_df, summary_df)
    plot_similarity_matrix(OVERLAP_FILE)

if __name__ == "__main__":
    process_and_plot()
    print("\n✨ All plotting tasks completed successfully!")