import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy import interpolate
import matplotlib.ticker as ticker

# --- 配置 ---
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

# 【修改点 2】全局字体基准放大至 2.0
sns.set_theme(style="white", context="paper", font_scale=2.0)

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'axes.unicode_minus': False, 
    'axes.edgecolor': 'black',    
    'axes.linewidth': 1.5,        # 稍微加粗边框适应大图
    'axes.grid': False,           
    'xtick.bottom': True,         
    'xtick.top': True,            
    'ytick.left': True,           
    'ytick.right': True,          
    'xtick.direction': 'in',      
    'ytick.direction': 'in',      
    'xtick.color': 'black',       
    'ytick.color': 'black',       
    
    # 【修改点 2】显式设置各大元素的字体大小
    'axes.labelsize': 22,         
    'xtick.labelsize': 18,        
    'ytick.labelsize': 18,        
    'legend.fontsize': 18, 
    
    # 【修改点 1】图例透明化配置
    'legend.frameon': True,       
    'legend.framealpha': 0.8,     # 半透明
    'legend.edgecolor': 'gray',  
    'legend.facecolor': 'white',  
    'legend.fancybox': False,     
})

COMMON_LINE_WIDTH = 2.0 # 稍微加粗线条适应大图

ALGO_CONFIG = {
    'sasa':   {'color': '#D62728', 'ls': '-', 'marker': 'o', 'zorder': 10}, 
    'sghc':   {'color': '#FF7F0E', 'ls': '-', 'marker': 's', 'zorder': 9},  
    'greedy': {'color': '#1F77B4', 'ls': '--', 'marker': '^', 'zorder': 2}, 
    'sa':     {'color': '#2CA02C', 'ls': '--', 'marker': 'v', 'zorder': 2}, 
    'qpr':    {'color': '#17BECF', 'ls': '--', 'marker': 'D', 'zorder': 2},
    'qpbo':   {'color': '#5F6A6A', 'ls': '--', 'marker': 'X', 'zorder': 2}, 
}

DEFAULT_CONFIG = {'color': 'gray', 'ls': '--', 'marker': '.', 'zorder': 1}
MAX_PLOT_TIME = 3200 

# --- 辅助函数 ---
def get_ilp_score_for_instance(dataset_name):
    try:
        score_path = os.path.join(DATA_BASE_DIR, dataset_name, "ilp_result", "score.txt")
        if os.path.exists(score_path):
            with open(score_path, 'r') as f:
                return max(float(f.read().strip()), 0.0)
    except: pass
    return None

def clean_data(df):
    if 'Best_Value' in df.columns:
        df['Best_Value'] = df['Best_Value'].replace(-np.inf, 0.0)
        
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(subset=['Best_Value', 'Time_Elapsed'], inplace=True)
    
    if 'Best_Value' in df.columns:
        df['Best_Value'] = df['Best_Value'].clip(lower=0.0)
    return df

# --- 绘图函数 ---

def plot_individual_datasets(df):
    print("\n🎨 正在生成 [单独数据集] 详情图...")
    datasets = df['Dataset'].unique()
    palette = {k: v['color'] for k, v in ALGO_CONFIG.items()}
    markers = {k: v['marker'] for k, v in ALGO_CONFIG.items()}
    hue_order = ['sasa', 'sghc'] + [k for k in ALGO_CONFIG.keys() if k not in ['sasa', 'sghc']]
    
    for dataset in datasets:
        data_subset = df[df['Dataset'] == dataset].copy()
        if MAX_PLOT_TIME is not None:
            data_subset = data_subset[data_subset['Time_Elapsed'] <= MAX_PLOT_TIME]
        data_subset.sort_values(by=['Algorithm', 'Time_Elapsed'], inplace=True)
        if data_subset.empty: continue
            
        fig, ax = plt.subplots(figsize=(8, 5))
        
        sns.lineplot(data=data_subset, x='Time_Elapsed', y='Best_Value', hue='Algorithm',
                    palette=palette, hue_order=hue_order, style='Algorithm', markers=markers,
                    linewidth=COMMON_LINE_WIDTH, markersize=9, dashes=False, ax=ax,
                    drawstyle='steps-post')
        
        ilp_score = get_ilp_score_for_instance(dataset)
        max_val = data_subset['Best_Value'].max()
        if ilp_score is not None:
            ax.axhline(y=ilp_score, color='#C00000', linestyle='--', linewidth=2.0, label='ILP (Optimal)')
            # 【修改点 2】放大最优解提示字体
            ax.text(x=MAX_PLOT_TIME*0.02 if MAX_PLOT_TIME else 10, y=ilp_score, 
                    s=f" Optimal: {ilp_score:.0f}", color='#C00000', va='bottom', fontweight='bold', fontsize=18)
            max_val = max(max_val, ilp_score)
        
        # 【修改点 2】放大坐标轴标签
        ax.set_xlabel("Time (s)", fontsize=22, fontweight='bold')
        ax.set_ylabel("Best Score", fontsize=22, fontweight='bold')
        ax.set_xlim(0, MAX_PLOT_TIME if MAX_PLOT_TIME else None)
        ax.set_ylim(bottom=0, top=max_val * 1.05) 
        
        # 【修改点 3】强制 X 轴每隔 1000 标一个刻度，防止文字重叠
        ax.xaxis.set_major_locator(ticker.MultipleLocator(1000))
        
        # 【修改点 1】图例透明化且置顶
        leg = ax.legend(loc='lower right', frameon=True, fontsize=18)
        leg.set_zorder(100)
        leg.get_frame().set_alpha(0.8)
        
        plt.tight_layout(pad=0.5)
        plt.savefig(os.path.join(DETAIL_DIR, f"detail_{dataset}.pdf"), dpi=300, bbox_inches='tight')
        plt.close()

def plot_averaged_categories(conv_df, summary_df):
    print("\n🎨 正在生成 [平均] 对比图 (已移除方差阴影)...")
    
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
                if len(x_vals) < 2: continue
                
                f = interpolate.interp1d(x_vals, y_vals, kind='previous', bounds_error=False, fill_value=(y_vals[0], y_vals[-1]))
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
                    markersize=9,
                    markevery=20,
                    alpha=1.0, 
                    zorder=config['zorder']
                )
        
        if not summary_df.empty:
            ilp_row = summary_df[(summary_df['Category'] == category) & (summary_df['Method'] == 'ilp')]
            if not ilp_row.empty:
                ilp_avg = ilp_row.iloc[0]['Average_Score']
                if ilp_avg > global_max_y: global_max_y = ilp_avg
                ax.axhline(y=ilp_avg, color='#C00000', linestyle='--', linewidth=2.0, label='ILP Avg', zorder=1)
                # 【修改点 2】放大最优解提示字体
                ax.text(x=target_max_time*0.02, y=ilp_avg, s=f" Optimal: {ilp_avg:.0f}", 
                        color='#C00000', va='bottom', fontweight='bold', fontsize=18)

        # 【修改点 2】放大坐标轴标签
        ax.set_xlabel("Time (s)", fontsize=22, fontweight='bold')
        ax.set_ylabel("Average Score", fontsize=22, fontweight='bold')
        ax.set_xlim(0, target_max_time)
        ax.set_ylim(bottom=0, top=global_max_y * 1.05)
        
        # 【修改点 3】强制 X 轴每隔 1000 标一个刻度
        ax.xaxis.set_major_locator(ticker.MultipleLocator(1000))

        ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
        ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

        handles, labels = ax.get_legend_handles_labels()
        def legend_priority(label):
            if label in ['sasa', 'sghc']: return 3
            if label == 'ILP Avg': return 2
            return 1
        hl = sorted(zip(handles, labels), key=lambda x: legend_priority(x[1]), reverse=True)
        if hl:
            handles_sorted, labels_sorted = zip(*hl)
            # 【修改点 1】图例透明化且置顶
            leg = ax.legend(handles_sorted, labels_sorted, loc='lower right', frameon=True, fontsize=18)
            leg.set_zorder(100)
            leg.get_frame().set_alpha(0.8)
        
        plt.tight_layout(pad=0.5)
        plt.savefig(os.path.join(AVG_DIR, f"average_{category}.pdf"), dpi=300, bbox_inches='tight')
        plt.close()

def plot_similarity_matrix(overlap_file):
    print("\n🎨 正在生成 [相似度] 热力图...")
    if not os.path.exists(overlap_file): return
    try:
        df = pd.read_csv(overlap_file)
    except: return
    if df.empty: return

    df.replace('qp', 'qpr', inplace=True)

    categories = df['Category'].unique()
    for cat in categories:
        cat_df = df[df['Category'] == cat]
        pivot_table = cat_df.pivot(index="Method_A", columns="Method_B", values="Avg_Jaccard_Similarity")
        
        # 适当拉宽图表防止轴标签和内容重叠
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # 【修改点 2】放大了热力图内部数字 (annot_kws)
        sns.heatmap(pivot_table, annot=True, cmap="Blues", vmin=0, vmax=1, fmt=".2f", 
                    linewidths=0.5, linecolor='black', ax=ax, 
                    cbar_kws={'label': 'Jaccard Similarity'},
                    annot_kws={"size": 16})
                    
        # 【修改点 2】放大坐标轴标签
        ax.set_xlabel("Method B", fontsize=18, fontweight='bold')
        ax.set_ylabel("Method A", fontsize=18, fontweight='bold')
        
        # 调整颜色条(colorbar)标签大小
        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(labelsize=14)
        cbar.set_label('Jaccard Similarity', size=16, weight='bold')

        plt.tight_layout()
        plt.savefig(os.path.join(MATRIX_DIR, f"overlap_{cat}.pdf"), dpi=300, bbox_inches='tight')
        plt.close()

def process_and_plot():
    if os.path.exists(CONV_FILE):
        print("📖 读取收敛数据...")
        conv_df = pd.read_csv(CONV_FILE)
        
        conv_df['Algorithm'] = conv_df['Algorithm'].replace('qp', 'qpr')
        conv_df = clean_data(conv_df)
    else:
        print(f"⚠️ 找不到: {CONV_FILE}")
        return

    if os.path.exists(SUMMARY_FILE):
        print("📖 读取汇总数据...")
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
    print("\n✨ 所有绘图任务已完成！")