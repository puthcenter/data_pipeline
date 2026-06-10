import pandas as pd
import numpy as np

def get_kneedle_point(group):
    """
    使用 Kneedle 几何距离法寻找平均收敛曲线的大拐点
    """
    df_valid = group[group['Best_Value'] != -np.inf].sort_values('Time_Elapsed')
    if df_valid.empty:
        return np.nan
        
    # 注意：绝对不能用 drop_duplicates，必须保留长平缓期！
    t = df_valid['Time_Elapsed'].values
    v = df_valid['Best_Value'].values
    
    if len(t) <= 1:
        return t[0] if len(t) == 1 else np.nan
        
    t_min, t_max = t.min(), t.max()
    v_min, v_max = v.min(), v.max()
    
    if t_max == t_min or v_max == v_min:
        return t_min
        
    t_norm = (t - t_min) / (t_max - t_min)
    v_norm = (v - v_min) / (v_max - v_min)
    
    distances = v_norm - t_norm
    
    if np.max(distances) <= 0:
        return t[np.argmax(v)]
    
    knee_idx = np.argmax(distances)
    return t[knee_idx]

def get_mean_convergence_curve(df_algo_cat):
    """合成某算法在特定类别图上的平均曲线"""
    df_clean = df_algo_cat.groupby(['Dataset', 'Time_Elapsed'])['Best_Value'].max().reset_index()
    pivot_df = df_clean.pivot(index='Time_Elapsed', columns='Dataset', values='Best_Value')
    pivot_df = pivot_df.ffill().fillna(0)
    mean_values = pivot_df.mean(axis=1)
    
    mean_curve = pd.DataFrame({
        'Time_Elapsed': pivot_df.index,
        'Best_Value': mean_values.values
    }).reset_index(drop=True)
    return mean_curve


def analyze_convergence_kneedle(convergence_file, output_file):
    # 读取收敛数据
    df_conv = pd.read_csv(convergence_file)
    
    # 1. 负数全部归 0
    df_conv.loc[df_conv['Best_Value'] < 0, 'Best_Value'] = 0
    
    # 2. 提取类别 Category (前两项一致的视为同一类图，例如 dag101_MEDIUM_0 -> dag101_MEDIUM)
    df_conv['Category'] = df_conv['Dataset'].apply(lambda x: "_".join(str(x).split("_")[:2]))
    
    # 3. 计算各算法在各类图上的最佳结果与平均拐点
    results = []
    for (algo, cat), group in df_conv.groupby(['Algorithm', 'Category']):
        # 该类别下，提取每个具体 Dataset (比如 _0, _1, _2) 的最高得分
        dataset_max_vals = group.groupby('Dataset')['Best_Value'].max()
        # 对这几个最高得分取平均
        avg_best_val = dataset_max_vals.mean()
        
        # 合成该类别下图的平均收敛曲线，并计算拐点
        mean_curve = get_mean_convergence_curve(group)
        knee_time = get_kneedle_point(mean_curve)
        
        results.append({
            'Category': cat,
            'Algorithm': algo, 
            'Avg_Best_Value': avg_best_val,
            'Avg_Time_to_Knee': knee_time,
            'Evaluated_Datasets': len(dataset_max_vals)
        })
        
    category_summary = pd.DataFrame(results)
    
    # 4. 排序：按图的类别聚合，同类别下按得分降序，时间升序
    category_summary = category_summary.sort_values(
        by=['Category', 'Avg_Best_Value', 'Avg_Time_to_Knee'], 
        ascending=[True, False, True]
    ).reset_index(drop=True)
    
    # 5. 输出到指定的 CSV 文件
    category_summary.to_csv(output_file, index=False)
    print(f"✅ 统计完成！结果已成功保存至文件：{output_file}")
    
    return category_summary


# ========= 运行 =========
if __name__ == "__main__":
    # 配置输入与输出文件路径
    input_csv = 'convergence_summary.csv'
    output_csv = 'category_convergence_stats.csv'
    
    category_df = analyze_convergence_kneedle(input_csv, output_csv)
    
    # 打印前几行作为预览
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    print("\n输出数据预览 (Top 5)：")
    print(category_df.head())