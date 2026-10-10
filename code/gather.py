import pandas as pd
import numpy as np

def get_kneedle_point(group):
    """
    Find the main knee point of the average convergence curve using Kneedle's geometric distance method.
    """
    df_valid = group[group['Best_Value'] != -np.inf].sort_values('Time_Elapsed')
    if df_valid.empty:
        return np.nan
        
    # Do not use drop_duplicates: long plateaus must be preserved.
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
    """Build an algorithm's average convergence curve for a graph category."""
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
    # Read convergence data.
    df_conv = pd.read_csv(convergence_file)
    
    # 1. Clamp all negative values to zero.
    df_conv.loc[df_conv['Best_Value'] < 0, 'Best_Value'] = 0
    
    # 2. Extract the category from the first two components, e.g., dag101_MEDIUM_0 -> dag101_MEDIUM.
    df_conv['Category'] = df_conv['Dataset'].apply(lambda x: "_".join(str(x).split("_")[:2]))
    
    # 3. Compute the best scores and average-curve knee points for each algorithm and graph category.
    results = []
    for (algo, cat), group in df_conv.groupby(['Algorithm', 'Category']):
        # Extract the highest score for each dataset in this category (e.g., _0, _1, _2).
        dataset_max_vals = group.groupby('Dataset')['Best_Value'].max()
        # Average these highest scores.
        avg_best_val = dataset_max_vals.mean()
        
        # Build the category's average convergence curve and find its knee point.
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
    
    # 4. Group by graph category, then sort by descending score and ascending time.
    category_summary = category_summary.sort_values(
        by=['Category', 'Avg_Best_Value', 'Avg_Time_to_Knee'], 
        ascending=[True, False, True]
    ).reset_index(drop=True)
    
    # 5. Write to the specified CSV file.
    category_summary.to_csv(output_file, index=False)
    print(f"✅ Statistics complete! Results saved to: {output_file}")
    
    return category_summary


# ========= Execution =========
if __name__ == "__main__":
    # Configure input and output file paths.
    input_csv = 'convergence_summary.csv'
    output_csv = 'category_convergence_stats.csv'
    
    category_df = analyze_convergence_kneedle(input_csv, output_csv)
    
    # Print the first few rows as a preview.
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    print("\nOutput preview (Top 5):")
    print(category_df.head())