import pandas as pd
import os
results_dir = "runs/detect"

# List all directories in the parent directory
# directories = [d for d in os.listdir(results_dir) if os.path.isdir(os.path.join(results_dir, d))]
directories = ["DeeperDarts4"]
# other_results_df = pd.read_csv("test_results.csv")
# other_results_df.set_index('test_name', inplace=True)
# print(other_results_df.loc['SGD']['avg_error'])
output_csv_filepath = "All Results.csv"
for dir in directories:
    # other_results = other_results_df.loc[dir]
    results_path = f"{results_dir}/{dir}/results.csv"
    print(f"Results path: {results_path}")
    df = pd.read_csv(results_path)
    df.columns = df.columns.str.strip()
    name = dir
    map50 = round(df['metrics/mAP50(B)'].iloc[-1],3)
    map50_95 =  round(df['metrics/mAP50-95(B)'].iloc[-1],3)
    Precision = df['metrics/precision(B)'].iloc[-1]
    Recall = df['metrics/recall(B)'].iloc[-1]
    epoch = df['epoch'].iloc[-1]

    # pcs = other_results['PCS']
    # avg_error = other_results['avg_error']
    # abs_avg_error = other_results['avg_abs_error']
    # inference_time = other_results['avg_inference_time_ms']
    f1 = round((2 * (Precision * Recall) / (Precision + Recall)),3)
    
    output = f"{name},{epoch},{map50},{map50_95},{f1}\n"#,{pcs},{avg_error},{abs_avg_error},{inference_time}\n"
    with open(output_csv_filepath, 'a') as f:
        f.writelines(output)