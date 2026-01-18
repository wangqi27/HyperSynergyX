import os
import subprocess
import pandas as pd
import json

def run_cmd(cmd):
    print(f"Running: {cmd}", flush=True)
    subprocess.check_call(cmd, shell=True)

def main():
    root = r"d:\科研\数据\data"
    outdir = os.path.join(root, "results_v2")
    os.makedirs(outdir, exist_ok=True)
    
    datasets = [
        ("breast", "splits/breast.json", "dbrwh_breast"),
        ("lung", "splits/lung.json", "dbrwh_lung")
    ]
    
    variants = [
        ("_no_restart", "No Restart Bias"),
        ("_no_attraction", "No Attraction Bias"),
        ("_no_restart_no_attraction", "Random Walk")
    ]
    
    metrics = []
    
    for ds_name, split_json, base_model in datasets:
        # Load baseline if exists
        base_json = os.path.join(outdir, f"metrics_{base_model}_{ds_name}_mean.json")
        if os.path.exists(base_json):
            with open(base_json, "r") as f:
                m = json.load(f)
                m["Dataset"] = ds_name.capitalize()
                m["Model"] = "Proposed DBRWH"
                metrics.append(m)
        else:
            print(f"Baseline {base_model} not found, skipping or run it first.")
            
        for suffix, label in variants:
            model = base_model + suffix
            res_json = os.path.join(outdir, f"metrics_{model}_{ds_name}_mean.json")
            if not os.path.exists(res_json):
                cmd = f"python {os.path.join(root, 'scripts', 'run_cv_eval.py')} {os.path.join(root, split_json)} --model {model} --root {root} --outdir {outdir}"
                run_cmd(cmd)
            
            with open(res_json, "r") as f:
                m = json.load(f)
                m["Dataset"] = ds_name.capitalize()
                m["Model"] = label
                metrics.append(m)
                
    df = pd.DataFrame(metrics)
    cols = ["Dataset", "Model", "AUROC", "AUPRC", "F1", "Precision", "Recall"]
    df = df[cols]
    out_csv = os.path.join(outdir, "Table_VII.csv")
    df.to_csv(out_csv, index=False)
    print(f"Saved Table VII to {out_csv}")
    print(df)

if __name__ == "__main__":
    main()
