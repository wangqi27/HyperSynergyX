import os
import numpy as np
import pandas as pd
import tensorly as tl
from tensorly.decomposition import parafac
from sklearn.cluster import KMeans
from dbrwh import dbrwh_scores, load_incidence
from sim_builder import build_breast, build_feature_map_target

def get_top_targets(drugs, map_tgt, top_n=5):
    all_targets = []
    for d in drugs:
        ts = map_tgt.get(d, [])
        all_targets.extend(ts)
    
    if not all_targets:
        return []
        
    from collections import Counter
    c = Counter(all_targets)
    return [t for t, count in c.most_common(top_n)]

def main():
    root = r"d:\科研\数据\data"
    out_dir = os.path.join(root, "results_v2")
    os.makedirs(out_dir, exist_ok=True)
    
    print("Loading Breast Cancer Data...")
    h_path = os.path.join(root, "Breast_141_Drug_Combination_Incidence_Matrix.csv")
    if not os.path.exists(h_path):
        # try find in subfolders
        import glob
        files = glob.glob(os.path.join(root, "**", "Breast_141_Drug_Combination_Incidence_Matrix.csv"), recursive=True)
        if files:
            h_path = files[0]
    
    if not os.path.exists(h_path):
        print(f"Error: {h_path} not found.")
        return

    names, H = load_incidence(h_path)
    names2, S_atc, S_tgt = build_breast(root)
    Sim = np.maximum(S_atc, S_tgt)
    
    print("Computing DBRWH Scores...")
    T = dbrwh_scores(H, alpha=0.8, Sim=Sim)
    
    print("Running Tensor Decomposition (Rank=25)...")
    tl.set_backend('numpy')
    # Fix random state for reproducibility
    factors = parafac(T, rank=25, init='random', n_iter_max=50, random_state=42)
    
    # Drug factors are in mode 0, 1, 2. Use mode 0 (DrugA)
    drug_features = factors[1][0]
    
    print("Running K-Means (k=3)...")
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    labels = kmeans.fit_predict(drug_features)
    
    # Load targets
    tgt_path = os.path.join(root, 'ComInfo_v1', 'Target', 'Full', 'Target_Breast.xlsx')
    map_tgt = build_feature_map_target(tgt_path)
    
    rows = []
    for i in range(3):
        cluster_indices = np.where(labels == i)[0]
        cluster_drugs = [names[idx] for idx in cluster_indices]
        
        top_targets = get_top_targets(cluster_drugs, map_tgt, top_n=10)
        
        # Format example drugs (first 5)
        example_drugs = ", ".join(sorted(cluster_drugs)[:5])
        
        # Format targets
        targets_str = ", ".join(top_targets)
        
        print(f"Mode {i}: {len(cluster_drugs)} drugs. Targets: {targets_str}")
        
        rows.append({
            "Latent Mode": f"Mode {i+1}",
            "Drug Count": len(cluster_drugs),
            "Representative Drugs": example_drugs,
            "Enriched Targets": targets_str
        })
        
    df = pd.DataFrame(rows)
    out_csv = os.path.join(out_dir, "Table_VIII.csv")
    df.to_csv(out_csv, index=False)
    print(f"Saved Table VIII to {out_csv}")

if __name__ == "__main__":
    main()
