import os
import json
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_score, recall_score
from models_registry import build_model
from dbrwh import dbrwh_scores
from sim_builder import build_breast, build_lung

def load_split(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)

def triad_key(a,b,c):
    return tuple(sorted([str(a),str(b),str(c)]))

def build_freq_index_breast(root):
    path = os.path.join(root, "Feature_AIO", "Feature_AIO", "Breast_all.csv")
    idx = {}
    if not os.path.isfile(path):
        return idx
    df = pd.read_csv(path)
    cols = [c for c in df.columns if "name_" in str(c).lower()]
    for _, r in df.iterrows():
        drugs = [str(r[c]).strip() for c in cols if pd.notna(r[c])]
        if len(drugs) >= 3:
            for i in range(len(drugs)):
                for j in range(i+1, len(drugs)):
                    for k in range(j+1, len(drugs)):
                        key = triad_key(drugs[i],drugs[j],drugs[k])
                        idx[key] = idx.get(key,0)+1
    return idx

def build_freq_index_lung(root):
    path = os.path.join(root, "Feature_AIO", "Feature_AIO", "Lung_all.xlsx")
    idx = {}
    if not os.path.isfile(path):
        return idx
    df = pd.read_excel(path)
    cols = [c for c in df.columns if "name_" in str(c).lower()]
    for _, r in df.iterrows():
        drugs = [str(r[c]).strip() for c in cols if pd.notna(r[c])]
        if len(drugs) >= 3:
            for i in range(len(drugs)):
                for j in range(i+1, len(drugs)):
                    for k in range(j+1, len(drugs)):
                        key = triad_key(drugs[i],drugs[j],drugs[k])
                        idx[key] = idx.get(key,0)+1
    return idx

def score_freq(items, idx):
    scores = []
    for it in items:
        key = triad_key(it["drugA"], it["drugB"], it["drugC"])
        s = idx.get(key, 0)
        scores.append(float(s))
    arr = np.array(scores, dtype=float)
    if arr.max() > 0:
        arr = arr / arr.max()
    return arr.tolist()

def compute_metrics(y_true, y_score):
    auroc = roc_auc_score(y_true, y_score)
    auprc = average_precision_score(y_true, y_score)
    thr = np.percentile(y_score, 50)
    y_pred = (np.array(y_score) >= thr).astype(int)
    f1 = f1_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    return {"AUROC": auroc, "AUPRC": auprc, "F1": f1, "Precision": prec, "Recall": rec}

def run_fold(split_json, fold, model, root):
    all_items = load_split(split_json)
    ds = "breast" if "breast" in split_json.lower() else ("lung" if "lung" in split_json.lower() else "dataset")
    items = [d for d in all_items if int(d["fold"]) == int(fold)]
    y_true = [int(d["label"]) for d in items]
    if model == "random":
        rng = np.random.default_rng(42+fold)
        scores = rng.random(len(items))
    elif model == "freq_breast":
        idx = build_freq_index_breast(root)
        scores = score_freq(items, idx)
    elif model == "freq_lung":
        idx = build_freq_index_lung(root)
        scores = score_freq(items, idx)
    elif model == "freq_train":
        train_pos = [d for d in all_items if int(d["fold"]) != int(fold) and int(d["label"]) == 1]
        idx = {}
        for it in train_pos:
            key = triad_key(it["drugA"], it["drugB"], it["drugC"])
            idx[key] = idx.get(key,0)+1
        scores = score_freq(items, idx)
    elif model == "pair_train":
        train_pos = [d for d in all_items if int(d["fold"]) != int(fold) and int(d["label"]) == 1]
        pair_idx = {}
        for it in train_pos:
            a,b,c = it["drugA"], it["drugB"], it["drugC"]
            pairs = [tuple(sorted([a,b])), tuple(sorted([a,c])), tuple(sorted([b,c]))]
            for p in pairs:
                pair_idx[p] = pair_idx.get(p,0)+1
        scores = []
        for it in items:
            a,b,c = it["drugA"], it["drugB"], it["drugC"]
            pairs = [tuple(sorted([a,b])), tuple(sorted([a,c])), tuple(sorted([b,c]))]
            s = sum(pair_idx.get(p,0) for p in pairs)
            scores.append(float(s))
        arr = np.array(scores, dtype=float)
        if arr.max() > 0:
            arr = arr / arr.max()
        scores = arr.tolist()
    elif model in ["higsyn","mhclsyn","deepsynergy","gcn","gat"]:
        train_items = [d for d in all_items if int(d["fold"]) != int(fold)]
        mdl = build_model(model, ds, root)
        mdl.fit(train_items)
        scores = mdl.score(items)
    elif model in ["dbrwh_breast","dbrwh_lung"] or model.startswith("dbrwh_breast_w") or model.startswith("dbrwh_lung_w") or ("dbrwh_" in model and ("_no_restart" in model or "_no_attraction" in model)):
        # Build fold-specific incidence matrix from training positives to avoid leakage
        train_items = [d for d in all_items if int(d["fold"]) != int(fold)]
        # collect unique drug names across all items to ensure consistent indexing
        drug_names = sorted(list({str(d["drugA"]) for d in all_items} | {str(d["drugB"]) for d in all_items} | {str(d["drugC"]) for d in all_items}))
        name2idx = {n:i for i,n in enumerate(drug_names)}
        # build incidence H: N x E, E = number of training positive triads
        pos_train = [d for d in train_items if int(d["label"]) == 1]
        E = len(pos_train)
        N = len(drug_names)
        import numpy as np
        H = np.zeros((N, E), dtype=float)
        for e, it in enumerate(pos_train):
            ia = name2idx.get(str(it["drugA"]))
            ib = name2idx.get(str(it["drugB"]))
            ic = name2idx.get(str(it["drugC"]))
            if ia is None or ib is None or ic is None:
                continue
            H[ia, e] = 1.0
            H[ib, e] = 1.0
            H[ic, e] = 1.0
        # optional weighted similarity for ATC/Target when using *_wXXX models
        Sim = None
        if model.startswith("dbrwh_breast_w") or model.startswith("dbrwh_lung_w"):
            wtxt = model.split("_w")[-1]
            try:
                w = int(wtxt)/100.0
            except Exception:
                w = 0.5
            if "breast" in model:
                names2, S_atc, S_tgt = build_breast(root)
            else:
                names2, S_atc, S_tgt = build_lung(root)
            # align
            import numpy as np
            S_atc_al = np.zeros((N,N), dtype=float)
            S_tgt_al = np.zeros((N,N), dtype=float)
            idx2 = {n:i for i,n in enumerate(names2)}
            for i,n in enumerate(drug_names):
                ii = idx2.get(n)
                if ii is None:
                    continue
                for j,m in enumerate(drug_names):
                    jj = idx2.get(m)
                    if jj is None:
                        continue
                    S_atc_al[i,j] = S_atc[ii,jj]
                    S_tgt_al[i,j] = S_tgt[ii,jj]
            Sim = w * S_atc_al + (1.0 - w) * S_tgt_al
        
        restart_bias = True
        attraction_bias = True
        if "_no_restart" in model:
            restart_bias = False
        if "_no_attraction" in model:
            attraction_bias = False
            
        S = dbrwh_scores(H, alpha=0.8, Sim=Sim, restart_bias=restart_bias, attraction_bias=attraction_bias)
        scores = []
        for it in items:
            ia = name2idx.get(str(it["drugA"]))
            ib = name2idx.get(str(it["drugB"]))
            ic = name2idx.get(str(it["drugC"]))
            s = float(S[ia, ib, ic]) if ia is not None and ib is not None and ic is not None else 0.0
            scores.append(s)
    elif model in ["dbrwh_breast_atc","dbrwh_breast_target","dbrwh_breast_both"]:
        train_items = [d for d in all_items if int(d["fold"]) != int(fold)]
        drug_names = sorted(list({str(d["drugA"]) for d in all_items} | {str(d["drugB"]) for d in all_items} | {str(d["drugC"]) for d in all_items}))
        name2idx = {n:i for i,n in enumerate(drug_names)}
        import numpy as np
        pos_train = [d for d in train_items if int(d["label"]) == 1]
        N = len(drug_names)
        E = len(pos_train)
        H = np.zeros((N, E), dtype=float)
        for e, it in enumerate(pos_train):
            ia = name2idx.get(str(it["drugA"]))
            ib = name2idx.get(str(it["drugB"]))
            ic = name2idx.get(str(it["drugC"]))
            if ia is None or ib is None or ic is None:
                continue
            H[ia,e]=1;H[ib,e]=1;H[ic,e]=1
        names2, S_atc, S_tgt = build_breast(root)
        # align Sim to current names
        import numpy as np
        S_atc_al = np.zeros((N,N), dtype=float)
        S_tgt_al = np.zeros((N,N), dtype=float)
        idx2 = {n:i for i,n in enumerate(names2)}
        for i,n in enumerate(drug_names):
            ii = idx2.get(n)
            if ii is None:
                continue
            for j,m in enumerate(drug_names):
                jj = idx2.get(m)
                if jj is None:
                    continue
                S_atc_al[i,j] = S_atc[ii,jj]
                S_tgt_al[i,j] = S_tgt[ii,jj]
        Sim = None
        if model.endswith("atc"):
            Sim = S_atc_al
        elif model.endswith("target"):
            Sim = S_tgt_al
        else:
            Sim = np.maximum(S_atc_al, S_tgt_al)
        T = dbrwh_scores(H, alpha=0.8, Sim=Sim)
        scores = []
        for it in items:
            a,b,c = it["drugA"], it["drugB"], it["drugC"]
            ia = name2idx.get(str(a))
            ib = name2idx.get(str(b))
            ic = name2idx.get(str(c))
            s = float(T[ia, ib, ic]) if ia is not None and ib is not None and ic is not None else 0.0
            scores.append(s)
    elif model in ["dbrwh_lung_atc","dbrwh_lung_target","dbrwh_lung_both"]:
        train_items = [d for d in all_items if int(d["fold"]) != int(fold)]
        drug_names = sorted(list({str(d["drugA"]) for d in all_items} | {str(d["drugB"]) for d in all_items} | {str(d["drugC"]) for d in all_items}))
        name2idx = {n:i for i,n in enumerate(drug_names)}
        import numpy as np
        pos_train = [d for d in train_items if int(d["label"]) == 1]
        N = len(drug_names)
        E = len(pos_train)
        H = np.zeros((N, E), dtype=float)
        for e, it in enumerate(pos_train):
            ia = name2idx.get(str(it["drugA"]))
            ib = name2idx.get(str(it["drugB"]))
            ic = name2idx.get(str(it["drugC"]))
            if ia is None or ib is None or ic is None:
                continue
            H[ia,e]=1;H[ib,e]=1;H[ic,e]=1
        names2, S_atc, S_tgt = build_lung(root)
        S_atc_al = np.zeros((N,N), dtype=float)
        S_tgt_al = np.zeros((N,N), dtype=float)
        idx2 = {n:i for i,n in enumerate(names2)}
        for i,n in enumerate(drug_names):
            ii = idx2.get(n)
            if ii is None:
                continue
            for j,m in enumerate(drug_names):
                jj = idx2.get(m)
                if jj is None:
                    continue
                S_atc_al[i,j] = S_atc[ii,jj]
                S_tgt_al[i,j] = S_tgt[ii,jj]
        Sim = None
        if model.endswith("atc"):
            Sim = S_atc_al
        elif model.endswith("target"):
            Sim = S_tgt_al
        else:
            Sim = np.maximum(S_atc_al, S_tgt_al)
        T = dbrwh_scores(H, alpha=0.8, Sim=Sim)
        scores = []
        for it in items:
            a,b,c = it["drugA"], it["drugB"], it["drugC"]
            ia = name2idx.get(str(a))
            ib = name2idx.get(str(b))
            ic = name2idx.get(str(c))
            s = float(T[ia, ib, ic]) if ia is not None and ib is not None and ic is not None else 0.0
            scores.append(s)
    else:
        rng = np.random.default_rng(42+fold)
        scores = rng.random(len(items))
    metrics = compute_metrics(y_true, scores)
    return metrics, items, y_true, scores

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("split_json")
    parser.add_argument("--model", required=True)
    parser.add_argument("--root", default=".")
    parser.add_argument("--outdir", default="results")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    print(f"Starting evaluation for model {args.model} on {args.split_json}")
    ds = "breast" if "breast" in args.split_json.lower() else ("lung" if "lung" in args.split_json.lower() else "dataset")
    metrics = []
    for fold in range(5):
        print(f"Running fold {fold}...")
        try:
            m, items, y_true, scores = run_fold(args.split_json, fold, args.model, args.root)
        except Exception as e:
            print(f"Error in fold {fold}: {e}")
            import traceback
            traceback.print_exc()
            continue
        m["fold"] = fold
        metrics.append(m)
        print(f"Fold {fold} finished. Saving results...")

        with open(os.path.join(args.outdir, f"metrics_{args.model}_{ds}_fold{fold}.json"), "w", encoding="utf-8") as f:
            json.dump(m, f, indent=2)
        rows = []
        for it, lbl, sc in zip(items, y_true, scores):
            rows.append({"drugA": it["drugA"], "drugB": it["drugB"], "drugC": it["drugC"], "label": int(lbl), "score": float(sc), "fold": int(fold)})
        pd.DataFrame(rows).to_csv(os.path.join(args.outdir, f"scores_{args.model}_{ds}_fold{fold}.csv"), index=False)
    df = pd.DataFrame(metrics)
    agg = df[["AUROC","AUPRC","F1","Precision","Recall"]].mean().to_dict()
    with open(os.path.join(args.outdir, f"metrics_{args.model}_{ds}_mean.json"), "w", encoding="utf-8") as f:
        json.dump(agg, f, indent=2)
    df.to_csv(os.path.join(args.outdir, f"metrics_{args.model}_{ds}_folds.csv"), index=False)
    # save per-item scores for significance tests
    print(json.dumps(agg, indent=2))

if __name__ == "__main__":
    main()
