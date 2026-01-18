import os
import csv
import numpy as np
from sim_builder import build_feature_map_atc, build_feature_map_target

def tokenize(s):
    return [p.strip() for p in str(s).split(',') if p.strip()]

def jaccard(a, b):
    if not a or not b:
        return 0.0
    ia = len(a.intersection(b))
    ua = len(a.union(b))
    return float(ia)/float(ua) if ua>0 else 0.0

def mean_pair_jaccard(sets):
    vals = []
    for i in range(len(sets)):
        for j in range(i+1, len(sets)):
            vals.append(jaccard(sets[i], sets[j]))
    return float(np.mean(vals)) if vals else 0.0

def load_topk(results_dir, ds, models, topk=50):
    items = []
    import glob
    for m in models:
        for f in glob.glob(os.path.join(results_dir, f"scores_{m}_{ds}_fold*.csv")):
            with open(f,'r',encoding='utf-8') as fp:
                r = csv.DictReader(fp)
                for row in r:
                    items.append({'drugA':row['drugA'],'drugB':row['drugB'],'drugC':row['drugC'],'label':int(row['label']),'score':float(row['score']),'model':m})
    items.sort(key=lambda x:x['score'], reverse=True)
    return items[:topk]

def build_maps(root, ds):
    if ds=='breast':
        atc_path = os.path.join(root, 'ComInfo_v1','ATC','Full','ATC_Breast.xlsx')
        tgt_path = os.path.join(root, 'ComInfo_v1','Target','Full','Target_Breast.xlsx')
    else:
        atc_path = os.path.join(root, 'ComInfo_v1','ATC','Full','ATC_Lung.xlsx')
        tgt_path = os.path.join(root, 'ComInfo_v1','Target','Full','Target_Lung.xlsx')
    fmap_atc = build_feature_map_atc(atc_path)
    fmap_tgt = build_feature_map_target(tgt_path)
    return fmap_atc, fmap_tgt

def explain(root, results_dir, ds, out_csv):
    models = ['dbrwh_'+ds,'pair_train','higsyn','deepsynergy','gcn','gat']
    tops = load_topk(results_dir, ds, models, topk=50)
    fmap_atc, fmap_tgt = build_maps(root, ds)
    rows = []
    for it in tops:
        drugs = [str(it['drugA']), str(it['drugB']), str(it['drugC'])]
        atc_sets = [set(fmap_atc.get(d, set())) for d in drugs]
        tgt_sets = [set(fmap_tgt.get(d, set())) for d in drugs]
        atc_div = 1.0 - mean_pair_jaccard(atc_sets)
        tgt_comp = 1.0 - mean_pair_jaccard(tgt_sets)
        expl = 'Targets partly complementary; ATC diverse.'
        rows.append({
            'Dataset': ds,
            'drugA': drugs[0], 'drugB': drugs[1], 'drugC': drugs[2],
            'pred_score': round(it['score'],4), 'model': it['model'],
            'ATC_diversity': round(atc_div,4), 'Target_complement': round(tgt_comp,4),
            'evidence_count': 0, 'sources': 'localKB', 'explanation': expl
        })
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    import pandas as pd
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(out_csv)

def main():
    root = '.'
    results_dir = 'results'
    explain(root, results_dir, 'breast', os.path.join(results_dir, 'explanations_topk_breast.csv'))
    explain(root, results_dir, 'lung', os.path.join(results_dir, 'explanations_topk_lung.csv'))

if __name__=='__main__':
    main()

