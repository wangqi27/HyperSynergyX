import pandas as pd
import re
import random
from rouge_score import rouge_scorer

# 1. Load Data
try:
    df_candidates = pd.read_csv("d:\\科研\\数据\\data\\results\\human_eval_candidates.csv")
    print(f"Loaded {len(df_candidates)} candidates.")
except FileNotFoundError:
    print("Candidates file not found.")
    exit()

# Initialize ROUGE scorer
scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
rouge_scores = []
entity_hits = 0
total_entities = 0

print("\n--- Evaluating ---")

for index, row in df_candidates.iterrows():
    # 1. Extract Drug Names from the "Drug_Combination" column
    # Format: "DrugA + DrugB + DrugC"
    combo_str = str(row['Drug_Combination'])
    drug_names = [d.strip() for d in combo_str.split('+')]
    
    # 2. Extract Explanation
    cand_text = str(row.get('Generated_Explanation', ''))
    
    # 3. Factuality (Entity Overlap)
    # Extract BE numbers from generated text
    pred_entities = re.findall(r'BE\d+', cand_text)
    
    # We verify if these entities are "real" by checking if they appear in the "KG_Context" column
    # The KG_Context column contains the raw retrieval data.
    context_text = str(row.get('KG_Context', ''))
    
    valid_entities = [e for e in pred_entities if e in context_text]
    
    if len(pred_entities) > 0:
        hit_rate = len(valid_entities) / len(pred_entities)
    else:
        hit_rate = 1.0 
        
    entity_hits += len(valid_entities)
    total_entities += len(pred_entities)
    
    # 4. ROUGE Calculation
    # Mock Reference: "The combination of A, B, C targets [List of Valid Targets]..."
    # We construct a reference using the drug names and the entities that ARE in the context.
    
    # Find all targets in context
    context_targets = re.findall(r'BE\d+', context_text)
    # Take a subset (e.g. top 5 unique) to simulate a concise summary
    unique_targets = list(set(context_targets))[:5]
    
    ref_text = f"The combination of {', '.join(drug_names)} exhibits synergistic effects. It targets {', '.join(unique_targets)} to inhibit cancer cell proliferation."
    
    scores = scorer.score(ref_text, cand_text)
    rouge_scores.append(scores['rougeL'].fmeasure)
    
    if index < 1:
        print(f"Cand: {cand_text[:50]}...")
        print(f"Ref : {ref_text[:50]}...")
        print(f"ROUGE-L: {scores['rougeL'].fmeasure:.4f}")

# 5. Compute Averages
avg_rouge = sum(rouge_scores) / len(rouge_scores) if rouge_scores else 0
factuality_score = (entity_hits / total_entities * 100) if total_entities else 0

print("\n--- Final Results ---")
print(f"Average ROUGE-L: {avg_rouge:.4f}")
print(f"Factuality Score: {factuality_score:.2f}%")
print(f"Total Entities Verified: {entity_hits}/{total_entities}")

# Save results
with open("d:\\科研\\数据\\data\\results\\nlp_metrics.txt", "w") as f:
    f.write(f"ROUGE-L: {avg_rouge:.4f}\n")
    f.write(f"Factuality: {factuality_score:.2f}%\n")
