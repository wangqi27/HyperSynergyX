import os
import csv
import json
from neo4j import GraphDatabase

def env(key, default=None):
    v = os.environ.get(key)
    return v if v else default

def connect():
    uri = env('NEO4J_URI','bolt://localhost:7687')
    user = env('NEO4J_USER','neo4j')
    password = env('NEO4J_PASS','neo4j')
    return GraphDatabase.driver(uri, auth=(user, password))

def load_drug_mapping(path):
    m = {}
    if not os.path.isfile(path):
        return m
    with open(path,'r',encoding='utf-8') as f:
        r = csv.DictReader(f)
        for row in r:
            name = str(row.get('Drug') or row.get('name') or '').strip()
            dbid = str(row.get('DBID') or row.get('id') or '').strip()
            if name:
                m[name] = {'name': name, 'dbid': dbid}
    return m

def ensure_schema(tx):
    tx.run("CREATE CONSTRAINT IF NOT EXISTS FOR (d:Drug) REQUIRE d.name IS UNIQUE")
    tx.run("CREATE CONSTRAINT IF NOT EXISTS FOR (t:Target) REQUIRE t.name IS UNIQUE")
    tx.run("CREATE CONSTRAINT IF NOT EXISTS FOR (p:Paper) REQUIRE p.url IS UNIQUE")
    tx.run("CREATE CONSTRAINT IF NOT EXISTS FOR (h:HyperEdge) REQUIRE h.uid IS UNIQUE")

def upsert_drug(tx, name, props=None):
    tx.run("MERGE (d:Drug {name:$name}) SET d += $props", name=name, props=props or {})

def upsert_paper(tx, url, title=None):
    tx.run("MERGE (p:Paper {url:$url}) SET p.title=$title", url=url, title=title)

def create_hyperedge_tx(tx, uid, htype, drugs, props):
    tx.run("MERGE (h:HyperEdge {uid:$uid}) SET h.type=$type, h.drugs=$drugs, h += $props",
           uid=uid, type=htype, drugs=drugs, props=props)
    for d in drugs:
        tx.run("MATCH (h:HyperEdge {uid:$uid}), (x:Drug {name:$name}) MERGE (h)-[:HAS_DRUG]->(x)", uid=uid, name=d)

def read_topk_scores(results_dir, ds, model_list, topk=50):
    items = []
    import glob
    for m in model_list:
        for f in glob.glob(os.path.join(results_dir, f"scores_{m}_{ds}_fold*.csv")):
            with open(f,'r',encoding='utf-8') as fp:
                r = csv.DictReader(fp)
                for row in r:
                    items.append({
                        'drugA': row['drugA'], 'drugB': row['drugB'], 'drugC': row['drugC'],
                        'label': int(row['label']), 'score': float(row['score']), 'model': m
                    })
    items.sort(key=lambda x: x['score'], reverse=True)
    return items[:topk]

def main():
    driver = connect()
    mapping = load_drug_mapping(os.path.join('RAG','DB_Drug_1.csv'))
    with driver.session() as sess:
        sess.execute_write(ensure_schema)
        # upsert drugs from mapping
        for name, props in mapping.items():
            sess.execute_write(upsert_drug, name, props)
        # candidate hyperedges from topK
        models = ['dbrwh_breast','pair_train','higsyn','deepsynergy','gcn','gat']
        for ds in ['breast','lung']:
            tops = read_topk_scores('results', ds, models, topk=50)
            for it in tops:
                drugs = [str(it['drugA']), str(it['drugB']), str(it['drugC'])]
                uid = f"{ds}:{it['model']}:{'|'.join(drugs)}"
                props = {'pred_score': round(it['score'],4), 'label': it['label'], 'source': it['model'], 'dataset': ds}
                sess.execute_write(create_hyperedge_tx, uid, 'CandidateTriplet', drugs, props)
    print('Neo4j hyperedges loaded')

if __name__=='__main__':
    main()

