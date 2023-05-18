import json
import pandas as pd
import os
import numpy as np
import tqdm

def load_json(filepath):
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))  
    return data

sys_path = 'data/SUMM/system-outputs'
sys_infos = []
for root, directories, files in os.walk(sys_path):
    sys_name = root.split('/')[-2]
    for f in files:
        if '.jsonl' in f:
            sys_infos.append((os.path.join(root, f), sys_name))


crossing_ids = set()
for sys_path, sys_name in sys_infos:
    cur_ids = set([instance['id'] for instance in load_json(sys_path) if 'decoded' in instance and 'reference' in instance])
    if len(crossing_ids) == 0:
        crossing_ids = cur_ids
    else:
        crossing_ids = crossing_ids & cur_ids

num_instances = len(list(crossing_ids))
    
crossing_ids = list(crossing_ids)[:num_instances]

num_sys = len(sys_infos)

data = {"System":[],"Utterance":[],"ref1":[],"hyp":[]}

for i in tqdm.tqdm(range(len(sys_infos))):

    sys_path, sys_name = sys_infos[i]

    id2text = dict()
    for instance in load_json(sys_path):
        if 'decoded' in instance and 'reference' in instance:
            id2text[instance['id']] = (instance['decoded'], instance['reference'])

    subset = [id2text[k] for k in crossing_ids]
    for j, (r, s) in enumerate(subset): 
        data["System"].append(sys_name)
        data["Utterance"].append(j)
        data["ref1"].append(r)
        data["hyp"].append(s)

final_df = pd.DataFrame.from_dict(data).sort_values(by=['System'],key=lambda col: col.apply(lambda M:int(M[1:])), kind = 'stable')
final_df.to_csv('data/SUMM/SUMM.csv',index=False)