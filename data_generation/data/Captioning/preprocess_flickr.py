import tqdm
import pandas as pd
import numpy as np
import json

def load_json(filepath):
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))  
    return data

sys_path = 'data/Captioning/flickr.json'
dataset = load_json(sys_path)[0]
    

dataset = {int(k): v for k,v in dataset.items()}

per_instance, data = [], []
prev = ''
for i in range(len(dataset)):
    if not prev: 
        prev = dataset[i]['refs'][0]
    cur = dataset[i]['refs'][0]

    if prev == cur:
        per_instance.append(dataset[i])
    else:
        prev = dataset[i]['refs'][0]
        data.append(per_instance)
        per_instance = []
        per_instance.append(dataset[i])

num_sys = 8
dataset = [data[i] for i in range(len(data)) if len(data[i]) == num_sys]

num_instances = len(dataset) 


data = {"System":[],"Utterance":[],"ref1":[],"ref2":[],"ref3":[],"ref4":[],"ref5":[],"hyp":[],"H:score":[]}
print(dataset[0][0])
for i in tqdm.tqdm(range(num_instances)): 
    
    per_instance = dataset[i]

    for j in range(0, len(per_instance)):
#                score = 0.0 
        references = per_instance[j]['refs']  

        s = per_instance[j]['cand'][0]
        
        data["System"].append("M"+str(j))
        data["Utterance"].append(i)
        for c in range(0,5):
            if c<len(references):
                data["ref"+str(c+1)].append(references[c])
            else:
                data["ref"+str(c+1)].append(np.nan)
        data["hyp"].append(s)
        data["H:score"].append(per_instance[j]['score'])
        

final_df = pd.DataFrame.from_dict(data).sort_values(by=['System'],key=lambda col: col.apply(lambda M:int(M[1:])), kind = 'stable')
final_df.to_csv('csvs/Captioning/Flickr_data.csv',index=False)
