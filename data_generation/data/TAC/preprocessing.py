import pandas as pd
import numpy as np
import codecs
import json
import tqdm

def load_json(filepath):
    with codecs.open(filepath, 'r', encoding='utf-8') as f:
        return json.loads(f.read())
    
for year in ['08', '09', '11']: 
    sys_path = 'data/TAC/tac.'+year+'.mds.gen.resp-pyr'
    dataset = list(load_json(sys_path).items())
            
    num_instances = len(dataset) # 44 instances in TAC-2009

    if year == '09':
        num_sys = 55 + 4  
    elif year == '08':
        num_sys = 58 + 4  
    elif year == '11':
        num_sys = 50 + 4  
    qa = ['responsiveness','pyr_score']
    d = {"System":[],"Utterance":[],"ref1":[],"ref2":[],"ref3":[],"ref4":[],"hyp":[],"H:responsiveness":[],"H:pyr_score":[]}


    for i in tqdm.tqdm(range(num_instances)): 
        tid, data = dataset[i]
        references = [' '.join(ref['text']) for ref in data['references']]
        j = 0
        for system in data['annotations']: 
            s = ' '.join(system['text']) 
            d["System"].append("M"+str(j))
            d["Utterance"].append(i)
            for c in range(0,4):
                if c<len(references):
                    d["ref"+str(c+1)].append(references[c])
                else:
                    d["ref"+str(c+1)].append(np.nan)
            d["hyp"].append(s)
            d["H:responsiveness"].append(float(system[qa[0]]))
            d["H:pyr_score"].append(float(system[qa[1]]))

            j += 1

    final_df = pd.DataFrame.from_dict(d).sort_values(by=['System'],key=lambda col: col.apply(lambda M:int(M[1:])), kind = 'stable')
    final_df.to_csv('data/TAC/TAC_'+year+'.csv',index=False)