import json
import pandas as pd

data = {}
datasets = {"DIALOGUE_pc_data":"data/Dialogue/pc_usr_data.json","DIALOGUE_tc_data":"data/Dialogue/tc_usr_data.json"}
for dataset in datasets:
    data[dataset] = json.load(open(datasets[dataset]))

cols = ["System","Utterance","Context","Fact","ref1","hyp"]
for i in data["DIALOGUE_pc_data"][0]['responses'][0]:
    if i!="response" and i!="model":
        cols.append("H:" + i)
print(cols)

final_data = {}
for dataset in datasets:
    final_data[dataset] = {}
    for col in cols:
        final_data[dataset][col] = []
for dataset in datasets:
    for i in range(len(data[dataset])):
        for resp in data[dataset][i]['responses']:
            if resp['model']!="Original Ground Truth":
                final_data[dataset]["System"].append(resp['model'])
                final_data[dataset]["Utterance"].append(i)
                final_data[dataset]["Context"].append(data[dataset][i]['context'])
                final_data[dataset]["Fact"].append(data[dataset][i]['fact'])
                final_data[dataset]["ref1"].append(data[dataset][i]['responses'][0]["response"])
                final_data[dataset]["hyp"].append(resp["response"])
                for heval in data["DIALOGUE_pc_data"][0]['responses'][0]:
                    if heval!="response" and heval!="model":
                        final_data[dataset]["H:" + heval].append(sum(resp[heval]) / len(resp[heval]))


for dataset in datasets:
    final_df = pd.DataFrame.from_dict(final_data[dataset]).sort_values(by=['System'], kind = 'stable')

    final_df.to_csv("csvs/Dialogue/"+dataset+".csv",index=False)

