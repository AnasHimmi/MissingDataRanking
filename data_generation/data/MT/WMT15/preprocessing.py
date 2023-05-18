from os import listdir
import re
import numpy as np
import pandas as pd

L = ["A","B","C","D"]

# getting raw system outputs


data = {}

datasets = listdir("system-outputs")

for dataset in datasets:
    data[dataset] = {}
    languages = listdir("system-outputs/"+dataset)
    for l in languages:
        data[dataset][l] = {}
        systems = listdir(f"system-outputs/{dataset}/{l}")
        for s in systems:
            system = s[len(dataset)+1:len(s)-6]
            with open(f"system-outputs/{dataset}/{l}/{s}") as f:
                data[dataset][l][s] = {"name":system,"data":f.read().splitlines()}

# getting raw references
refs = {}
for dataset in datasets:
    refs[dataset] = {}

ref_files = listdir("references")
for ref_file in ref_files:
    l = ref_file[len(datasets[0])+1:len(datasets[0])+5]
    with open(f"references/{ref_file}") as f:
        refs[dataset][l[:2]+'-'+l[2:]] = f.read().splitlines()

# getting raw sources
sources = {}
for dataset in datasets:
    sources[dataset] = {}

source_files = listdir("sources")
for source_file in source_files:
    if source_file[:len(datasets[0])] == datasets[0]:
        l = source_file[len(datasets[0])+1:len(datasets[0])+5]
        with open(f"sources/{source_file}") as f:
            print(l[:2]+'-'+l[2:])
            sources[dataset][l[:2]+'-'+l[2:]] = f.read().splitlines()
# for i in refs["florestest2021"]["hi-bn"]["ref-A"]:
#     print(i)
d = {}
for dataset in datasets:
    d[dataset] = {}
    for l in data[dataset]:
        d[dataset][l] = {"System":[],"Utterance":[],"Source":[],"ref1":[],"hyp":[]}

for dataset in datasets:
    for l in data[dataset]:
        for s in data[dataset][l]:
            for i in range(len(data[dataset][l][s]["data"])):
                if refs[dataset][l][i]!="NO REFERENCE AVAILABLE":
                    d[dataset][l]["System"].append(data[dataset][l][s]["name"])
                    d[dataset][l]["Utterance"].append(i)
                    d[dataset][l]["Source"].append(sources[dataset][l][i])
                    d[dataset][l]["ref1"].append(refs[dataset][l][i])
                    d[dataset][l]["hyp"].append(data[dataset][l][s]["data"][i])


for dataset in datasets:
    for l in data[dataset]:
        final_df = pd.DataFrame(d[dataset][l])
        final_df.dropna(how='all', axis=1, inplace=True)
        final_df.to_csv(f"csvs/{dataset}.{l}.data.csv",index=False)

# src = {}
# for dataset in datasets:
#     src[dataset] = {}

