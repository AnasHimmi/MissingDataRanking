from os import listdir
import re
import numpy as np
import pandas as pd

# getting raw system outputs
L = ["A","B","C","D"]


data = {}

datasets = listdir("system-outputs")

for dataset in datasets:
    data[dataset] = {}
    languages = listdir("system-outputs/"+dataset)
    for l in languages:
        data[dataset][l] = {}
        systems = listdir(f"system-outputs/{dataset}/{l}")
        for s in systems:
            system = re.findall("hyp.(.*)."+l[3:],s)
            if(len(system)>0):
                with open(f"system-outputs/{dataset}/{l}/{s}") as f:
<<<<<<< HEAD
                    data[dataset][l][s] = {"name":system[0],"data":f.read().splitlines()}
=======
                    data[dataset][l][s] = {"name":system[0],"data":f.readlines()}
>>>>>>> origin

# getting raw references
refs = {}
for dataset in datasets:
    refs[dataset] = {}
    for l in data[dataset]:
        refs[dataset][l] = {}
ref_files = listdir("references")
for ref_file in ref_files:
    dataset = ref_file.split('.')[0]
    l = re.findall(dataset+".(.+?).ref.ref-",ref_file)[0]
    num_ref = re.findall(l+".ref.(.+?)."+l[3:],ref_file)[0]
    with open(f"references/{ref_file}") as f:
<<<<<<< HEAD
        refs[dataset][l][num_ref] = f.read().splitlines()
=======
        refs[dataset][l][num_ref] = f.readlines()
>>>>>>> origin

# getting raw sources
sources = {}
for dataset in datasets:
    sources[dataset] = {}
    for l in data[dataset]:
        with open(f"sources/{dataset}.{l}.src.{l[:2]}") as f:
<<<<<<< HEAD
            sources[dataset][l] = f.read().splitlines()
=======
            sources[dataset][l] = f.readlines()
>>>>>>> origin

# for i in refs["florestest2021"]["hi-bn"]["ref-A"]:
#     print(i)
d = {}
for dataset in datasets:
    d[dataset] = {}
    for l in data[dataset]:
        d[dataset][l] = {"System":[],"Utterance":[],"Source":[],"ref1":[],"ref2":[],"ref3":[],"ref4":[],"hyp":[]}

for dataset in datasets:
    for l in data[dataset]:
        for s in data[dataset][l]:
            for i in range(len(data[dataset][l][s]["data"])):
                d[dataset][l]["System"].append(data[dataset][l][s]["name"])
                d[dataset][l]["Utterance"].append(i)
                d[dataset][l]["Source"].append(sources[dataset][l][i])
                for c in range(1,5):
                    if c<=len(refs[dataset][l]):
                        d[dataset][l]["ref"+str(c)].append(refs[dataset][l]["ref-"+L[c-1]][i])
                    else:
                        d[dataset][l]["ref"+str(c)].append(np.nan)
                d[dataset][l]["hyp"].append(data[dataset][l][s]["data"][i])


for dataset in datasets:
    for l in data[dataset]:
        final_df = pd.DataFrame(d[dataset][l])
        final_df.dropna(how='all', axis=1, inplace=True)
        final_df.to_csv(f"csvs/{dataset}.{l}.data.csv",index=False)

# src = {}
# for dataset in datasets:
#     src[dataset] = {}

