import os
import pandas as pd
from xml.dom import minidom
import numpy as np

# get directories' path
current_dir = os.getcwd()
submissions_dir = current_dir + '/submissions'
refs_file = current_dir + '/testdata_with_lex.xml'

# system paths
systems = {"adaptCenter":"adaptCenter/ADAPTcentreWebNLGsubmission.txt",
    "melbourne":"melbourne/final_result.txt",
    "pkuwriter":"pkuwriter/PKUWriter_results.txt",
    "tiburg_nmt":"tilburg/tilburg_nmt.txt",
    "tilburg_smt":"tilburg/tilburg_smt.txt",
    "uit-danglt-clnlp":"uit-danglt-clnlp/Submission-UIT-DANGNT-CLNLP.txt",
    "upf":"upf/UPF_All_sent_final.txt",
    "baseline":"baseline_sorted.txt"}

# get refs
refs = {}
d = minidom.parse(refs_file)
entries = d.getElementsByTagName('entry') 
for e in entries:
    e_refs = e.getElementsByTagName('lex')
    refs[e.attributes["eid"].value] = []
    for e_ref in e_refs:
        refs[e.attributes["eid"].value].append(e_ref.firstChild.data)

# init data
data = {"System":[],"Utterance":[]}
for c in range(1,9):
    data["ref"+str(c)] = []
data["hyp"] = []

for s in systems :
    with open(submissions_dir + '/' + systems[s]) as f:
        hyps = f.read().splitlines()
        for i in range(len(hyps)):
            if hyps[i]!="null": # some outputs from uit-danglt-clnlp are null
                data["System"].append(s)
                data["Utterance"].append(i)
                for c in range(1,9):
                    if c <= len(refs["Id"+str(i+1)]):
                        data["ref"+str(c)].append(refs["Id"+str(i+1)][c-1])
                    else:
                        data["ref"+str(c)].append(np.nan)
                data["hyp"].append(hyps[i])

final_data = pd.DataFrame(data)
final_data.to_csv("WebNLG2017.csv",index=False)

#for s in systems