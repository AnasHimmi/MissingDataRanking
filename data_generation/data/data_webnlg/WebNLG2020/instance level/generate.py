import os
import pandas as pd
import json
import numpy as np

current_dir = os.getcwd()
submissions_dir = current_dir + '/submissions'
references_dir = current_dir + '/references'
heval_dir = current_dir + '/results'

# systems = {"adaptCenter":"adaptCenter/ADAPTcentreWebNLGsubmission.txt",
#     "melbourne":"melbourne/final_result.txt",
#     "pkuwriter":"pkuwriter/PKUWriter_results.txt",
#     "tiburg_nmt":"tilburg/tilburg_nmt.txt",
#     "tilburg_smt":"tilburg/tilburg_smt.txt",
#     "uit-danglt-clnlp":"uit-danglt-clnlp/Submission-UIT-DANGNT-CLNLP.txt",
#     "upf":"upf/UPF_All_sent_final.txt"}

systems = {
    "rdf2text":{
        "en":[],
        "ru":[]
    },
    # "text2rdf":{

    #     "en":[],
    #     "ru":[]
# }
}

for r in systems:
    for lang in systems[r]:
        root = submissions_dir + '/' + r + '/' + lang
        dirlist = [ item for item in os.listdir(root) if os.path.isdir(os.path.join(root, item)) ]
        systems[r][lang] = [system for system in dirlist]

#print(systems)
# baseline = []
# with open(baseline_dir) as f:
#     baseline = f.read().splitlines()
Hm = ["H:Correctness","H:DataCoverage","H:Fluency","H:Relevance","H:TextStructure"]
data = {}
for r in systems:
    data[r] = {}
    for lang in systems[r]:
        data[r][lang] = {"System":[],"Utterance":[]}
        for c in range(1,6):
            data[r][lang]["ref"+str(c)] = []
        data[r][lang]["hyp"] = []
        for h in Hm:
            data[r][lang][h] = []


for r in systems : 
    for lang in systems[r]:
        for s in systems[r][lang] :
            with open(submissions_dir + '/' + r + '/' + lang + '/' + s + '/primary.' + lang) as f_sys:
                with open(references_dir + '/references-' + lang + '.json') as f_ref:
                    with open(heval_dir + '/' + lang + '/' + s + '/primary.json') as f_heval:
                        hyps = f_sys.read().splitlines()
                        refs = json.load(f_ref)
                        heval = json.load(f_heval)
                        for i in range(len(hyps)):
                            if hyps[i]not in ["null",""]: # some outputs from uit-danglt-clnlp are null
                                data[r][lang]["System"].append(s)
                                data[r][lang]["Utterance"].append(i)
                                temp_ref = []
                                for re in refs["entries"][i][str(i+1)]['lexicalisations'] :
                                    temp_ref.append(re["lex"])
                                for c in range(1,6):
                                    if c<=len(temp_ref):
                                        data[r][lang]["ref"+str(c)].append(temp_ref[c-1])
                                    else:
                                        data[r][lang]["ref"+str(c)].append(np.nan)
                                data[r][lang]["hyp"].append(hyps[i])
                                if str(i+1) in heval and len(heval[str(i+1)])>0:
                                    #print(heval[str(i+1)])
                                    for h in Hm:
                                        data[r][lang][h].append(sum(float(heval[str(i+1)][d][h[2:]])for d in heval[str(i+1)]) / len(heval[str(i+1)]))
                                else:
                                    for h in Hm:
                                        data[r][lang][h].append(np.nan)

                            # for he in temp_heval:
                            #     data[r][lang][he] = temp_heval[he]

for r in systems:
    for lang in systems[r]:
        final_data = pd.DataFrame(data[r][lang])

        final_data.to_csv("WebNLG2020_"+r+"_"+lang+".csv",index=False)

#for s in systems