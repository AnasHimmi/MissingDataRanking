import pandas as pd
import csv
from metrics.metrics import *
from metrics import word_embeddings
import numpy as np
import torch
from datetime import datetime
import pathlib, os
import argparse
from loguru import logger
import traceback

parser = argparse.ArgumentParser()
parser.add_argument("--metric", default="BLEU", type=str)
parser.add_argument("--dataset", default="csvs/Dialogue/DIALOGUE_pc_data.csv", type=str)
parser.add_argument("--num", default="-1", type=str)

args = parser.parse_args()
logger.info(str(args.metric))
def add_metric(csv_path,metric_path,metric,metric_name,replace = False):
    # csv_path : path to the csv to wich the function will add metric scores
    # metric_path : path to the csv that will store the metric scores
    # metric : takes a string and a list of strings (hyp,refs) and output a score
    # metric_name : a string
    # replace : replace metric column if a column with the same name already exists
    

    # df['Utterance']=df['Utterance'].astype(str)
    # df['System']=df['System'].astype(str)
    # metric_file_path = metric_path + '/' + os.path.splitext(os.path.basename(dataset))[0] + '+' + metric_name +'.csv'
    # if os.path.exists(metric_file_path) and os.stat(metric_file_path).st_size != 0:
    #     metric_df = pd.read_csv(metric_file_path)
    #     # metric_df['Utterance']=metric_df['Utterance'].astype(str)
    #     # merged_df = df.merge(metric_df, on=['System', 'Utterance'], indicator=True, how='left')
    #     # filtered_df = merged_df[merged_df['_merge'] == 'left_only']
    #     # df = filtered_df.drop('_merge', axis=1) # get only not yet calculated rows
    #     df = pd.merge(df, metric_df, on=["System","Utterance"], how="outer", indicator=True)
    #     df = df.loc[df["_merge"] == "left_only"].drop("_merge", axis=1)
    # else:
    #     with open(metric_file_path, 'w', newline='') as csvfile:
    #         writer = csv.DictWriter(csvfile, fieldnames=['System', 'Utterance',metric_name])
    #         writer.writeheader()
    #     metric_df = df[["System","Utterance"]]
    print(df)
    num_refs = 0
    for i in df.columns:
        if len(i)>3 and i[:3]=="ref":
            num_refs = num_refs +1
    print(num_refs)
    if metric_name not in ["BaryScore_all"]+comets:
        for index, row in df.iterrows():
            l = [metric(row["hyp"],row["ref"+str(i)]) for i in range(1,num_refs+1) if row["ref"+str(i)]==row["ref"+str(i)] and row["hyp"]==row["hyp"]]
            if len(l)>0:
                mm = max(l)
                print(metric_name,index,len_rows,mm,os.path.basename(csv_path),datetime.now().strftime("%H:%M:%S"))
            else:
                mm="WTF"
                print(metric_name,index,len_rows,os.path.basename(csv_path),"WTF")
            with open(metric_file_path, 'a', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=['System', 'Utterance',metric_name])
                writer.writerow({"System":row["System"],"Utterance":row["Utterance"],metric_name:mm})

    elif metric_name=="BaryScore_all":
        bary_metrics = ["baryscore_W","baryscore_SD_10","baryscore_SD_1","baryscore_SD_5"]
        for index, row in df.iterrows():
            mm = {}
            l = [metric(row["hyp"],row["ref"+str(i)]) for i in range(1,num_refs+1) if row["ref"+str(i)]==row["ref"+str(i)] and row["hyp"]==row["hyp"]]
            if len(l)>0:
                for bary_name in bary_metrics:
                    mm[bary_name] = max([k[bary_name][0] for k in l])
                    #bary_metrics[bary_name].append(mm)
                    print(bary_name,index,len_rows,mm,os.path.basename(csv_path),datetime.now().strftime("%H:%M:%S"))
            else:
                for bary_name in bary_metrics:
                    mm[bary_name] = "WTF"
                    #new_col['bary_name'].append(100)
                    print(bary_name,index,len_rows,os.path.basename(csv_path),str(100)+"WTF")
            with open(metric_file_path, 'a', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=['System', 'Utterance']+bary_metrics)
                dict = {"System":row["System"],"Utterance":row["Utterance"]}
                for bary_name in bary_metrics:
                    dict[bary_name] = mm[bary_name]
                writer.writerow(dict)
            # for bary_name in new_col:
            #     df[bary_name] = new_col[bary_name]
            # df.to_csv(csv_path,index=False)

    else:
        for index, row in df.iterrows():
            l = [metric(row["Source"],row["hyp"],row["ref"+str(i)]) for i in range(1,num_refs+1) if row["ref"+str(i)]==row["ref"+str(i)] and row["hyp"]==row["hyp"]]
            if len(l)>0:
                mm = max(l)
                print(metric_name,index,len_rows,mm,os.path.basename(csv_path),datetime.now().strftime("%H:%M:%S"))
            else:
                mm="WTF"
                print(metric_name,index,len_rows,os.path.basename(csv_path),"WTF")
            with open(metric_file_path, 'a', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=['System', 'Utterance',metric_name])
                writer.writerow({"System":row["System"],"Utterance":row["Utterance"],metric_name:mm})
    
def remove_metrics(csv_path,metric_names):
    # csv_path : path to the csv to wich the function will remove metric scores
    # metric_names: a list of columns to remove
    df = pd.read_csv(csv_path)
    df = df.drop(metric_names,axis=1)
    df.to_csv(csv_path,index=False)

def loadmetric(metric):
    return(getattr(m, metric))

metric = str(args.metric)
dataset = str(args.dataset)
num = int(args.num)
# create the directory in which the scores will be stored
path = os.path.dirname(dataset)
path = os.path.join('metric_scores', path.split('/', 1)[1])
try:
    os.makedirs(path)
except OSError:
    print(f"Creation of the directory {path} failed")
else:
    print(f"Successfully created the directory {path}")

comets = ['Comet_wmt20_comet_da','Comet_wmt21_comet_qe_mqm','Comet_eamt22_cometinho_da'] # Only run on MT


metric_classes = {
    # 'S3_pyr',
    # 'S3_resp', 
    'ROUGE_WE_1':ROUGE_WE_1,
    'ROUGE_WE_2':ROUGE_WE_2,
    'JS_1':JS_1, 
    'JS_2':JS_2, 
    'ROUGE_L':ROUGE_L, 
    'ROUGE_1':ROUGE_1,
    'ROUGE_2':ROUGE_2,           
    'BLEU':BLEU,   
    'Chrfpp':Chrfpp,              
    'BERTScore':BERTScore, 
    'MoverScore':MoverScore,
    'BaryScore_all':BaryScore_all,
    'DepthScore':DepthScore,
    'Infolm_kl':Infolm_kl,
    'Infolm_alpha':Infolm_alpha,
    'Infolm_renyi':Infolm_renyi,
    'Infolm_beta':Infolm_beta,
    'Infolm_ab':Infolm_ab,
    'Infolm_l1':Infolm_l1,
    'Infolm_l2':Infolm_l2,
    'Infolm_linf':Infolm_linf,
    'Infolm_fisher_rao':Infolm_fisher_rao,
    'CharErrorRate':CharErrorRate,
    'ExtendedEditDistance':ExtendedEditDistance,
    'MatchErrorRate':MatchErrorRate,
    'TranslationEditRate':TranslationEditRate,
    'WordErrorRate':WordErrorRate,
    'WordInfoLost':WordInfoLost,
    'Bleurt':Bleurt,
    'Comet_wmt20_comet_da':Comet_wmt20_comet_da,
    'Comet_wmt21_comet_qe_mqm':Comet_wmt21_comet_qe_mqm,
    'Comet_eamt22_cometinho_da':Comet_eamt22_cometinho_da
}

df = pd.read_csv(dataset)
#systems = df['System'].value_counts().keys()

if num==-1:
    pass
else:
    df = df.iloc[list(range(num*25,min(len(df),(num+1)*25)))]

len_rows = df.shape[0]
metric_file_path = path + '/' + os.path.splitext(os.path.basename(dataset))[0] + '+' + metric +'.csv'
if os.path.exists(metric_file_path) and os.stat(metric_file_path).st_size != 0:
    metric_df = pd.read_csv(metric_file_path)
    # metric_df['Utterance']=metric_df['Utterance'].astype(str)
    # merged_df = df.merge(metric_df, on=['System', 'Utterance'], indicator=True, how='left')
    # filtered_df = merged_df[merged_df['_merge'] == 'left_only']
    # df = filtered_df.drop('_merge', axis=1) # get only not yet calculated rows
    df = pd.merge(df, metric_df, on=["System","Utterance"], how="outer", indicator=True)
    df = df.loc[df["_merge"] == "left_only"].drop("_merge", axis=1)
else:
    with open(metric_file_path, 'w', newline='') as csvfile:
        if metric =='BaryScore_all':
            bary_metrics = ["baryscore_W","baryscore_SD_10","baryscore_SD_1","baryscore_SD_5"]
            writer = csv.DictWriter(csvfile, fieldnames=['System', 'Utterance']+bary_metrics)
        else:
            writer = csv.DictWriter(csvfile, fieldnames=['System', 'Utterance',metric])
        writer.writeheader()
    metric_df = df[["System","Utterance"]]
# faire ces checks avant d'init la métrique
# il faut charger le csv et le fichier des metriques pour verifier ce qu'il reste à calculer (ne pas charger la métrique si rien à calculer) 
# if sysnum = -1 or nbre de ligne<10000: tout calculer (cas par défaut à ne pas utiliser en prod sinon)
# elif sysnum< nbre de systemes du dataset : calculer les metrics pour ce systeme
#else : ne rien faire, arretter le programme


need_we = ['ROUGE_WE_1','ROUGE_WE_2']
if len(df)>0:
    if metric in need_we:
        word_embs = word_embeddings.load_embeddings('deps','deps.words.bz2')
        m = metric_classes[metric](we=word_embs,device='cpu')
    else:
        m = metric_classes[metric](device='cpu')

    try : 
        dir_dataset = os.path.dirname(dataset)

        add_metric(dataset,path,m.compute_score,metric,replace=False)
    except Exception as e:
        print('CATCHED ERROR')
        print(dataset,metric,e)
        print(traceback.format_exc())
else:
    print("already calculated. CLOSING")
