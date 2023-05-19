import json
import glob
import pandas as pd
import collections

# read all json files that contain "scores" in "scores_and_outputs/GEM-outputs/"
files = glob.glob("scores_and_outputs/GEM-outputs/*.scores.json") # each file is a system output
json_files = {file:json.load(open(file)) for file in files} # scores for each system in many tasks


tasks = {file:[k for k in list(json_files[file].keys()) if type(json_files[file][k]) is dict and "predictions_file" in json_files[file][k]] 
         for file in json_files} # tasks for each system


# file = "scores_and_outputs/GEM-outputs/t5-small2.scores.json"
# print(tasks[file])

task_scores = {} # task_scores[task][system][metric] = score
for file in json_files:
    system = file.split("/")[-1].split(".")[0]
    for task in tasks[file]:
        if task not in task_scores:
            task_scores[task] = {}
        if system not in task_scores[task]:
            task_scores[task][system] = {}
        for m in json_files[file][task]:
            if type(json_files[file][task][m]) is dict:
                for mm in json_files[file][task][m]:
                    task_scores[task][system][m+"_"+mm] = json_files[file][task][m][mm]
            else:
                task_scores[task][system][m] = json_files[file][task][m]

scores = {} # scores[system][task][metric] = score

for task in task_scores:
    for system in task_scores[task]:
        if system not in scores:
            scores[system] = {}
        if task not in scores[system]:
            scores[system][task] = {}
        for metric in task_scores[task][system]:
            if metric not in ["predictions_file","N","total_length","references_file"]:
                scores[system][task][metric] = task_scores[task][system][metric]
scores_mean = {} # scores[system][metric] = score

for system in scores:
    scores_mean[system] = {}
    for task in scores[system]:
        for metric in scores[system][task]:
            if metric not in scores_mean[system]:
                scores_mean[system][metric] = []
            scores_mean[system][metric].append(scores[system][task][metric])
    for metric in scores_mean[system]:
        scores_mean[system][metric] = sum(scores_mean[system][metric])/len(scores_mean[system][metric])
 
df = pd.DataFrame.from_dict(scores_mean,orient="index")
df.reset_index(inplace=True)
df = df.rename(columns = {'index':'Model'})
df = df.sort_values(by=['Model'])
df.reset_index(inplace=True,drop=True)
# drop rows that have less than 67 non nan values
df.dropna(axis=0,thresh=67,inplace=True)
print(df.shape)
# drop columns with nans
df.dropna(axis=1,inplace=True)
print(df.shape)
print(df.columns)
df.to_csv("gem.csv",index=False)
# dfs = {}
# for task in task_scores: # first column are systems
#     try:
#         dfs[task] = pd.DataFrame.from_dict(task_scores[task],orient="index")
#         dfs[task].reset_index(inplace=True)
#         dfs[task] = dfs[task].rename(columns = {'index':'Model'})
#         dfs[task] = dfs[task].sort_values(by=['Model'])
#         dfs[task].reset_index(inplace=True,drop=True)
#         # get number of nan values in each column
#         nan_values = {}
#         for column in dfs[task].columns:
#             nan_values[column] = dfs[task][column].isnull().sum()
#         second_lowest = sorted(nan_values.values())[1]
#         for column in dfs[task].columns:
#             if nan_values[column] > second_lowest:
#                 dfs[task].drop(column,axis=1,inplace=True)
#         # drop all systems with NaNs
#         dfs[task].dropna(axis=0,inplace=True)
#         # drop these columns if exists predictions_file,N,total_length,references_file
#         if "predictions_file" in dfs[task].columns:
#             dfs[task].drop("predictions_file",axis=1,inplace=True)
#         if "N" in dfs[task].columns:
#             dfs[task].drop("N",axis=1,inplace=True)
#         if "total_length" in dfs[task].columns:
#             dfs[task].drop("total_length",axis=1,inplace=True)
#         if "references_file" in dfs[task].columns:
#             dfs[task].drop("references_file",axis=1,inplace=True)
#     except Exception as e:
#         # print(collections.Counter([len(task_scores_final[task][k]) for k in task_scores_final[task]]))
#         print(task)
#         print(e)


# # write to csv all dfs with no NaNs
# for task in dfs:
#     # if number of columns is lower than 2, drop it
#     if len(dfs[task].columns) < 3:
#         print(task)
#     else:
#         try:
#             #replace "/" with "-" in task name
#             #dfs[task].to_csv("csvs/"+task+".csv",index=False)
#             dfs[task].to_csv("csvs/"+task.replace("/","-")+".csv",index=False)
#         except Exception as e:
#             print(task)
#             print(e)

# # dropped : 
# # dart_validation
# # common_gen_test
# # wiki_auto_asset_turk_test_asset_contrast_challenge_syncomp_simpl-Level1
# # wiki_auto_asset_turk_challenge_train_sample
# # dart_test
# # wiki_auto_asset_turk_validation
# # common_gen_validation
# # totto_validation
# # web_nlg_ru_challenge_train_sample
# # mlsum_de_challenge_train_sample
# # schema_guided_dialog_challenge_train_sample
# # web_nlg_en_challenge_train_sample
# # xsum_challenge_train_sample
# # mlsum_es_challenge_train_sample
# # e2e_nlg_challenge_train_sample
# # common_gen_challenge_train_sample
# # cs_restaurants_challenge_train_sample
# # common_gen_challenge_test_scramble
# # totto_challenge_train_sample
# # totto_challenge_test_scramble
# # dart_val
# # web_nlg_en_challenge_test