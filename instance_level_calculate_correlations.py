# to calculate new correlations : 
# python3 instance_level_calculate_correlations.py --file=... --sample=...


# This file assumes that a lower score is a better score
# a ranking [7,0,2,...] reads as follows: S0 is ranked 8th, S1 is ranked 1st, S2 is ranked 3rd, etc.


import numpy as np
import pandas as pd
import scipy.stats as stats
import partial_rankings as prank
import warnings
from loguru import logger
import argparse
import os
import random
import json
import time
metric_names = [
    'ROUGE_WE_1',
    'ROUGE_WE_2',
    'JS_1', 
    'JS_2', 
    'ROUGE_L', 
    'ROUGE_1', 
    'ROUGE_2',           
    'BLEU',   
    'Chrfpp',              
    'BERTScore', 
    'MoverScore',
    'baryscore_W',
    'baryscore_SD_10',
    'baryscore_SD_1',
    'baryscore_SD_5',
    'DepthScore',
    'Infolm_kl',
    'Infolm_alpha',
    'Infolm_renyi',
    'Infolm_beta',
    'Infolm_ab',
    'Infolm_l1',
    'Infolm_l2',
    'Infolm_linf',
    'Infolm_fisher_rao',
    'CharErrorRate',
    'ExtendedEditDistance',
    'MatchErrorRate',
    'TranslationEditRate',
    'WordErrorRate',
    'WordInfoLost',
    'Bleurt',
    'Comet_wmt20_comet_da',
    'Comet_wmt21_comet_qe_mqm',
    'Comet_eamt22_cometinho_da'
]

# removes random columns for each system
def instance_level_remove_system_column(df,eta):
    new_df = df.copy(deep=True)
    for system in df['System'].unique():
        for column in df.columns[2:]:
            if np.random.random() < eta:
                new_df.loc[new_df['System'] == system, column] = np.nan
    return new_df


# partial borda aggregation according to a single metric
def incomplete_ranking_aggregation(df,metric):
    n=df['System'].nunique()
    p_total = np.zeros((n,n))    
    grouped_df = df.groupby('Utterance')
    p_ranks = [prank.partial_scores_to_ranking(g[metric].values.tolist()) for u, g in grouped_df]
    for perm in p_ranks:
        p = prank.p_rank_to_mat(perm, return_ratios=True)
        p_total += p
    return prank.borda_mat(p_total)

# partial one level borda aggregation
def one_levels_incomplete_aggregation(df,return_counts=False):
    n=df['System'].nunique()
    p_total = np.zeros((n,n))       
    p_ranks=[]
    for metric in metric_names:
        if metric in df.columns:
            grouped_df = df.groupby('Utterance')
            p_ranks = p_ranks + [prank.partial_scores_to_ranking(g[metric].values.tolist()) for u, g in grouped_df]
    for perm in p_ranks:
        p = prank.p_rank_to_mat(perm, return_ratios=True)
        p_total += p
    systems = df['System'].unique()
    if return_counts:
        return prank.borda_mat(p_total), p_total.sum(axis=0), systems
    else:
        return prank.borda_mat(p_total), p_total, systems

# partial two levels borda aggregation
def two_levels_incomplete_aggregation(df):
    rankings_by_metrics = []
    for metric in metric_names:
        if metric in df.columns:
            rankings_by_metrics.append(incomplete_ranking_aggregation(df, metric))
    r = np.array(rankings_by_metrics)
    borda_count = r.sum(axis=0)
    systems = df['System'].unique()
    #return np.argsort(prank.rand_argsort(borda_count)), borda_count, systems
    return prank.rank_data_nan(borda_count), borda_count, systems

# mean aggregation
def mean_aggregation_instance_level(df):
    means = df.loc[:, df.columns != 'Utterance'].groupby('System').mean().mean(axis=1).values
    systems = df.loc[:, df.columns != 'Utterance'].groupby('System').mean().index.values
    return prank.rank_data_nan(means), means, systems

# normalized mean aggregation
# the metrics are normalized according to the lowest and highest value in each metric
def normalized_mean_aggregation_instance_level(df):
    temp_df = df.loc[:, df.columns != 'Utterance'].copy(deep=True)
    for metric in metric_names:
        if metric in df.columns:
            if temp_df[metric].isnull().sum() < temp_df[metric].size:
                temp_df[metric] = (temp_df[metric] - np.nanmin(temp_df[metric])) / (np.nanmax(temp_df[metric]) - np.nanmin(temp_df[metric]))
    means = temp_df.groupby('System').mean().mean(axis=1).values
    systems = temp_df.groupby('System').mean().index.values
    return prank.rank_data_nan(means), means, systems


data_removals = {"system-col":instance_level_remove_system_column,}

aggregations = {"two_level":lambda df:two_levels_incomplete_aggregation(df)[0],
                "one_level":lambda df:one_levels_incomplete_aggregation(df)[0],
                "mean":lambda df:mean_aggregation_instance_level(df)[0],
                "normalized_mean":lambda df:normalized_mean_aggregation_instance_level(df)[0]}


# compares the rankings of the complete and incomplete data
def robustness_test(df,eta,json_results):
    new_dfs = {}
    for removal in data_removals:
        new_dfs[removal] = data_removals[removal](df,eta)

    # compute incomplete rankings
    incomplete_rankings = {}
    for removal in data_removals:
        incomplete_rankings[removal] = {}
        for aggregation in aggregations:
            incomplete_rankings[removal][aggregation] = aggregations[aggregation](new_dfs[removal])

    # compute correlations
    correlations = {}
    for removal in data_removals:
        correlations[removal] = {}
        for aggregation in aggregations:
            correlations[removal][aggregation] = stats.kendalltau(complete_rankings[aggregation],incomplete_rankings[removal][aggregation])[0]
    
    # save results
    for removal in data_removals:
        json_results["robustness"][removal]["eta"].append(eta)
        json_results["robustness"][removal]["seeds"].append(seed)
        for aggregation in aggregations:
            json_results["robustness"][removal][aggregation].append({"correlation":correlations[removal][aggregation],
                                                                         "ranking":incomplete_rankings[removal][aggregation].tolist()})
    return json_results

tests = {"robustness":robustness_test}

# returns the utterances that are common to all systems
def get_common_utterances(df):
    grouped = df.groupby('System')
    common_utterances = set(grouped.get_group(list(grouped.groups.keys())[0])['Utterance'])
    for _, group in list(grouped)[1:]:
        common_utterances = common_utterances.intersection(set(group['Utterance']))
    return set(common_utterances)



# load and preprocess data        
def load_file(file):
    df = pd.read_csv(file,index_col=False).sort_values(by=['System'], kind = 'stable') # sort alphabetically by system
    droplist = [i for i in df.columns if i not in metric_names+['System','Utterance']] # get columns that are not metrics
    df.drop(droplist,axis=1,inplace=True) # drop columns that are not metrics
    df[[i for i in metric_names if i in df.columns]] = df[[i for i in metric_names if i in df.columns]].astype(float) # convert to float
    utterances = get_common_utterances(df) # get the common utterances
    df = df[df['Utterance'].isin(utterances)] # keep only the common utterances
    return df

# compute complete rankings
def compute_complete_rankings(df):
    complete_rankings = {}
    for aggregation in aggregations:
        complete_rankings[aggregation] = aggregations[aggregation](df)
    return complete_rankings

# python3 

if __name__ == "__main__":
    warnings.simplefilter(action='ignore', category=FutureWarning)
    warnings.simplefilter(action='ignore', category=pd.errors.ParserWarning)

    parser = argparse.ArgumentParser()
    parser.add_argument("--file", default="final_df/Dialogue/DIALOGUE_pc_data.csv", type=str)
    parser.add_argument("--sample", default="0", type=str)

    args = parser.parse_args()
    logger.info(str(args.file))
    logger.info(str(args.sample))

    file = args.file

    path = os.path.dirname(file)
    path = os.path.join('instance_level_correlations', path.split('/', 1)[1])
    try:
        os.makedirs(path)
    except OSError:
        print(f"Creation of the directory {path} failed, already exists")
    else:
        print(f"Successfully created the directory {path}")

    sample = int(args.sample)

    corr_path = path + '/' + os.path.splitext(os.path.basename(file))[0] + '_' + str(sample)+ '.json'

    len_etas = 21
    seed = sample*len_etas

    random.seed(seed)
    np.random.seed(seed)
    etas = np.linspace(0,1,len_etas)

    # create json file if it does not exist
    if not os.path.exists(corr_path):
        with open(corr_path, 'w') as f:
            j = {'file':file,'sample':-1,'etas':list(etas),'results':{"robustness":{}}}
            for removal in data_removals:
                j['results']["robustness"][removal] = {"eta":[],"seeds":[]}
                for aggregation in aggregations:
                    j['results']["robustness"][removal][aggregation] = []
            json.dump(j, f)
    else:
        print("file already exists")
        json_data = {}
    
    with open(corr_path, 'r') as f:
        json_data = json.load(f)
            

    df = load_file(file)


    # compute complete rankings
    complete_rankings = compute_complete_rankings(df)

    # initialize json_results
    json_results = json_data["results"]


    # compute correlations for each eta
    i=0
    for eta in etas:
        # the seed changes for each iteration and each sample
        random.seed(seed)
        np.random.seed(seed)
        # if not already computed
        if seed not in json_data["results"]["robustness"][list(data_removals.keys())[0]]["seeds"]:
            time_sample = time.time()

            json_results = robustness_test(df,eta,json_results)

            with open(corr_path, 'r+') as f:
                # Read the JSON data
                json_data = json.load(f)
                # Update the JSON data
                json_data['sample']=sample
                json_data['results'] = json_results
                # Write the JSON data back to the file
                f.seek(0)
                json.dump(json_data, f)
                f.truncate()

            i+=1
            print(f"eta {i}/{len(etas)} done in {time.time()-time_sample} seconds")
            t = time.time()-time_sample
        else:
            print(f"eta {i+1}/{len(etas)} already computed")
            i+=1
        seed = seed+1

