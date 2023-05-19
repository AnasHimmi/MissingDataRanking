# to calculate new correlations : 
# python3 task_level_calculate_correlations.py --file=... --seed=...


# This file assumes that a lower score is a better score
# a ranking [7,0,2,...] reads as follows: S0 is ranked 8th, S1 is ranked 1st, S2 is ranked 3rd, etc.

import warnings
import random
import numpy as np
import pandas as pd
import scipy.stats as stats
import time
import argparse
import os
import json
import partial_rankings as prank
from loguru import logger

def task_level_remove_data(df,eta):
    nan_mask = (np.random.random(df.shape)<eta)
    nan_mask[:,:1] = False
    new_df = df.mask(nan_mask).copy(deep=True)
    return new_df

def one_levels_incomplete_aggregation_task_level(df,return_counts=False):
    n=df['Model'].nunique()
    p_total = np.zeros((n,n))    
    p_ranks=[]
    for column in df.columns:
        if column!="Model":
            p_ranks.append(prank.partial_scores_to_ranking(df[column]))
    for perm in p_ranks:
        p = prank.p_rank_to_mat(perm, return_ratios=False)
        p_total += p
    systems = df['Model'].values
    if return_counts:
        return prank.borda_mat(p_total), p_total.sum(axis=0), systems
    else:
        return prank.borda_mat(p_total), p_total, systems

def mean_aggregation_task_level(df):
    means = df.mean(axis=1,numeric_only=True).values
    systems = df['Model'].values
    return prank.rank_data_nan(means), means, systems


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
            # print(complete_rankings[aggregation])
            # print(incomplete_rankings[removal][aggregation])
            correlations[removal][aggregation] = stats.kendalltau(complete_rankings[aggregation],incomplete_rankings[removal][aggregation])[0]
    
    # save results
    for removal in data_removals:
        json_results["robustness"][removal]["eta"].append(eta)
        for aggregation in aggregations:
            json_results["robustness"][removal][aggregation].append({"correlation":correlations[removal][aggregation],
                                                                         "ranking":incomplete_rankings[removal][aggregation].tolist()})
    return json_results

tests = {"robustness":robustness_test}

data_removals = {"noise":task_level_remove_data}

aggregations = {"one_level":lambda df:one_levels_incomplete_aggregation_task_level(df)[0],
                "mean":lambda df:mean_aggregation_task_level(df)[0]}

if __name__ == "__main__":
    warnings.simplefilter(action='ignore', category=FutureWarning)
    warnings.simplefilter(action='ignore', category=pd.errors.ParserWarning)

    parser = argparse.ArgumentParser()
    parser.add_argument("--file", default="data_task/glue.csv", type=str)
    parser.add_argument("--seed", default="0", type=str)


    args = parser.parse_args()
    logger.info(str(args.file))
    logger.info(str(args.seed))

    file = args.file

    path = 'task_level_correlations'

    try:
        os.makedirs(path)
    except OSError:
        print(f"Creation of the directory {path} failed")
    else:
        print(f"Successfully created the directory {path}")

    corr_path = path + '/' + os.path.splitext(os.path.basename(file))[0] + '.json'


    seed = int(args.seed)
    random.seed(seed)
    np.random.seed(seed)
    etas = np.linspace(0,1,20)
    if not os.path.exists(corr_path):
        with open(corr_path, 'w') as f:
            j = {'file':file,'seed':[],'samples':0,'etas':list(etas),'results':{"robustness":{}}}
            for removal in data_removals:
                j['results']["robustness"][removal] = {}
                for aggregation in aggregations:
                    j['results']["robustness"][removal][aggregation] = []
            json.dump(j, f)
    with open(corr_path, 'r') as f:
        try:
            data = json.load(f)
            if seed in data["seed"]:
                print("seed already done")
                exit(0)
        except json.decoder.JSONDecodeError as e:
            print("json file corrupted")
            with open(corr_path, 'r') as f:
                print(f.read())
            print(e)
            exit(0)

    df = pd.read_csv(file,index_col=False).sort_values(by=['Model'], kind = 'stable')
    df[[i for i in df.columns if i!="Model"]] = df[[i for i in df.columns if i!="Model"]].astype(float)


    # compute complete rankings
    complete_rankings = {}
    for aggregation in aggregations:
        complete_rankings[aggregation] = aggregations[aggregation](df)

    json_results = {"robustness":{}}
    for removal in data_removals:
        json_results['robustness'][removal] = {"eta":[]}
        for aggregation in aggregations:
            json_results['robustness'][removal][aggregation] = []    
    i=0

    for eta in etas:
        time_sample = time.time()

        json_results = robustness_test(df,eta,json_results)        
        

        i+=1
        print(f"eta {i}/{len(etas)} done in {time.time()-time_sample} seconds")
        t = time.time()-time_sample
    with open(corr_path, 'r+') as f:
        data = json.load(f)

        # Modify the JSON data
        if seed not in data['seed']:
            data['seed'].append(seed)
            data['samples'] += 1
            for removal in data_removals:
                for aggregation in aggregations:
                    data['results']['robustness'][removal][aggregation] = data['results']['robustness'][removal][aggregation] + json_results['robustness'][removal][aggregation]
            f.seek(0)
            f.truncate()
            json.dump(data, f)


