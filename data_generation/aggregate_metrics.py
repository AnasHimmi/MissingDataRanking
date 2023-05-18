import pandas as pd
import os
import glob
import pathlib 

files = [fn for fn in glob.glob(r'csvs/**/*.csv',recursive=True) 
         if not 'system_level' in fn]
metric_names_1 = [
    # 'S3_pyr',
    # 'S3_resp', 
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
    'BaryScore_all',
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
    'Bleurt']
comet = ['Comet_wmt20_comet_da',
    'Comet_wmt21_comet_qe_mqm',
    'Comet_eamt22_cometinho_da']

for dataset in files:
    path = os.path.dirname(dataset)
    path = os.path.join('metric_scores', path.split('/', 1)[1])
    df = pd.read_csv(dataset)
    total = len(df)
    if pathlib.Path(dataset).parts[1]=='MT':metrics=metric_names_1+comet
    else:metrics=metric_names_1

    for metric in metrics:
        metric_file_path = path + '/' + os.path.splitext(os.path.basename(dataset))[0] + '+' + metric +'.csv'
        try:
            dfm = pd.read_csv(metric_file_path)
            merged_df = pd.concat([dfm.set_index(['System', 'Utterance']), df.set_index(['System', 'Utterance'])], axis=1).reset_index()
            df = merged_df.loc[:,~merged_df.columns.str.lower().duplicated()].copy()
        except Exception as e:
            print(e)
    folders = dataset.split(os.path.sep)
    folders[0] = "final_df"
    final = os.path.join(*folders)
    path = os.path.dirname(final)

    if not os.path.exists(path):
        # Create the folder if it doesn't exist
        os.makedirs(path)
        print(f"Folder '{path}' created successfully!")
    df.to_csv(final,index=False)
    print(dataset, "DONE")
