import pandas as pd
#from metrics.metrics import Metrics

def add_metric(csv_path,metric,metric_name,replace = False):
    # csv_path : path to the csv to wich the function will add metric scores
    # metric : takes a string and a list of strings (hyp,refs) and output a score
    # metric_name : a string
    # replace : replace metric column if a column with the same name already exists
    df = pd.read_csv(csv_path)
    len_rows = df.shape[0]
    if not metric_name in df or replace : 
        new_col = []
        for index, row in df.iterrows():
            new_col.append(max([metric(row["hyp"],[row["ref"+str(i)]]) for i in range(1,6) if row["ref"+str(i)]==row["ref"+str(i)]]))
            print(index,len_rows)
        df[metric_name] = new_col
        df.to_csv(csv_path,index=False)
    return df[metric_name]
    
def remove_metrics(csv_path,metric_names):
    # csv_path : path to the csv to wich the function will remove metric scores
    # metric_names: a list of columns to remove
    df = pd.read_csv(csv_path)
    df = df.drop(metric_names,axis=1)
    df.to_csv(csv_path,index=False)
#m = Metrics()

datasets = ["WebNLG2020_rdf2text_en.csv","WebNLG2020_rdf2text_ru.csv"]

#add_metric(datasets[0],m.BLEU,"BLEU",True)
#remove_metrics(datasets[0],["BLEU"])