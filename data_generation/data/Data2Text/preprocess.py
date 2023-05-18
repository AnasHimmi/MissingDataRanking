import json
import pandas as pd

data = json.load(open("data2text.webnlg"))

final_data = {}
cols = ["System","Utterance","H:fluency","H:grammar","H:semantics"]
for metric in data["0"]["ref_1"]:
    cols.append(metric) # adding metric scores

for i in data["12"]:
    print(i)

# systems = ["ref_1","ref_3"]

# for col in cols:
#     final_data[col] = []

# for i in data:
#     for s in range(len(systems)):
#         final_data["System"].append(s)
#         final_data["Utterance"].append("M"+i)
#         final_data["H:fluency"].append(data[i]["fluency"])
#         final_data["H:grammar"].append(data[i]["grammar"])
#         final_data["H:semantics"].append(data[i]["semantics"])
#         for metric in data[i][systems[s]]:
#             # print(i)
#             # print(metric)
#             final_data[metric].append(data[i][systems[s]][metric])

# final_df = pd.DataFrame.from_dict(final_data).sort_values(by=['System'], kind = 'stable')

# print(final_df.info())
# print(final_df)


