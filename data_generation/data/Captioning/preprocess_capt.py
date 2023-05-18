import os
import tqdm
import pandas as pd
import numpy as np
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from pycocotools.coco import COCO
# from pycocoevalcap.eval import COCOEvalCap
sys_path = "data/Captioning/coco_captioning_challenge"
ref_path = "data/Captioning/captions_val2014.json"
sys_infos = []
for root, dirs, files in os.walk(sys_path):
    sys_name = root.split('/')[-1]
    if sys_name in ['human', 'mmitchell', 'MSR_Captivator', 'NearestNeighbor', 'random']:
        continue
    for f in files:
        if f.endswith("_results.json") and 'val2014' in f :   
            sys_infos.append((os.path.join(root, f), sys_name))

num_sys = len(sys_infos)

coco = COCO(ref_path)

num_instances = len(coco.getImgIds())
    
crossing_ids = coco.getImgIds()[:num_instances]
sorted(crossing_ids)
data = {"System":[],"Utterance":[],"ref1":[],"ref2":[],"ref3":[],"ref4":[],"ref5":[],"ref6":[],"ref7":[],"hyp":[]}

for i in tqdm.tqdm(range(len(sys_infos))):

    sys_path, sys_name = sys_infos[i]

    cocoRes = coco.loadRes(sys_path)
    # cocoEval = COCOEvalCap(coco, cocoRes, num_instances = num_instances)
    # cocoEval.evaluate(metric_eval)
    # imgIds = self.coco.getImgIds()
    gts = {}
    res = {}
    for imgId in coco.getImgIds()[:num_instances]:
        gts[imgId] = coco.imgToAnns[imgId]
        res[imgId] = cocoRes.imgToAnns[imgId]
    tokenizer = PTBTokenizer()
    gts  = tokenizer.tokenize(gts)
    res = tokenizer.tokenize(res)
        # print(coco.imgToAnns[imgId],  cocoRes.imgToAnns[imgId])
    for j, imgId in enumerate(crossing_ids):
        data["System"].append(sys_name)
        data["Utterance"].append(j)
        for c in range(0,7):
            if c<len(gts[imgId]):
                data["ref"+str(c+1)].append(gts[imgId][c])
            else:
                data["ref"+str(c+1)].append(np.nan)
        data["hyp"].append(res[imgId][0])

final_df = pd.DataFrame.from_dict(data).sort_values(by=['System'], kind = 'stable')
final_df.to_csv('csvs/Captioning/coco_data.csv',index=False)
