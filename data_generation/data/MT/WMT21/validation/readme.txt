Requires:
python>=3.6
pandas, numpy 

This program checks for NaNs in your scores, and compares your scores with the demo scores (that were generated randomly).  

The script generates a warning for missing/extra language pairs or testsets or systems or segments. Note that you are welcome to choose which language-pairs you submit scores for, so you can ignore these warnings.
 
If your metric takes part in a language pair, please score all the testsets available. 


The script expects METRICNAME.sys.score(.gz) and METRICNAME.seg.score(.gz) 
(that is, it supports both the zipped and unzipped file format) in this folder.


if your metric is source based, you'll need to download srcmetric.*.score, and run 
>>    python validation.py -n METRICNAME --src

if your metric is reference based, you'll need to download refmetric.*.score, and run 
>>    python validation.py -n METRICNAME 




usage: validation.py [-h] [-n METRICNAME] [--src]

optional arguments:
  -h, --help            show this help message and exit
  -n METRICNAME, --metricname METRICNAME
  --src                 if your metric is source-based 

