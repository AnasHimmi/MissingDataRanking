# MissingDataRanking

## Hardware requirements:

All our experiements can be run on CPUs. To reproduce all the results of our paper we estimate the total runtime to 1k CPU hours. 
To reduce this requirement we provide notebooks already executed.

To regenerate the data, in particular calculating the metrics, we estimate the total runtime to 10k CPU hours. The final data is available in the folder **data_task** and **final_df**.


## Installation:

Start by installing our requirements:
```pip install -r requirements.txt```

## Usage:

### Command Line Interface (CLI):
We provide a command line interface (CLI). You can use it as follows on a cpu:

``` 
# TASK LEVEL INFORMATION
export PATH_TO_DF_TO_RANK=example_data_cli/xtreme_missing.csv
export MODE=task_level
export BEST_SCORE=highest

python ranking_cli.py --df_to_rank=$PATH_TO_DF_TO_RANK --mode=$MODE --best_score=$BEST_SCORE

# INSTANCE LEVEL INFORMATION
export PATH_TO_DF_TO_RANK=example_data_cli/DIALOGUE_pc_missing.csv
export MODE=instance_level
export BEST_SCORE=lowest
python ranking_cli.py --df_to_rank=$PATH_TO_DF_TO_RANK --mode=$MODE --best_score=$BEST_SCORE
```

## To reproduce the results of our paper:

### Task Level:
**task_level_calculate_correlations.py** : Contains functions to calculate the one level Borda ranking for a DataFrame with missing values in the task level and conduct the robustness analysis. \
The DataFrame should be in this form :

$$
M = \begin{bmatrix} \text{Model} & M_{1} & M_{2} & \cdots & M_{|M|} \ 
S_1 & & & & \ 
S_1 & & & & \ 
\cdots & & & & \ 
S_{|S|} & & & & \
\end{bmatrix}
$$ 

- **Example of usage** : For a one use case on a DataFrame with missing values (represented as `np.nan`):
```python
import pandas as pd
import task_level_calculate_correlations as tcorr
file = "example.csv"
df = pd.read_csv(file)
ranking, p_total, systems = tcorr.one_levels_incomplete_aggregation_task_level(df)
```
- For the robustness analysis on a DataFrame with no missing values :
```bash
for seed in {0..99}; do
python3 task_level_calculate_correlations.py --file=example.csv --seed=${seed};
done
```
The results for the robustness analysis can be plotted using **task_level_plot_correlations.ipynb**

### Instance Level:
- **task_level_calculate_correlations.py** : Contains functions to calculate the one level and two level Borda ranking for a DataFrame with missing values in the instance level and conduct the robustness analysis. \
The DataFrame should be in this form, and all systems should have the same utterances :

$$
M = \begin{bmatrix} \text{System} & \text{Utterance} & M_{1} & M_{2} & \cdots & M_{|M|} \ 
S_1 & u^{S_1}1 & & & & \ 
S_1 & \cdots & & & &\ 
S_1 & u^{S_1}{|S_1|} & & & & \ 
\cdots & & & & \ 
\cdots & & & &\ 
S_{|S|} & u^{S_{|S|}}1 & & & & \ 
S{|S|} & \cdots & & & & \ 
S_{|S|} & u^{S_{|S|}}_{|S|} & & & & \ 
\end{bmatrix}
$$

- **Example of usage** : For a one use case on a DataFrame with missing values (represented as `np.nan`):
```python
import pandas as pd
import instance_level_calculate_correlations as icorr
file = "example.csv"
df = icorr.load_file(file)
ranking_1, p_total_1, systems_1 = icorr.one_levels_incomplete_aggregation(df)
ranking_2, p_total_2, systems_2 = icorr.two_levels_incomplete_aggregation(df)
```
- For the robustness analysis on a DataFrame with no missing values :
```bash
for sample in {0..99}; do
python3 instance_level_calculate_correlations.py --file=example.csv --sample=${sample};
done
```
The results for the robustness analysis can be plotted using **task_level_plot_correlations.ipynb**

### Confidence intervals:
See the notebook **confidence_intervals.ipynb** for the code to calculate the confidence intervals.

### Synthetic experiment:
See the notebook **synthetic.ipynb** for the code to run the synthetic experiment.

## Our data:
- The raw data as well as the script to preprocess it is available in the folder **data_generation**.
- The final task level data is in **data_task**.
- The final instance level data is in **final_df**.
- **data_visualization.ipynb** contains the code to plot the size of the data.

## Results:
- The raw results of the robustness analysis are available in the folder **instance_level_correlations** and **task_level_correlations**.
- They can be plotted using the notebooks **instance_level_plot_correlations.ipynb** and **task_level_plot_correlations.ipynb**.
