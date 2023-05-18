# code_submission
## Task Level
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

Usage : 
- For a one use case on a DataFrame with missing values (represented as `np.nan`):
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
The results for the robustness analysis can be plotted using **task_level_plot_correlations.ipynb** \

## Instance Level
**task_level_calculate_correlations.py** : Contains functions to calculate the one level and two level Borda ranking for a DataFrame with missing values in the instance level and conduct the robustness analysis. \
The DataFrame should be in this form :

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

Usage : 
- For a one use case on a DataFrame with missing values (represented as `np.nan`):
```python
import pandas as pd
import instance_level_calculate_correlations as icorr
file = "example.csv"
df = pd.read_csv(file)
ranking_1, p_total_1, systems_1 = icorr.one_levels_incomplete_aggregation(df)
ranking_2, p_total_2, systems_2 = icorr.two_levels_incomplete_aggregation(df)
```
- For the robustness analysis on a DataFrame with no missing values :
```bash
for seed in {0..99}; do
python3 task_level_calculate_correlations.py --file=example.csv --seed=${seed};
done
```
The results for the robustness analysis can be plotted using **task_level_plot_correlations.ipynb** \