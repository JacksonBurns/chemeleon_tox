| model | balanced accuracy (BA) | Mathew’s correlation coefficient (MCC) | area under curve-receiver operating characteristic (AUC-ROC) | sensitivity | specificity | F1 score | likelihood ratio (LR+) | positive predictive value (PPV) | average precision score (AP) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| proxy-DILI data only                              | **0.63** | 0.24     | 0.67     | 0.60     | 0.66     | 0.72     | 1.76     | 0.79     | **0.81** |
| chemical structure only                           | 0.54     | 0.08     | 0.57     | 0.34     | 0.74     | 0.73     | 1.31     | 0.72     | 0.76     |
| chemical structure and physicochemical properties | 0.57     | 0.13     | 0.62     | 0.43     | 0.71     | 0.72     | 1.47     | 0.74     | 0.79     |
| DILIPredictor                                     | 0.59     | 0.16     | 0.63     | 0.61     | 0.56     | 0.65     | 1.40     | 0.77     | 0.79     |
| CheMeleon High Quality Direct Training            | 0.54     | 0.08     | 0.50     | 0.76     | 0.31     | 0.74     | 1.11     | 0.72     | 0.69     |
| CheMeleon Low Quality Direct Training             | 0.60     | 0.18     | 0.65     | 0.45     | **0.75** | 0.58     | **1.77** | **0.80** | 0.79     |
| CheMeleon Low + High Transfer Learning            | **0.63** | **0.29** | **0.68** | **0.86** | 0.40     | **0.81** | 1.44     | 0.77     | 0.80     |
