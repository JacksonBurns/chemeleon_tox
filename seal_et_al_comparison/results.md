CheMeleon foundation model toxicity prediction performance compared to other models as presented in [_Improved Detection of Drug-Induced Liver Injury by Integrating Predicted In Vivo and In Vitro Data_](https://doi.org/10.1021/acs.chemrestox.4c00015) by Seal et al.
Each row in the table gives a model's performance on various binary classification metrics when predicting on the 223 witheld compounds from the referenced study.
CheMeleon results include training on a small high-quality dataset ("High Quality Direct Training"), training a large low-quality dataset ("Low Quality Direct Training"), and pretraining on the latter then transfer learning to the former ("Low + High Transfer Learning").
All datasets were retrieved from the reference study - see `data.ipynb` for further details.

| Model | BA | MCC | AUC-ROC | Sensitivity | Specificity | F1 | LR+ | PPV | AP |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| proxy-DILI data only                              | **0.63** | 0.24     | 0.67     | 0.60     | 0.66     | 0.72     | 1.76     | 0.79     | **0.81** |
| chemical structure only                           | 0.54     | 0.08     | 0.57     | 0.34     | 0.74     | 0.73     | 1.31     | 0.72     | 0.76     |
| chemical structure and physicochemical properties | 0.57     | 0.13     | 0.62     | 0.43     | 0.71     | 0.72     | 1.47     | 0.74     | 0.79     |
| DILIPredictor                                     | 0.59     | 0.16     | 0.63     | 0.61     | 0.56     | 0.65     | 1.40     | 0.77     | 0.79     |
| CheMeleon High Quality Direct Training            | 0.54     | 0.08     | 0.50     | 0.76     | 0.31     | 0.74     | 1.11     | 0.72     | 0.69     |
| CheMeleon Low Quality Direct Training             | 0.60     | 0.18     | 0.65     | 0.45     | **0.75** | 0.58     | **1.77** | **0.80** | 0.79     |
| CheMeleon Low + High Transfer Learning            | **0.63** | **0.29** | **0.68** | **0.86** | 0.40     | **0.81** | 1.44     | 0.77     | 0.80     |

CheMeleon outperforms all other models on all but one of the tested metrics, and then only narrowly.
The most effective strategy is the two stage "Low + High Transfer Learning" in which CheMeleon is first pretrained on a large dataset of toxicity results scraped from the literature.
The resulting model is then re-used on a smaller, carefully curated dataset to improve its performance.
Perhaps the most telling metric is the Sensitivity of CheMeleon's best model, 0.86, a substantial improvement over the next best model (DILIPredictor's 0.61) and a good indicator of overall classifier performance.

Acronym Key:
 - BA: Balanced Accuracy
 - MCC: Matthew's Correlation Coefficient
 - AUC-ROC: Area Under Curve-Reciever Operating Characteristic
 - LR+: Positive Likelihood Ratio
 - PPV: Positive Predictive Value
 - AP: Average Precision
