## Benchmarks
    
All methods except for `ecg_is_normal` were trained on [CPSC 2018](http://2018.icbeb.org/Challenge.html). For benchmarking [PTB-XL](https://physionet.org/content/ptb-xl/1.0.1/) dataset was used.

Method `ecg_is_normal` was trained, validated and tested on [PTB-XL](https://physionet.org/content/ptb-xl/1.0.1/) dataset. Folds 1-8 were used as the training set, fold 9 as the validation set and fold 10 as the test set of the entire dataset, respectively.

### Detect significant ST-elevation
We perform evaluation using 2 metrics:
1. failure rate (% of cases method failed to process)
2. F1-score on cases method succeeded to process

| method | F1-score  | Failure rate |
| --- | --- | --- |
| `check_ST_elevation`  | 28.63%  | < 0.01%  |
| `check_ST_elevation_with_NN`  | 13.58%  | 0%  |

### Differential diagnosis of MI and BER in case of significant ST-elevation

#### Differential diagnosis between MI and BER
Performed with `diagnose_with_risk_markers` with either original formula or tuned.

Balanced accuracy is used for evaluation.

| option | Accuracy  |
| --- | --- |
| original  | 51.43%  |
| tuned  | 58.29%  |

#### Differential diagnosis between MI, BER, and other
Performed with `check_BER_with_NN` and `check_MI_with_NN`.

F1-score is used to evaluate each.

| diagnosis | F1-score  |
| --- | --- |
| BER  | 33.79%  |
| MI  | 56.55%  |

### Classification between Normal and Abnormal classes

Performed with `ecg_is_normal`.

The model supports specifying the abnormality class for which subset data will be selected:
 - STTC (ST/T Change)
 - MI (Myocardial Infarction)
 - HYP (Hypertrophy)
 - CD (Conduction Disturbance)
 - ALL (All available classes)

Accuracy and F1-score are used to evaluate each scenario.

| Abnormality class | Accuracy | F1-score |
| --- | --- | --- |
| STTC | 89.75%  | 92.44% |
| MI | 88.74%  | 91.55% |
| HYP | 87.39%  | 91.6% |
| CD | 91.69%  | 94.04% |
| ALL | 89.04%  | 84.44% |