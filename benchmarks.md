## Benchmarks

Development and testing of the ECG Recognition Library was performed on [CPSC 2018](http://2018.icbeb.org/Challenge.html).

### Differential diagnosis of MI and BER in case of significant ST-elevation

Evaluation metrics: balanced accuracy.

| option | Accuracy  |
| --- | --- |
| original  | 73%  |
| tuned  | 82%  |

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
| STTC | 88.41%  | 90.85% |
| MI | 86.95%  | 90.42% |
| HYP | 91.02%  | 94.27% |
| CD | 89.31%  | 92.61% |
| ALL | 86.91%  | 81.71% |