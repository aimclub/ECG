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
| STTC | 89.75%  | 92.44% |
| MI | 88.74%  | 91.55% |
| HYP | 87.39%  | 91.6% |
| CD | 91.69%  | 94.04% |
| ALL | 89.04%  | 84.44% |