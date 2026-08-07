# Supervisor Quick Results Pack

Generated from completed runs on Deakin host `luthin` (RTX 4000 Ada).

## Tufts — missing teeth (100 cases, patient-mean over 32 Universal teeth)

### Zero-shot

```
Model               | F1    | Precision | Recall | Specificity | Accuracy | Mean_pred_per_case | Mean_GT_per_case
--------------------+-------+-----------+--------+-------------+----------+--------------------+-----------------
LLaVA-1.5-7B        | 0.042 | 0.127     | 0.039  | 0.873       | 0.722    | 3.080              | 6.630           
LLaVA-Med-v1.5-7B   | 0.052 | 0.082     | 0.047  | 0.921       | 0.772    | 1.800              | 6.630           
HuatuoGPT-Vision-7B | 0.188 | 0.216     | 0.228  | 0.706       | 0.646    | 8.830              | 6.630           
DentVLM             | 0.018 | 0.018     | 0.026  | 0.942       | 0.782    | 1.000              | 6.630           
```

### Few-shot

```
Model               | F1    | Precision | Recall | Specificity | Accuracy | Mean_pred_per_case | Mean_GT_per_case
--------------------+-------+-----------+--------+-------------+----------+--------------------+-----------------
LLaVA-1.5-7B        | 0.185 | 0.264     | 0.174  | 0.793       | 0.701    | 5.980              | 6.630           
LLaVA-Med-v1.5-7B   | 0.073 | 0.103     | 0.074  | 0.910       | 0.769    | 2.350              | 6.630           
HuatuoGPT-Vision-7B | 0.199 | 0.200     | 0.288  | 0.658       | 0.618    | 10.510             | 6.630           
DentVLM             | 0.000 | 0.000     | 0.000  | 0.970       | 0.793    | 0.000              | 6.630           
```

### Chain-of-thought (partial — 2/4 models done)

```
Model             | F1    | Precision | Recall | Specificity | Accuracy | Mean_pred_per_case | Mean_GT_per_case
------------------+-------+-----------+--------+-------------+----------+--------------------+-----------------
LLaVA-1.5-7B      | 0.070 | 0.204     | 0.060  | 0.900       | 0.746    | 2.320              | 6.630           
LLaVA-Med-v1.5-7B | 0.009 | 0.012     | 0.007  | 0.960       | 0.786    | 0.290              | 6.630           
```

HuatuoGPT-Vision and DentVLM CoT are still running.


## DENTEX — abnormal tooth + disease (val 50, patient-mean over 32 FDI teeth, zero-shot)

```
Model               | F1    | Precision | Recall | Specificity | Accuracy | Mean_pred_per_case | Mean_GT_per_case | Sec_per_case
--------------------+-------+-----------+--------+-------------+----------+--------------------+------------------+-------------
LLaVA-1.5-7B        | 0.076 | 0.110     | 0.070  | 0.945       | 0.848    | 1.800              | 3.580            | 13.200      
LLaVA-Med-v1.5-7B   | 0.000 | 0.000     | 0.000  | 1.000       | 0.888    | 0.000              | 3.580            | 1.210       
HuatuoGPT-Vision-7B | 0.052 | 0.067     | 0.051  | 0.949       | 0.849    | 1.600              | 3.580            | 32.906      
DentVLM             | 0.000 | 0.000     | 0.000  | 1.000       | 0.888    | 0.000              | 3.580            | 104.794     
```

### Per-disease F1

```
Model               | Caries | Deep Caries | Impacted | Periapical Lesion
--------------------+--------+-------------+----------+------------------
DentVLM             | 0.000  | 0.000       | 0.000    | 0.000            
HuatuoGPT-Vision-7B | 0.010  | 0.000       | 0.000    | 0.000            
LLaVA-1.5-7B        | 0.010  | 0.008       | 0.000    | 0.020            
LLaVA-Med-v1.5-7B   | 0.000  | 0.000       | 0.000    | 0.000            
```

## Takeaways for discussion

- On Tufts, HuatuoGPT-Vision leads zero-shot/few-shot F1 (~0.19–0.20); few-shot helps LLaVA a lot (0.04→0.18).

- DentVLM few-shot F1=0 (empty/no valid predictions); CoT so far weak for LLaVA-Med.

- On DENTEX, absolute abnormal F1 is low; LLaVA-Med and DentVLM score F1=0 with Spec=1.0 (no predicted abnormalities).

- High Accuracy/Specificity can be misleading under class imbalance; focus on F1/Recall.


Open `index.html` in a browser for all charts on one page.
