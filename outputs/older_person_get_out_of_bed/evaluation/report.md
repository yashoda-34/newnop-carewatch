# Evaluation

- Accuracy: **95.2%**
- Macro-F1: **0.93**
- Mean duration error: **0.5 s**

## Confusions

- SITTING_ON_BED predicted as LYING_IN_BED: 1 s

## Bed events

- bed_exit: TP 0, FP 0, FN 0, precision None, recall None
- bed_return: TP 0, FP 0, FN 0, precision None, recall None

## Duration

| State | Ground truth | Predicted | Error |
|---|---|---|---|
| LYING_IN_BED | 16.5 s | 17.0 s | 0.5 s |
| SITTING_ON_BED | 4.5 s | 4.0 s | 0.5 s |
