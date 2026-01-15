Running the model 
``python main.py``
Terminal Output:
[INFO] Using device: mps
[INFO] Loaded 5174 sequences.
[INFO] Train: 3621, Val: 776, Test: 777
[INFO] Model initialized
[Epoch 1] Loss: 1.3940 | D1 Acc: 0.970 | D2 Acc: 0.707 | Jersey Acc: 0.701
[Epoch 2] Loss: 0.8696 | D1 Acc: 0.981 | D2 Acc: 0.821 | Jersey Acc: 0.813
[Epoch 3] Loss: 0.6214 | D1 Acc: 0.988 | D2 Acc: 0.865 | Jersey Acc: 0.857
[Epoch 4] Loss: 0.4774 | D1 Acc: 0.969 | D2 Acc: 0.905 | Jersey Acc: 0.883
[Epoch 5] Loss: 0.4068 | D1 Acc: 0.976 | D2 Acc: 0.905 | Jersey Acc: 0.893
[Epoch 6] Loss: 0.3434 | D1 Acc: 0.991 | D2 Acc: 0.910 | Jersey Acc: 0.903
[Epoch 7] Loss: 0.2872 | D1 Acc: 0.992 | D2 Acc: 0.905 | Jersey Acc: 0.899
[Epoch 8] Loss: 0.2483 | D1 Acc: 0.987 | D2 Acc: 0.915 | Jersey Acc: 0.907
[Epoch 9] Loss: 0.2168 | D1 Acc: 0.997 | D2 Acc: 0.919 | Jersey Acc: 0.918
[Epoch 10] Loss: 0.1808 | D1 Acc: 0.992 | D2 Acc: 0.916 | Jersey Acc: 0.912
[INFO] Training curves saved to training_curves.png
Digit1 Acc: 0.992, Prec: 0.939, Rec: 0.973, F1: 0.952
Digit2 Acc: 0.916, Prec: 0.908, Rec: 0.903, F1: 0.904
Jersey Acc: 0.912
[SAMPLE PREDICTIONS]
GT: 8 → Pred: 8
GT: 8 → Pred: 8
GT: 8 → Pred: 8
GT: 6 → Pred: 6
GT: 9 → Pred: 9

[Test Set] Digit1 Acc: 0.987, Prec: 0.911, Rec: 0.984, F1: 0.944
[Test Set] Digit2 Acc: 0.928, Prec: 0.920, Rec: 0.919, F1: 0.919
[Test Set] Jersey Acc: 0.921