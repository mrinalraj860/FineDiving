Training: 
Dataset size: 7021
Class weights: [1.1974689e-01 2.9327485e-01 4.1514900e-01 4.3382353e-01 4.7490531e-01
 6.2844610e-01 6.8510932e-01 8.0368590e-01 1.1194197e+00 1.7413194e+00
 1.8170290e+00 2.2388394e+00 2.5850515e+00 2.5074999e+00 2.7554946e+00
 2.9851191e+00 3.5316901e+00 3.7425373e+00 6.1158538e+00 1.0447917e+01
 9.2870369e+00 1.0447917e+01 1.4750000e+01 2.5075001e+01 3.1343750e+01
 8.3583336e+01 1.2537500e+02 0.0000000e+00 2.5075000e+02]
Classes in dataset: [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23
 24 25 26 28]
Missing classes in training set: {27}
==========================================================================================
Layer (type:depth-idx)                   Output Shape              Param #
==========================================================================================
MotionCNN                                [1, 29]                   --
├─Conv2d: 1-1                            [1, 32, 64, 1000]         608
├─Conv2d: 1-2                            [1, 64, 64, 1000]         18,496
├─AdaptiveAvgPool2d: 1-3                 [1, 64, 16, 16]           --
├─Linear: 1-4                            [1, 128]                  2,097,280
├─Linear: 1-5                            [1, 29]                   3,741
==========================================================================================
Total params: 2,120,125
Trainable params: 2,120,125
Non-trainable params: 0
Total mult-adds (Units.GIGABYTES): 1.22
==========================================================================================
Input size (MB): 0.51
Forward/backward pass size (MB): 49.15
Params size (MB): 8.48
Estimated Total Size (MB): 58.15
==========================================================================================
Epoch [1/50] Loss: 2165.3848  Accuracy: 39.25%
Epoch [2/50] Loss: 1781.0743  Accuracy: 46.96%
Epoch [3/50] Loss: 1607.9074  Accuracy: 50.33%
Epoch [4/50] Loss: 1512.6075  Accuracy: 52.51%
Epoch [5/50] Loss: 1422.1040  Accuracy: 53.37%
Epoch [6/50] Loss: 1375.3934  Accuracy: 54.38%
Epoch [7/50] Loss: 1325.6407  Accuracy: 54.56%
Epoch [8/50] Loss: 1266.6307  Accuracy: 55.41%
Epoch [9/50] Loss: 1268.1822  Accuracy: 56.36%
Epoch [10/50] Loss: 1188.7464  Accuracy: 59.36%
Epoch [11/50] Loss: 1168.1310  Accuracy: 59.98%
Epoch [12/50] Loss: 1108.0255  Accuracy: 61.36%
Epoch [13/50] Loss: 1116.8767  Accuracy: 62.10%
Epoch [14/50] Loss: 1066.2456  Accuracy: 61.17%
Epoch [15/50] Loss: 1076.7875  Accuracy: 63.01%
Epoch [16/50] Loss: 1060.7247  Accuracy: 63.15%
Epoch [17/50] Loss: 995.3321  Accuracy: 63.89%
Epoch [18/50] Loss: 984.7955  Accuracy: 65.23%
Epoch [19/50] Loss: 987.5584  Accuracy: 65.02%
Epoch [20/50] Loss: 965.7743  Accuracy: 65.87%
Epoch [21/50] Loss: 937.5673  Accuracy: 65.80%
Epoch [22/50] Loss: 926.0311  Accuracy: 66.50%
Epoch [23/50] Loss: 887.7828  Accuracy: 67.58%
Epoch [24/50] Loss: 852.6531  Accuracy: 68.86%
Epoch [25/50] Loss: 840.1299  Accuracy: 68.41%
Epoch [26/50] Loss: 818.2093  Accuracy: 68.84%
Epoch [27/50] Loss: 805.9700  Accuracy: 69.63%
Epoch [28/50] Loss: 805.9848  Accuracy: 69.08%
Epoch [29/50] Loss: 771.0218  Accuracy: 71.49%
Epoch [30/50] Loss: 759.9509  Accuracy: 71.21%
Epoch [31/50] Loss: 719.3673  Accuracy: 71.67%
Epoch [32/50] Loss: 764.4397  Accuracy: 72.38%
Epoch [33/50] Loss: 718.8675  Accuracy: 71.93%
Epoch [34/50] Loss: 693.0844  Accuracy: 72.87%
Epoch [35/50] Loss: 680.6142  Accuracy: 73.29%
Epoch [36/50] Loss: 651.9855  Accuracy: 74.33%
Epoch [37/50] Loss: 645.2126  Accuracy: 74.19%
Epoch [38/50] Loss: 646.1827  Accuracy: 75.00%
Epoch [39/50] Loss: 624.5809  Accuracy: 75.84%
Epoch [40/50] Loss: 597.2743  Accuracy: 76.21%
Epoch [41/50] Loss: 591.5538  Accuracy: 76.11%
Epoch [42/50] Loss: 604.8243  Accuracy: 76.20%
Epoch [43/50] Loss: 571.0692  Accuracy: 76.91%
Epoch [44/50] Loss: 542.5580  Accuracy: 77.78%
Epoch [45/50] Loss: 550.6108  Accuracy: 77.38%
Epoch [46/50] Loss: 521.7227  Accuracy: 78.01%
Epoch [47/50] Loss: 507.2530  Accuracy: 78.86%
Epoch [48/50] Loss: 519.1220  Accuracy: 78.75%
Epoch [49/50] Loss: 528.0503  Accuracy: 78.71%
Epoch [50/50] Loss: 484.2041  Accuracy: 79.66%
Training complete. Metrics, plots, and confusion matrix saved in 'plots/' folder.