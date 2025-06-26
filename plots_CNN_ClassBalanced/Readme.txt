Class weights: [1.2034198e-01 3.2317400e-01 4.1595623e-01 4.5725685e-01 4.3125895e-01
 6.2747651e-01 7.2037113e-01 8.6832613e-01 9.6589082e-01 1.3865207e+00
 2.6586893e+00 2.1138759e+00 1.9103174e+00 2.2822378e+00 2.3878968e+00
 2.9987543e+00 4.6889610e+00 3.6841836e+00 4.9594779e+00 7.8149352e+00
 8.8928576e+00 1.1722403e+01 1.2894643e+01 1.4327381e+01 2.5789286e+01
 5.1578571e+01 1.2894643e+02 1.2894643e+02 0.0000000e+00]
Classes in dataset: [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23
 24 25 26 27]
Missing classes in training set: {28}
==========================================================================================
Layer (type:depth-idx)                   Output Shape              Param #
==========================================================================================
MotionCNN                                [1, 29]                   --
├─Conv2d: 1-1                            [1, 32, 64, 1000]         896
├─Conv2d: 1-2                            [1, 64, 64, 1000]         18,496
├─AdaptiveAvgPool2d: 1-3                 [1, 64, 16, 16]           --
├─Linear: 1-4                            [1, 128]                  2,097,280
├─Linear: 1-5                            [1, 29]                   3,741
==========================================================================================
Total params: 2,120,413
Trainable params: 2,120,413
Non-trainable params: 0
Total mult-adds (Units.GIGABYTES): 1.24
==========================================================================================
Input size (MB): 0.77
Forward/backward pass size (MB): 49.15
Params size (MB): 8.48
Estimated Total Size (MB): 58.40
==========================================================================================
Epoch [1/50] Loss: 2037.3469  Accuracy: 45.04%
Epoch [2/50] Loss: 1585.0336  Accuracy: 52.32%
Epoch [3/50] Loss: 1440.2437  Accuracy: 55.56%
Epoch [4/50] Loss: 1335.5060  Accuracy: 57.98%
Epoch [5/50] Loss: 1277.9243  Accuracy: 59.74%
Epoch [6/50] Loss: 1198.1106  Accuracy: 59.41%
Epoch [7/50] Loss: 1171.9382  Accuracy: 60.19%
Epoch [8/50] Loss: 1101.8979  Accuracy: 63.76%
Epoch [9/50] Loss: 1055.5467  Accuracy: 64.62%
Epoch [10/50] Loss: 977.3884  Accuracy: 66.53%
Epoch [11/50] Loss: 959.6579  Accuracy: 66.22%
Epoch [12/50] Loss: 945.0153  Accuracy: 65.84%
Epoch [13/50] Loss: 911.4822  Accuracy: 68.11%
Epoch [14/50] Loss: 883.8301  Accuracy: 67.53%
Epoch [15/50] Loss: 830.5474  Accuracy: 70.09%
Epoch [16/50] Loss: 821.0345  Accuracy: 69.82%
Epoch [17/50] Loss: 785.1775  Accuracy: 72.15%
Epoch [18/50] Loss: 772.0635  Accuracy: 71.14%
Epoch [19/50] Loss: 723.0816  Accuracy: 73.31%
Epoch [20/50] Loss: 712.8036  Accuracy: 72.10%
Epoch [21/50] Loss: 664.2456  Accuracy: 75.06%
Epoch [22/50] Loss: 669.9789  Accuracy: 74.17%
Epoch [23/50] Loss: 631.5787  Accuracy: 75.56%
Epoch [24/50] Loss: 583.0491  Accuracy: 76.01%
Epoch [25/50] Loss: 566.0048  Accuracy: 77.04%
Epoch [26/50] Loss: 569.5202  Accuracy: 76.62%
Epoch [27/50] Loss: 561.3173  Accuracy: 77.95%
Epoch [28/50] Loss: 518.4961  Accuracy: 79.16%
Epoch [29/50] Loss: 496.0261  Accuracy: 79.30%
Epoch [30/50] Loss: 471.2664  Accuracy: 81.19%
Epoch [31/50] Loss: 448.8584  Accuracy: 82.01%
Epoch [32/50] Loss: 451.6861  Accuracy: 81.06%
Epoch [33/50] Loss: 467.3366  Accuracy: 81.64%
Epoch [34/50] Loss: 390.5398  Accuracy: 83.77%
Epoch [35/50] Loss: 398.3038  Accuracy: 83.40%
Epoch [36/50] Loss: 403.2514  Accuracy: 82.83%
Epoch [37/50] Loss: 366.6224  Accuracy: 84.19%
Epoch [38/50] Loss: 354.5930  Accuracy: 85.60%
Epoch [39/50] Loss: 342.1743  Accuracy: 85.69%
Epoch [40/50] Loss: 343.1853  Accuracy: 85.96%
Epoch [41/50] Loss: 324.6777  Accuracy: 86.25%
Epoch [42/50] Loss: 323.1491  Accuracy: 85.81%
Epoch [43/50] Loss: 294.4246  Accuracy: 87.08%
Epoch [44/50] Loss: 298.5284  Accuracy: 87.00%
Epoch [45/50] Loss: 301.8439  Accuracy: 87.94%
Epoch [46/50] Loss: 251.2170  Accuracy: 88.51%
Epoch [47/50] Loss: 272.9230  Accuracy: 88.09%
Epoch [48/50] Loss: 267.0653  Accuracy: 88.98%
Epoch [49/50] Loss: 249.0196  Accuracy: 90.31%
Epoch [50/50] Loss: 301.7428  Accuracy: 87.83%
Training complete. Metrics, plots, and confusion matrix saved in 'plots/' folder.