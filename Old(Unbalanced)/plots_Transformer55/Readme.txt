/Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/site-packages/torch/nn/modules/transformer.py:306: UserWarning: enable_nested_tensor is True, but self.use_nested_tensor is False because encoder_layer.norm_first was True
  warnings.warn(f"enable_nested_tensor is True, but self.use_nested_tensor is False because {why_not_sparsity_fast_path}")
Class weights: [1.2034198e-01 3.2317400e-01 4.1595623e-01 4.5725685e-01 4.3125895e-01
 6.2747651e-01 7.2037113e-01 8.6832613e-01 9.6589082e-01 1.3865207e+00
 2.6586893e+00 2.1138759e+00 1.9103174e+00 2.2822378e+00 2.3878968e+00
 2.9987543e+00 4.6889610e+00 3.6841836e+00 4.9594779e+00 7.8149352e+00
 8.8928576e+00 1.1722403e+01 1.2894643e+01 1.4327381e+01 2.5789286e+01
 5.1578571e+01 1.2894643e+02 1.2894643e+02 0.0000000e+00]
Classes in dataset: [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23
 24 25 26 27]
Missing classes in training set: {28}
===============================================================================================
Layer (type:depth-idx)                        Output Shape              Param #
===============================================================================================
MotionTransformer                             [1, 29]                   131,072
├─Linear: 1-1                                 [64, 1000, 256]           1,024
├─TransformerEncoder: 1-2                     [1, 64, 256]              --
│    └─ModuleList: 2-1                        --                        --
│    │    └─TransformerEncoderLayer: 3-1      [1, 64, 256]              789,760
│    │    └─TransformerEncoderLayer: 3-2      [1, 64, 256]              789,760
│    │    └─TransformerEncoderLayer: 3-3      [1, 64, 256]              789,760
│    │    └─TransformerEncoderLayer: 3-4      [1, 64, 256]              789,760
├─Sequential: 1-3                             [1, 29]                   --
│    └─LayerNorm: 2-2                         [1, 256]                  512
│    └─Dropout: 2-3                           [1, 256]                  --
│    └─Linear: 2-4                            [1, 256]                  65,792
│    └─ReLU: 2-5                              [1, 256]                  --
│    └─Dropout: 2-6                           [1, 256]                  --
│    └─Linear: 2-7                            [1, 29]                   7,453
===============================================================================================
Total params: 3,364,893
Trainable params: 3,364,893
Non-trainable params: 0
Total mult-adds (Units.MEGABYTES): 0.14
===============================================================================================
Input size (MB): 0.77
Forward/backward pass size (MB): 131.08
Params size (MB): 0.30
Estimated Total Size (MB): 132.14
===============================================================================================
Epoch [1/50] Loss: 2858.8688  Accuracy: 16.95%
Epoch [2/50] Loss: 2355.1254  Accuracy: 31.60%
Epoch [3/50] Loss: 2226.5488  Accuracy: 35.67%
Epoch [4/50] Loss: 2204.7646  Accuracy: 37.32%
Epoch [5/50] Loss: 2178.0436  Accuracy: 37.40%
Epoch [6/50] Loss: 2074.8861  Accuracy: 36.59%
Epoch [7/50] Loss: 1997.4302  Accuracy: 37.71%
Epoch [8/50] Loss: 1967.7766  Accuracy: 40.17%
Epoch [9/50] Loss: 1932.0844  Accuracy: 41.34%
Epoch [10/50] Loss: 1878.3494  Accuracy: 44.11%
Epoch [11/50] Loss: 1870.4950  Accuracy: 41.66%
Epoch [12/50] Loss: 1828.3665  Accuracy: 44.08%
Epoch [13/50] Loss: 1816.2340  Accuracy: 43.05%
Epoch [14/50] Loss: 1754.7984  Accuracy: 45.38%
Epoch [15/50] Loss: 1761.3763  Accuracy: 45.92%
Epoch [16/50] Loss: 1705.7356  Accuracy: 44.32%
Epoch [17/50] Loss: 1682.4872  Accuracy: 46.02%
Epoch [18/50] Loss: 1653.3833  Accuracy: 46.10%
Epoch [19/50] Loss: 1621.2747  Accuracy: 47.92%
Epoch [20/50] Loss: 1598.9136  Accuracy: 47.08%
Epoch [21/50] Loss: 1565.0638  Accuracy: 47.71%
Epoch [22/50] Loss: 1543.4881  Accuracy: 48.47%
Epoch [23/50] Loss: 1537.1952  Accuracy: 48.90%
Epoch [24/50] Loss: 1514.0940  Accuracy: 48.86%
Epoch [25/50] Loss: 1490.5963  Accuracy: 49.36%
Epoch [26/50] Loss: 1484.8000  Accuracy: 49.41%
Epoch [27/50] Loss: 1470.1403  Accuracy: 49.11%
Epoch [28/50] Loss: 1445.8110  Accuracy: 51.09%
Epoch [29/50] Loss: 1394.3731  Accuracy: 52.79%
Epoch [30/50] Loss: 1438.7290  Accuracy: 51.42%
Epoch [31/50] Loss: 1422.1801  Accuracy: 51.10%
Epoch [32/50] Loss: 1394.1096  Accuracy: 51.88%
Epoch [33/50] Loss: 1360.7758  Accuracy: 52.46%
Epoch [34/50] Loss: 1337.2525  Accuracy: 53.18%
Epoch [35/50] Loss: 1379.1119  Accuracy: 52.44%
Epoch [36/50] Loss: 1371.4866  Accuracy: 52.32%
Epoch [37/50] Loss: 1341.4286  Accuracy: 52.67%
Epoch [38/50] Loss: 1323.7929  Accuracy: 53.87%
Epoch [39/50] Loss: 1321.0781  Accuracy: 52.68%
Epoch [40/50] Loss: 1302.2060  Accuracy: 54.15%
Epoch [41/50] Loss: 1303.3230  Accuracy: 53.79%
Epoch [42/50] Loss: 1293.6649  Accuracy: 54.08%
Epoch [43/50] Loss: 1250.4082  Accuracy: 55.02%
Epoch [44/50] Loss: 1270.3682  Accuracy: 54.33%
Epoch [45/50] Loss: 1273.6936  Accuracy: 54.69%
Epoch [46/50] Loss: 1260.8279  Accuracy: 55.37%
Epoch [47/50] Loss: 1240.6341  Accuracy: 55.34%
Epoch [48/50] Loss: 1248.3849  Accuracy: 54.49%
Epoch [49/50] Loss: 1201.1577  Accuracy: 55.34%
Epoch [50/50] Loss: 1250.6916  Accuracy: 55.39%
Training complete. Metrics, plots, and confusion matrix saved in 'plots/' folder.