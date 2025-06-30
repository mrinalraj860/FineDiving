==========================================================================================
Epoch 01 | MSE 15.5668 | MAE 1.5425 | RMSE 1.8046
Epoch 02 | MSE 5.7787 | MAE 0.9656 | RMSE 1.1666
Epoch 03 | MSE 4.0104 | MAE 0.8779 | RMSE 1.0709
Epoch 04 | MSE 2.2536 | MAE 0.7505 | RMSE 0.9356
Epoch 05 | MSE 1.6819 | MAE 0.7061 | RMSE 0.8977
Epoch 06 | MSE 1.8597 | MAE 0.7230 | RMSE 0.9143
Epoch 07 | MSE 1.8359 | MAE 0.6814 | RMSE 0.8800
Epoch 08 | MSE 1.2774 | MAE 0.6230 | RMSE 0.8204
Epoch 09 | MSE 1.3224 | MAE 0.6479 | RMSE 0.8389
Epoch 10 | MSE 1.0847 | MAE 0.5880 | RMSE 0.7654
Epoch 11 | MSE 0.8721 | MAE 0.5517 | RMSE 0.7295
Epoch 12 | MSE 0.9654 | MAE 0.5689 | RMSE 0.7470
Epoch 13 | MSE 1.0374 | MAE 0.5444 | RMSE 0.7276
Epoch 14 | MSE 0.9307 | MAE 0.5278 | RMSE 0.7067
Epoch 15 | MSE 0.7452 | MAE 0.5074 | RMSE 0.6958
Epoch 16 | MSE 0.6544 | MAE 0.4775 | RMSE 0.6527
Epoch 17 | MSE 0.6732 | MAE 0.4718 | RMSE 0.6489
Epoch 18 | MSE 0.6858 | MAE 0.4569 | RMSE 0.6210
Epoch 19 | MSE 0.5890 | MAE 0.4281 | RMSE 0.6005
Epoch 20 | MSE 0.5518 | MAE 0.4314 | RMSE 0.5884
Epoch 21 | MSE 0.4188 | MAE 0.3863 | RMSE 0.5431
Epoch 22 | MSE 0.5812 | MAE 0.4186 | RMSE 0.5750
Epoch 23 | MSE 0.4871 | MAE 0.4133 | RMSE 0.5736
Epoch 24 | MSE 0.4735 | MAE 0.4158 | RMSE 0.5802
Epoch 25 | MSE 0.4299 | MAE 0.3902 | RMSE 0.5576
Epoch 26 | MSE 0.4191 | MAE 0.3753 | RMSE 0.5382
Epoch 27 | MSE 0.3053 | MAE 0.3403 | RMSE 0.5066
Epoch 28 | MSE 0.3338 | MAE 0.3506 | RMSE 0.5077
Epoch 29 | MSE 0.3080 | MAE 0.3536 | RMSE 0.5090
Epoch 30 | MSE 0.2837 | MAE 0.3330 | RMSE 0.4841
Epoch 31 | MSE 0.2213 | MAE 0.3193 | RMSE 0.4818
Epoch 32 | MSE 0.2461 | MAE 0.3085 | RMSE 0.4529
Epoch 33 | MSE 0.2878 | MAE 0.3262 | RMSE 0.4744
Epoch 34 | MSE 0.1928 | MAE 0.2973 | RMSE 0.4515
Epoch 35 | MSE 0.2098 | MAE 0.2887 | RMSE 0.4488
Epoch 36 | MSE 0.1787 | MAE 0.2855 | RMSE 0.4347
Epoch 37 | MSE 0.1764 | MAE 0.2742 | RMSE 0.4222
Epoch 38 | MSE 0.2018 | MAE 0.2887 | RMSE 0.4416
Epoch 39 | MSE 0.1601 | MAE 0.2536 | RMSE 0.4007
Epoch 40 | MSE 0.1584 | MAE 0.2669 | RMSE 0.4163
Epoch 41 | MSE 0.1880 | MAE 0.2718 | RMSE 0.4260
Epoch 42 | MSE 0.1655 | MAE 0.2631 | RMSE 0.4097
Epoch 43 | MSE 0.1674 | MAE 0.2591 | RMSE 0.4096
Epoch 44 | MSE 0.1616 | MAE 0.2691 | RMSE 0.4139
Epoch 45 | MSE 0.1321 | MAE 0.2475 | RMSE 0.3967
Epoch 46 | MSE 0.1737 | MAE 0.2701 | RMSE 0.4201
Epoch 47 | MSE 0.1712 | MAE 0.2626 | RMSE 0.4117
Epoch 48 | MSE 0.1447 | MAE 0.2514 | RMSE 0.4041
Epoch 49 | MSE 0.1331 | MAE 0.2429 | RMSE 0.3915
Epoch 50 | MSE 0.1392 | MAE 0.2307 | RMSE 0.3671

✅ Training complete — weighted model saved to difficulty_regressor.pt
> python -u "/Users/mrinalraj/Documents/FineDiving/TestForDifficulty.py"
✅ Loaded 2803 video samples with valid difficulty labels.
🧪 Test set size: 701

📊 Difficulty Regression Test Metrics:
MSE   : 0.7037
MAE   : 0.6542
RMSE  : 0.8389
R²    : -2.1006
Exact predictions (rounded 3dp): 0 / 701
✅ Saved predictions → plots/diff_test_preds.csv
> python -u "/Users/mrinalraj/Documents/FineDiving/TrainDifficulty.py"
✅ Loaded 2803 video samples with valid difficulty labels.
✅ Train dataset: 2102 samples

📊 Weighted Sampling Summary:
  Diff  1.6 → 14 samples | weight=6.8247
  Diff  1.8 → 5 samples | weight=19.1091
  Diff  1.9 → 22 samples | weight=4.3430
  Diff  2.0 → 155 samples | weight=0.6164
  Diff  2.1 → 7 samples | weight=13.6494
  Diff  2.3 → 1 samples | weight=95.5455
  Diff  2.4 → 8 samples | weight=11.9432
  Diff  2.6 → 8 samples | weight=11.9432
  Diff  2.7 → 32 samples | weight=2.9858
  Diff  2.8 → 82 samples | weight=1.1652
  Diff  2.9 → 55 samples | weight=1.7372
  Diff  3.0 → 437 samples | weight=0.2186
  Diff  3.1 → 90 samples | weight=1.0616
  Diff  3.2 → 369 samples | weight=0.2589
  Diff  3.3 → 151 samples | weight=0.6328
  Diff  3.4 → 274 samples | weight=0.3487
  Diff  3.5 → 94 samples | weight=1.0164
  Diff  3.6 → 154 samples | weight=0.6204
  Diff  3.7 → 66 samples | weight=1.4477
  Diff  3.8 → 50 samples | weight=1.9109
  Diff  3.9 → 21 samples | weight=4.5498
  Diff  4.1 → 7 samples | weight=13.6494
==========================================================================================
Layer (type:depth-idx)                   Output Shape              Param #
==========================================================================================
DifficultyRegressor                      [32]                      --
├─Conv1d: 1-1                            [32, 256, 152]            2,560,256
├─Sequential: 1-2                        [32, 256, 152]            --
│    └─ResidualBlock: 2-1                [32, 256, 152]            --
│    │    └─Conv1d: 3-1                  [32, 256, 152]            196,864
│    │    └─BatchNorm1d: 3-2             [32, 256, 152]            512
│    │    └─Conv1d: 3-3                  [32, 256, 152]            196,864
│    │    └─BatchNorm1d: 3-4             [32, 256, 152]            512
│    └─ResidualBlock: 2-2                [32, 256, 152]            --
│    │    └─Conv1d: 3-5                  [32, 256, 152]            196,864
│    │    └─BatchNorm1d: 3-6             [32, 256, 152]            512
│    │    └─Conv1d: 3-7                  [32, 256, 152]            196,864
│    │    └─BatchNorm1d: 3-8             [32, 256, 152]            512
│    └─ResidualBlock: 2-3                [32, 256, 152]            --
│    │    └─Conv1d: 3-9                  [32, 256, 152]            196,864
│    │    └─BatchNorm1d: 3-10            [32, 256, 152]            512
│    │    └─Conv1d: 3-11                 [32, 256, 152]            196,864
│    │    └─BatchNorm1d: 3-12            [32, 256, 152]            512
│    └─ResidualBlock: 2-4                [32, 256, 152]            --
│    │    └─Conv1d: 3-13                 [32, 256, 152]            196,864
│    │    └─BatchNorm1d: 3-14            [32, 256, 152]            512
│    │    └─Conv1d: 3-15                 [32, 256, 152]            196,864
│    │    └─BatchNorm1d: 3-16            [32, 256, 152]            512
├─AttnPool1D: 1-3                        [32, 256]                 256
├─Dropout: 1-4                           [32, 256]                 --
├─Sequential: 1-5                        [32, 1]                   --
│    └─Linear: 2-5                       [32, 128]                 32,896
│    └─ReLU: 2-6                         [32, 128]                 --
│    └─Linear: 2-7                       [32, 1]                   129
==========================================================================================
Total params: 4,172,545
Trainable params: 4,172,545
Non-trainable params: 0
Total mult-adds (Units.GIGABYTES): 20.11
==========================================================================================
Input size (MB): 38.91
Forward/backward pass size (MB): 169.44
Params size (MB): 16.69
Estimated Total Size (MB): 225.05
==========================================================================================
Epoch 01 | MSE 7.6329 | MAE 0.9274 | RMSE 1.1834
Epoch 02 | MSE 3.4784 | MAE 0.6370 | RMSE 0.7866
Epoch 03 | MSE 1.9701 | MAE 0.5107 | RMSE 0.6641
Epoch 04 | MSE 1.6790 | MAE 0.4883 | RMSE 0.6303
Epoch 05 | MSE 1.3918 | MAE 0.4499 | RMSE 0.5867
Epoch 06 | MSE 1.0342 | MAE 0.4294 | RMSE 0.5619
Epoch 07 | MSE 0.8122 | MAE 0.3877 | RMSE 0.5163
Epoch 08 | MSE 0.7286 | MAE 0.3502 | RMSE 0.4717
Epoch 09 | MSE 0.6824 | MAE 0.3431 | RMSE 0.4616
Epoch 10 | MSE 0.5752 | MAE 0.3390 | RMSE 0.4632
Epoch 11 | MSE 0.6445 | MAE 0.3442 | RMSE 0.4649
Epoch 12 | MSE 0.5688 | MAE 0.3291 | RMSE 0.4426
Epoch 13 | MSE 0.4946 | MAE 0.3099 | RMSE 0.4219
Epoch 14 | MSE 0.4893 | MAE 0.3067 | RMSE 0.4203
Epoch 15 | MSE 0.4918 | MAE 0.3005 | RMSE 0.4138
Epoch 16 | MSE 0.4839 | MAE 0.2820 | RMSE 0.3837
Epoch 17 | MSE 0.5295 | MAE 0.3030 | RMSE 0.4092
Epoch 18 | MSE 0.5070 | MAE 0.2938 | RMSE 0.4059
Epoch 19 | MSE 0.4217 | MAE 0.2859 | RMSE 0.3908
Epoch 20 | MSE 0.4413 | MAE 0.2826 | RMSE 0.3910
Epoch 21 | MSE 0.3846 | MAE 0.2720 | RMSE 0.3720
Epoch 22 | MSE 0.4026 | MAE 0.2745 | RMSE 0.3786
Epoch 23 | MSE 0.5355 | MAE 0.2920 | RMSE 0.3995
Epoch 24 | MSE 0.3874 | MAE 0.2777 | RMSE 0.3815
Epoch 25 | MSE 0.3863 | MAE 0.2598 | RMSE 0.3612
Epoch 26 | MSE 0.3456 | MAE 0.2613 | RMSE 0.3691
Epoch 27 | MSE 0.3540 | MAE 0.2661 | RMSE 0.3712
Epoch 28 | MSE 0.3559 | MAE 0.2493 | RMSE 0.3447
Epoch 29 | MSE 0.3431 | MAE 0.2580 | RMSE 0.3543
Epoch 30 | MSE 0.3597 | MAE 0.2575 | RMSE 0.3586
Epoch 31 | MSE 0.3064 | MAE 0.2494 | RMSE 0.3515
Epoch 32 | MSE 0.3239 | MAE 0.2427 | RMSE 0.3393
Epoch 33 | MSE 0.4061 | MAE 0.2527 | RMSE 0.3522
Epoch 34 | MSE 0.3603 | MAE 0.2622 | RMSE 0.3618
Epoch 35 | MSE 0.3473 | MAE 0.2488 | RMSE 0.3495
Epoch 36 | MSE 0.3633 | MAE 0.2522 | RMSE 0.3451
Epoch 37 | MSE 0.2979 | MAE 0.2306 | RMSE 0.3275
Epoch 38 | MSE 0.3476 | MAE 0.2414 | RMSE 0.3379
Epoch 39 | MSE 0.3260 | MAE 0.2421 | RMSE 0.3342
Epoch 40 | MSE 0.2927 | MAE 0.2397 | RMSE 0.3333
Epoch 41 | MSE 0.2834 | MAE 0.2327 | RMSE 0.3233
Epoch 42 | MSE 0.2947 | MAE 0.2309 | RMSE 0.3192
Epoch 43 | MSE 0.2826 | MAE 0.2273 | RMSE 0.3133
Epoch 44 | MSE 0.2983 | MAE 0.2265 | RMSE 0.3111
Epoch 45 | MSE 0.3356 | MAE 0.2361 | RMSE 0.3267
Epoch 46 | MSE 0.2864 | MAE 0.2302 | RMSE 0.3258
Epoch 47 | MSE 0.3185 | MAE 0.2309 | RMSE 0.3208
Epoch 48 | MSE 0.2710 | MAE 0.2237 | RMSE 0.3106
Epoch 49 | MSE 0.2013 | MAE 0.2071 | RMSE 0.2918
Epoch 50 | MSE 0.2428 | MAE 0.2140 | RMSE 0.2972

✅ Training complete — weighted model saved to difficulty_regressor.pt
> python -u "/Users/mrinalraj/Documents/FineDiving/TestForDifficulty.py"
✅ Loaded 2803 video samples with valid difficulty labels.
🧪 Test set size: 701

📊 Difficulty Regression Test Metrics:
MSE   : 0.3084
MAE   : 0.4386
RMSE  : 0.5553
R²    : -0.3587
Exact predictions (rounded 3dp): 0 / 701
✅ Saved predictions → plots/diff_test_preds.csv
> python -u "/Users/mrinalraj/Documents/FineDiving/TrainDifficulty.py"
✅ Loaded 2803 video samples with valid difficulty labels.
✅ Train dataset: 2102 samples

📊 Weighted Sampling Summary:
  Diff  1.6 → 14 samples | weight=6.8247
  Diff  1.8 → 5 samples | weight=19.1091
  Diff  1.9 → 22 samples | weight=4.3430
  Diff  2.0 → 155 samples | weight=0.6164
  Diff  2.1 → 7 samples | weight=13.6494
  Diff  2.3 → 1 samples | weight=95.5455
  Diff  2.4 → 8 samples | weight=11.9432
  Diff  2.6 → 8 samples | weight=11.9432
  Diff  2.7 → 32 samples | weight=2.9858
  Diff  2.8 → 82 samples | weight=1.1652
  Diff  2.9 → 55 samples | weight=1.7372
  Diff  3.0 → 437 samples | weight=0.2186
  Diff  3.1 → 90 samples | weight=1.0616
  Diff  3.2 → 369 samples | weight=0.2589
  Diff  3.3 → 151 samples | weight=0.6328
  Diff  3.4 → 274 samples | weight=0.3487
  Diff  3.5 → 94 samples | weight=1.0164
  Diff  3.6 → 154 samples | weight=0.6204
  Diff  3.7 → 66 samples | weight=1.4477
  Diff  3.8 → 50 samples | weight=1.9109
  Diff  3.9 → 21 samples | weight=4.5498
  Diff  4.1 → 7 samples | weight=13.6494
==========================================================================================
Layer (type:depth-idx)                   Output Shape              Param #
==========================================================================================
DifficultyRegressor                      [32]                      --
├─Conv1d: 1-1                            [32, 256, 136]            2,560,256
├─Sequential: 1-2                        [32, 256, 136]            --
│    └─ResidualBlock: 2-1                [32, 256, 136]            --
│    │    └─Conv1d: 3-1                  [32, 256, 136]            196,864
│    │    └─BatchNorm1d: 3-2             [32, 256, 136]            512
│    │    └─Conv1d: 3-3                  [32, 256, 136]            196,864
│    │    └─BatchNorm1d: 3-4             [32, 256, 136]            512
│    └─ResidualBlock: 2-2                [32, 256, 136]            --
│    │    └─Conv1d: 3-5                  [32, 256, 136]            196,864
│    │    └─BatchNorm1d: 3-6             [32, 256, 136]            512
│    │    └─Conv1d: 3-7                  [32, 256, 136]            196,864
│    │    └─BatchNorm1d: 3-8             [32, 256, 136]            512
│    └─ResidualBlock: 2-3                [32, 256, 136]            --
│    │    └─Conv1d: 3-9                  [32, 256, 136]            196,864
│    │    └─BatchNorm1d: 3-10            [32, 256, 136]            512
│    │    └─Conv1d: 3-11                 [32, 256, 136]            196,864
│    │    └─BatchNorm1d: 3-12            [32, 256, 136]            512
│    └─ResidualBlock: 2-4                [32, 256, 136]            --
│    │    └─Conv1d: 3-13                 [32, 256, 136]            196,864
│    │    └─BatchNorm1d: 3-14            [32, 256, 136]            512
│    │    └─Conv1d: 3-15                 [32, 256, 136]            196,864
│    │    └─BatchNorm1d: 3-16            [32, 256, 136]            512
├─AttnPool1D: 1-3                        [32, 256]                 256
├─Dropout: 1-4                           [32, 256]                 --
├─Sequential: 1-5                        [32, 1]                   --
│    └─Linear: 2-5                       [32, 128]                 32,896
│    └─ReLU: 2-6                         [32, 128]                 --
│    └─Linear: 2-7                       [32, 1]                   129
==========================================================================================
Total params: 4,172,545
Trainable params: 4,172,545
Non-trainable params: 0
Total mult-adds (Units.GIGABYTES): 18.00
==========================================================================================
Input size (MB): 34.82
Forward/backward pass size (MB): 151.62
Params size (MB): 16.69
Estimated Total Size (MB): 203.12
==========================================================================================
Epoch 01 | MSE 9.4462 | MAE 0.9025 | RMSE 1.1203
Epoch 02 | MSE 5.1584 | MAE 0.7385 | RMSE 0.8886
Epoch 03 | MSE 3.9523 | MAE 0.6985 | RMSE 0.8410
Epoch 04 | MSE 4.3878 | MAE 0.7028 | RMSE 0.8423
Epoch 05 | MSE 3.9750 | MAE 0.7208 | RMSE 0.8614
Epoch 06 | MSE 3.7399 | MAE 0.7038 | RMSE 0.8554
Epoch 07 | MSE 2.1620 | MAE 0.5675 | RMSE 0.7216
Epoch 08 | MSE 1.7869 | MAE 0.5107 | RMSE 0.6569
Epoch 09 | MSE 1.5781 | MAE 0.4821 | RMSE 0.6286
Epoch 10 | MSE 1.0359 | MAE 0.4219 | RMSE 0.5651
Epoch 11 | MSE 1.3340 | MAE 0.4603 | RMSE 0.5907
Epoch 12 | MSE 0.8759 | MAE 0.3968 | RMSE 0.5240
Epoch 13 | MSE 0.8671 | MAE 0.3739 | RMSE 0.5060
Epoch 14 | MSE 0.7626 | MAE 0.3670 | RMSE 0.4893
Epoch 15 | MSE 1.1333 | MAE 0.3827 | RMSE 0.5150
Epoch 16 | MSE 0.8795 | MAE 0.3617 | RMSE 0.4918
Epoch 17 | MSE 0.5644 | MAE 0.3315 | RMSE 0.4527
Epoch 18 | MSE 0.5089 | MAE 0.3022 | RMSE 0.4153
Epoch 19 | MSE 0.4179 | MAE 0.2910 | RMSE 0.4067
Epoch 20 | MSE 0.4886 | MAE 0.2912 | RMSE 0.4061
Epoch 21 | MSE 0.5071 | MAE 0.2989 | RMSE 0.4161
Epoch 22 | MSE 0.4090 | MAE 0.2858 | RMSE 0.3966
Epoch 23 | MSE 0.5416 | MAE 0.2925 | RMSE 0.4061
Epoch 24 | MSE 0.4052 | MAE 0.2745 | RMSE 0.3796
Epoch 25 | MSE 0.4099 | MAE 0.2834 | RMSE 0.3910
Epoch 26 | MSE 0.4042 | MAE 0.2683 | RMSE 0.3713
Epoch 27 | MSE 0.3711 | MAE 0.2742 | RMSE 0.3795
Epoch 28 | MSE 0.3588 | MAE 0.2585 | RMSE 0.3658
Epoch 29 | MSE 0.3340 | MAE 0.2505 | RMSE 0.3562
Epoch 30 | MSE 0.3747 | MAE 0.2737 | RMSE 0.3805
Epoch 31 | MSE 0.3651 | MAE 0.2591 | RMSE 0.3622
Epoch 32 | MSE 0.5922 | MAE 0.3051 | RMSE 0.4092
Epoch 33 | MSE 0.4763 | MAE 0.2734 | RMSE 0.3796
Epoch 34 | MSE 0.3458 | MAE 0.2613 | RMSE 0.3635
Epoch 35 | MSE 0.3211 | MAE 0.2554 | RMSE 0.3606
Epoch 36 | MSE 0.2958 | MAE 0.2544 | RMSE 0.3579
Epoch 37 | MSE 0.3456 | MAE 0.2488 | RMSE 0.3479
Epoch 38 | MSE 0.3749 | MAE 0.2579 | RMSE 0.3502
Epoch 39 | MSE 0.3298 | MAE 0.2390 | RMSE 0.3343
Epoch 40 | MSE 0.3538 | MAE 0.2473 | RMSE 0.3406
Epoch 41 | MSE 0.2993 | MAE 0.2392 | RMSE 0.3381
Epoch 42 | MSE 0.2791 | MAE 0.2309 | RMSE 0.3290
Epoch 43 | MSE 0.2820 | MAE 0.2410 | RMSE 0.3352
Epoch 44 | MSE 0.4423 | MAE 0.2657 | RMSE 0.3589
Epoch 45 | MSE 0.3297 | MAE 0.2385 | RMSE 0.3300
Epoch 46 | MSE 0.2559 | MAE 0.2264 | RMSE 0.3218
Epoch 47 | MSE 0.3377 | MAE 0.2397 | RMSE 0.3339
Epoch 48 | MSE 0.4116 | MAE 0.2598 | RMSE 0.3514
Epoch 49 | MSE 0.4075 | MAE 0.2651 | RMSE 0.3619
Epoch 50 | MSE 0.2769 | MAE 0.2308 | RMSE 0.3214
Epoch 51 | MSE 0.2324 | MAE 0.2301 | RMSE 0.3250
Epoch 52 | MSE 0.3404 | MAE 0.2451 | RMSE 0.3416
Epoch 53 | MSE 0.4252 | MAE 0.2530 | RMSE 0.3420
Epoch 54 | MSE 0.3444 | MAE 0.2452 | RMSE 0.3352
Epoch 55 | MSE 0.2473 | MAE 0.2237 | RMSE 0.3203
Epoch 56 | MSE 0.1954 | MAE 0.2073 | RMSE 0.2974
Epoch 57 | MSE 0.2869 | MAE 0.2233 | RMSE 0.3115
Epoch 58 | MSE 0.2742 | MAE 0.2283 | RMSE 0.3191
Epoch 59 | MSE 0.2708 | MAE 0.2149 | RMSE 0.3024
Epoch 60 | MSE 0.2461 | MAE 0.2141 | RMSE 0.3044
Epoch 61 | MSE 0.2611 | MAE 0.2242 | RMSE 0.3164
Epoch 62 | MSE 0.3944 | MAE 0.2344 | RMSE 0.3223
Epoch 63 | MSE 0.2743 | MAE 0.2414 | RMSE 0.3347
Epoch 64 | MSE 0.2844 | MAE 0.2275 | RMSE 0.3164
Epoch 65 | MSE 0.2342 | MAE 0.2140 | RMSE 0.2977
Epoch 66 | MSE 0.2015 | MAE 0.2021 | RMSE 0.2853
Epoch 67 | MSE 0.2413 | MAE 0.2118 | RMSE 0.2929
Epoch 68 | MSE 0.2016 | MAE 0.2024 | RMSE 0.2825
Epoch 69 | MSE 0.2554 | MAE 0.2176 | RMSE 0.3048
Epoch 70 | MSE 0.2678 | MAE 0.2186 | RMSE 0.3009
Epoch 71 | MSE 0.2458 | MAE 0.2133 | RMSE 0.2962
Epoch 72 | MSE 0.2612 | MAE 0.2075 | RMSE 0.2926
Epoch 73 | MSE 0.2208 | MAE 0.2021 | RMSE 0.2841
Epoch 74 | MSE 0.2378 | MAE 0.2063 | RMSE 0.2903
Epoch 75 | MSE 0.4104 | MAE 0.2355 | RMSE 0.3223
Epoch 76 | MSE 0.2164 | MAE 0.2062 | RMSE 0.2887
Epoch 77 | MSE 0.1903 | MAE 0.1965 | RMSE 0.2797
Epoch 78 | MSE 0.1813 | MAE 0.1897 | RMSE 0.2728
Epoch 79 | MSE 0.1989 | MAE 0.2002 | RMSE 0.2864
Epoch 80 | MSE 0.2622 | MAE 0.2111 | RMSE 0.3054
Epoch 81 | MSE 5.3463 | MAE 0.5865 | RMSE 0.7759
Epoch 82 | MSE 1.5485 | MAE 0.4574 | RMSE 0.6027
Epoch 83 | MSE 0.6215 | MAE 0.3164 | RMSE 0.4226
Epoch 84 | MSE 0.9138 | MAE 0.3255 | RMSE 0.4254
Epoch 85 | MSE 0.4316 | MAE 0.2765 | RMSE 0.3709
Epoch 86 | MSE 0.3456 | MAE 0.2528 | RMSE 0.3479
Epoch 87 | MSE 0.3325 | MAE 0.2472 | RMSE 0.3367
Epoch 88 | MSE 0.4375 | MAE 0.2661 | RMSE 0.3562
Epoch 89 | MSE 0.2905 | MAE 0.2314 | RMSE 0.3141
Epoch 90 | MSE 0.3140 | MAE 0.2326 | RMSE 0.3230
Epoch 91 | MSE 0.2586 | MAE 0.2244 | RMSE 0.3119
Epoch 92 | MSE 0.2569 | MAE 0.2145 | RMSE 0.2993
Epoch 93 | MSE 0.2259 | MAE 0.2114 | RMSE 0.2963
Epoch 94 | MSE 0.2097 | MAE 0.2039 | RMSE 0.2900
Epoch 95 | MSE 0.2119 | MAE 0.2109 | RMSE 0.2920
Epoch 96 | MSE 0.2243 | MAE 0.2073 | RMSE 0.2901
Epoch 97 | MSE 0.2404 | MAE 0.2222 | RMSE 0.3053
Epoch 98 | MSE 0.1959 | MAE 0.2076 | RMSE 0.2906
Epoch 99 | MSE 0.2104 | MAE 0.1958 | RMSE 0.2837
Epoch 100 | MSE 0.2087 | MAE 0.2066 | RMSE 0.2926

✅ Training complete — weighted model saved to difficulty_regressor.pt
> python -u "/Users/mrinalraj/Documents/FineDiving/TestForDifficulty.py"
✅ Loaded 2803 video samples with valid difficulty labels.
🧪 Test set size: 701

📊 Difficulty Regression Test Metrics:
MSE   : 0.3843
MAE   : 0.4739
RMSE  : 0.6199
R²    : -0.6931
Exact predictions (rounded 3dp): 1 / 701
✅ Saved predictions → plots/diff_test_preds.csv


✅ Loaded 2803 video samples with valid difficulty labels.
🧪 Test set size: 701

📊 Difficulty Regression Test Metrics:
MSE   : 0.2976
MAE   : 0.4297
RMSE  : 0.5456
R²    : -0.3113
Exact predictions (rounded 3dp): 0 / 701
✅ Saved predictions → plots/diff_test_preds.csv