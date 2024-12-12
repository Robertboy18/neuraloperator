(1000, 4, 2048) (1000, 1, 2048)
Total number of samples: 1000
Input data shape: (1000, 4, 2048)
Output series shape: (1000, 1, 2048)
Batch input series shape: torch.Size([32, 4, 2048])
Batch output series shape: torch.Size([32, 1, 2048])
Dtype torch.complex64 torch.complex64

Our model has 33919746 parameters.
torch.Size([8, 128, 1]) torch.Size([128, 128, 128, 2]) torch.Size([128, 128, 128, 2]) torch.Size([128, 128, 128, 2])

### MODEL ###
 FNO(
  (positional_embedding): GridEmbeddingND()
  (fno_blocks): FNOBlocks(
    (convs): SpectralConv(
      (weight): ModuleList(
        (0-7): 8 x ComplexDenseTensor(shape=torch.Size([128, 128, 128]), rank=None)
      )
    )
    (fno_skips): ModuleList(
      (0-7): 8 x ComplexValued(
        (fr): Conv1d(128, 128, kernel_size=(1,), stride=(1,), bias=False)
        (fi): Conv1d(128, 128, kernel_size=(1,), stride=(1,), bias=False)
      )
    )
  )
  (lifting): ComplexValued(
    (fr): ChannelMLP(
      (fcs): ModuleList(
        (0): Conv1d(5, 256, kernel_size=(1,), stride=(1,))
        (1): Conv1d(256, 128, kernel_size=(1,), stride=(1,))
      )
    )
    (fi): ChannelMLP(
      (fcs): ModuleList(
        (0): Conv1d(5, 256, kernel_size=(1,), stride=(1,))
        (1): Conv1d(256, 128, kernel_size=(1,), stride=(1,))
      )
    )
  )
  (projection): ComplexValued(
    (fr): ChannelMLP(
      (fcs): ModuleList(
        (0): Conv1d(128, 128, kernel_size=(1,), stride=(1,))
        (1): Conv1d(128, 1, kernel_size=(1,), stride=(1,))
      )
    )
    (fi): ChannelMLP(
      (fcs): ModuleList(
        (0): Conv1d(128, 128, kernel_size=(1,), stride=(1,))
        (1): Conv1d(128, 1, kernel_size=(1,), stride=(1,))
      )
    )
  )
)

### OPTIMIZER ###
 AdamW (
Parameter Group 0
    betas: (0.9, 0.999)
    correct_bias: True
    eps: 1e-06
    initial_lr: 0.001
    lr: 0.001
    weight_decay: 2e-06

Parameter Group 1
    betas: (0.9, 0.999)
    correct_bias: True
    dim: 5
    eps: 1e-06
    initial_lr: 0.001
    lr: 0.001
    proj_type: std
    rank: 0.01
    scale: 1.0
    type: tucker
    update_proj_gap: 1
    weight_decay: 2e-06
)

### SCHEDULER ###
 <torch.optim.lr_scheduler.StepLR object at 0x7fcb4c710df0>

### LOSSES ###

 * Train: <neuralop.losses.data_losses.H1Loss object at 0x7fcb4c7110f0>

 * Test: {'H1': <neuralop.losses.data_losses.H1Loss object at 0x7fcb4c7110f0>, 'L2': <neuralop.losses.data_losses.LpLoss object at 0x7fcb4c7116c0>}
using standard method to load data to device.
using standard method to compute loss.
self.override_load_to_device=False
self.overrides_loss=False
Training on 800 samples
Testing on [200] samples         on resolutions ['test'].
Raw outputs of size out.shape=torch.Size([32, 1, 2048])
[0] time=2.93, avg_loss=0.7987, train_err=31.9482, test_H1=0.9977, test_L2=0.9977
[3] time=1.53, avg_loss=0.7955, train_err=31.8218, test_H1=0.9947, test_L2=0.9947
[6] time=1.53, avg_loss=0.7897, train_err=31.5884, test_H1=0.9927, test_L2=0.9927
[9] time=1.53, avg_loss=0.7941, train_err=31.7645, test_H1=0.9975, test_L2=0.9975
[12] time=1.53, avg_loss=0.7850, train_err=31.4005, test_H1=0.9816, test_L2=0.9816
[15] time=1.53, avg_loss=0.7684, train_err=30.7348, test_H1=0.9544, test_L2=0.9544
[18] time=1.53, avg_loss=0.7647, train_err=30.5867, test_H1=0.9563, test_L2=0.9563
[21] time=1.53, avg_loss=0.7573, train_err=30.2911, test_H1=0.9594, test_L2=0.9594
[24] time=1.53, avg_loss=0.7519, train_err=30.0749, test_H1=0.9577, test_L2=0.9577
[27] time=1.53, avg_loss=0.7229, train_err=28.9171, test_H1=0.8964, test_L2=0.8964
[30] time=1.53, avg_loss=0.7138, train_err=28.5520, test_H1=0.8872, test_L2=0.8872
[33] time=1.53, avg_loss=0.6930, train_err=27.7189, test_H1=0.8606, test_L2=0.8606
[36] time=1.53, avg_loss=0.7106, train_err=28.4247, test_H1=0.8918, test_L2=0.8918
[39] time=1.53, avg_loss=0.7189, train_err=28.7541, test_H1=0.8630, test_L2=0.8630
[42] time=1.53, avg_loss=0.6563, train_err=26.2511, test_H1=0.8591, test_L2=0.8591
[45] time=1.53, avg_loss=0.6342, train_err=25.3690, test_H1=0.7951, test_L2=0.7951
[48] time=1.53, avg_loss=0.6203, train_err=24.8111, test_H1=0.7716, test_L2=0.7716
[51] time=1.53, avg_loss=0.6236, train_err=24.9424, test_H1=0.7244, test_L2=0.7244
[54] time=1.53, avg_loss=0.4857, train_err=19.4287, test_H1=0.6416, test_L2=0.6416
[57] time=1.53, avg_loss=0.5219, train_err=20.8771, test_H1=0.6636, test_L2=0.6636
[60] time=1.53, avg_loss=0.4431, train_err=17.7230, test_H1=0.5938, test_L2=0.5938
[63] time=1.53, avg_loss=0.4519, train_err=18.0763, test_H1=0.6069, test_L2=0.6069
[66] time=1.53, avg_loss=0.4643, train_err=18.5701, test_H1=0.5875, test_L2=0.5875
[69] time=1.53, avg_loss=0.4299, train_err=17.1940, test_H1=0.5226, test_L2=0.5226
[72] time=1.53, avg_loss=0.3860, train_err=15.4404, test_H1=0.5239, test_L2=0.5239
[75] time=1.53, avg_loss=0.3921, train_err=15.6843, test_H1=0.4885, test_L2=0.4885
[78] time=1.53, avg_loss=0.4017, train_err=16.0661, test_H1=0.5081, test_L2=0.5081
[81] time=1.53, avg_loss=0.3801, train_err=15.2041, test_H1=0.5373, test_L2=0.5373
[84] time=1.53, avg_loss=0.4050, train_err=16.1999, test_H1=0.4891, test_L2=0.4891
[87] time=1.53, avg_loss=0.3576, train_err=14.3042, test_H1=0.4750, test_L2=0.4750
[90] time=1.55, avg_loss=0.3364, train_err=13.4569, test_H1=0.4892, test_L2=0.4892
[93] time=3.24, avg_loss=0.3378, train_err=13.5140, test_H1=0.4755, test_L2=0.4755
[96] time=3.31, avg_loss=0.3296, train_err=13.1849, test_H1=0.4384, test_L2=0.4384
[99] time=3.38, avg_loss=0.3234, train_err=12.9368, test_H1=0.4418, test_L2=0.4418
[102] time=3.23, avg_loss=0.3297, train_err=13.1870, test_H1=0.4314, test_L2=0.4314
[105] time=3.37, avg_loss=0.3090, train_err=12.3608, test_H1=0.4155, test_L2=0.4155
[108] time=3.38, avg_loss=0.3134, train_err=12.5357, test_H1=0.4215, test_L2=0.4215
[111] time=3.24, avg_loss=0.3207, train_err=12.8300, test_H1=0.4306, test_L2=0.4306
[114] time=3.38, avg_loss=0.3318, train_err=13.2714, test_H1=0.4969, test_L2=0.4969
[117] time=3.25, avg_loss=0.3067, train_err=12.2698, test_H1=0.4795, test_L2=0.4795
[120] time=3.24, avg_loss=0.2995, train_err=11.9819, test_H1=0.3989, test_L2=0.3989
[123] time=3.39, avg_loss=0.2822, train_err=11.2886, test_H1=0.3908, test_L2=0.3908
[126] time=3.25, avg_loss=0.2897, train_err=11.5899, test_H1=0.4521, test_L2=0.4521
[129] time=3.36, avg_loss=0.3361, train_err=13.4443, test_H1=0.4248, test_L2=0.4248
[132] time=3.37, avg_loss=0.2779, train_err=11.1149, test_H1=0.3911, test_L2=0.3911
[135] time=3.22, avg_loss=0.2806, train_err=11.2250, test_H1=0.3852, test_L2=0.3852
[138] time=3.38, avg_loss=0.2849, train_err=11.3944, test_H1=0.3811, test_L2=0.3811
[141] time=3.38, avg_loss=0.2724, train_err=10.8963, test_H1=0.3874, test_L2=0.3874
[144] time=3.24, avg_loss=0.2771, train_err=11.0840, test_H1=0.4266, test_L2=0.4266
[147] time=3.38, avg_loss=0.2851, train_err=11.4047, test_H1=0.4137, test_L2=0.4137
[150] time=3.28, avg_loss=0.2881, train_err=11.5229, test_H1=0.3773, test_L2=0.3773
[153] time=3.26, avg_loss=0.2974, train_err=11.8958, test_H1=0.3922, test_L2=0.3922
[156] time=3.38, avg_loss=0.2735, train_err=10.9401, test_H1=0.3692, test_L2=0.3692
[159] time=3.23, avg_loss=0.2882, train_err=11.5297, test_H1=0.3918, test_L2=0.3918
[162] time=3.30, avg_loss=0.2709, train_err=10.8362, test_H1=0.3682, test_L2=0.3682
[165] time=3.38, avg_loss=0.2707, train_err=10.8269, test_H1=0.3732, test_L2=0.3732
[168] time=3.24, avg_loss=0.2694, train_err=10.7772, test_H1=0.3946, test_L2=0.3946
[171] time=3.38, avg_loss=0.2675, train_err=10.7008, test_H1=0.3757, test_L2=0.3757
[174] time=3.38, avg_loss=0.2533, train_err=10.1300, test_H1=0.3671, test_L2=0.3671
[177] time=3.25, avg_loss=0.2532, train_err=10.1268, test_H1=0.3831, test_L2=0.3831
[180] time=3.38, avg_loss=0.2533, train_err=10.1321, test_H1=0.3525, test_L2=0.3525
[183] time=3.31, avg_loss=0.2630, train_err=10.5205, test_H1=0.3621, test_L2=0.3621
[186] time=3.25, avg_loss=0.2578, train_err=10.3134, test_H1=0.3657, test_L2=0.3657
[189] time=3.38, avg_loss=0.2594, train_err=10.3768, test_H1=0.3655, test_L2=0.3655
[192] time=3.24, avg_loss=0.2753, train_err=11.0120, test_H1=0.3934, test_L2=0.3934
[195] time=3.30, avg_loss=0.2439, train_err=9.7579, test_H1=0.3486, test_L2=0.3486
[198] time=3.37, avg_loss=0.2497, train_err=9.9891, test_H1=0.3663, test_L2=0.3663
[201] time=3.24, avg_loss=0.3173, train_err=12.6922, test_H1=0.4141, test_L2=0.4141
[204] time=3.38, avg_loss=0.2454, train_err=9.8171, test_H1=0.3457, test_L2=0.3457
[207] time=3.38, avg_loss=0.2421, train_err=9.6829, test_H1=0.3578, test_L2=0.3578
[210] time=3.25, avg_loss=0.2366, train_err=9.4643, test_H1=0.3393, test_L2=0.3393
[213] time=3.38, avg_loss=0.2407, train_err=9.6267, test_H1=0.3406, test_L2=0.3406
[216] time=3.26, avg_loss=0.2364, train_err=9.4578, test_H1=0.3377, test_L2=0.3377
[219] time=3.24, avg_loss=0.2447, train_err=9.7866, test_H1=0.3445, test_L2=0.3445
[222] time=3.39, avg_loss=0.2550, train_err=10.1999, test_H1=0.3598, test_L2=0.3598
[225] time=3.24, avg_loss=0.2381, train_err=9.5245, test_H1=0.3560, test_L2=0.3560
[228] time=3.36, avg_loss=0.2343, train_err=9.3705, test_H1=0.3442, test_L2=0.3442
[231] time=3.38, avg_loss=0.2486, train_err=9.9437, test_H1=0.3782, test_L2=0.3782
[234] time=3.24, avg_loss=0.2479, train_err=9.9146, test_H1=0.3428, test_L2=0.3428
[237] time=3.38, avg_loss=0.2513, train_err=10.0519, test_H1=0.3661, test_L2=0.3661
[240] time=3.37, avg_loss=0.2273, train_err=9.0930, test_H1=0.3308, test_L2=0.3308
[243] time=3.24, avg_loss=0.2751, train_err=11.0043, test_H1=0.3880, test_L2=0.3880
[246] time=3.38, avg_loss=0.2454, train_err=9.8166, test_H1=0.3600, test_L2=0.3600
[249] time=3.25, avg_loss=0.2199, train_err=8.7948, test_H1=0.3352, test_L2=0.3352
[252] time=3.29, avg_loss=0.2219, train_err=8.8744, test_H1=0.3299, test_L2=0.3299
[255] time=3.39, avg_loss=0.2139, train_err=8.5568, test_H1=0.3201, test_L2=0.3201
[258] time=3.25, avg_loss=0.2224, train_err=8.8972, test_H1=0.3439, test_L2=0.3439
[261] time=3.38, avg_loss=0.2145, train_err=8.5799, test_H1=0.3297, test_L2=0.3297
[264] time=3.38, avg_loss=0.2200, train_err=8.8011, test_H1=0.3199, test_L2=0.3199
[267] time=3.23, avg_loss=0.2097, train_err=8.3881, test_H1=0.3163, test_L2=0.3163
[270] time=3.38, avg_loss=0.2118, train_err=8.4725, test_H1=0.3172, test_L2=0.3172
[273] time=3.31, avg_loss=0.2260, train_err=9.0382, test_H1=0.3238, test_L2=0.3238
[276] time=3.25, avg_loss=0.2036, train_err=8.1456, test_H1=0.3083, test_L2=0.3083
[279] time=3.38, avg_loss=0.2093, train_err=8.3740, test_H1=0.3098, test_L2=0.3098
[282] time=3.24, avg_loss=0.2112, train_err=8.4477, test_H1=0.3511, test_L2=0.3511
[285] time=3.37, avg_loss=0.2124, train_err=8.4958, test_H1=0.3190, test_L2=0.3190
[288] time=3.38, avg_loss=0.2129, train_err=8.5156, test_H1=0.3296, test_L2=0.3296
[291] time=3.24, avg_loss=0.2080, train_err=8.3207, test_H1=0.3124, test_L2=0.3124
[294] time=3.39, avg_loss=0.2026, train_err=8.1044, test_H1=0.3105, test_L2=0.3105
[297] time=3.39, avg_loss=0.2016, train_err=8.0647, test_H1=0.3036, test_L2=0.3036
[300] time=3.24, avg_loss=0.2059, train_err=8.2348, test_H1=0.3038, test_L2=0.3038
[303] time=3.37, avg_loss=0.2005, train_err=8.0210, test_H1=0.3409, test_L2=0.3409
[306] time=3.24, avg_loss=0.2022, train_err=8.0889, test_H1=0.3056, test_L2=0.3056
[309] time=3.29, avg_loss=0.1900, train_err=7.5998, test_H1=0.2937, test_L2=0.2937
[312] time=3.39, avg_loss=0.2134, train_err=8.5362, test_H1=0.3362, test_L2=0.3362
[315] time=3.24, avg_loss=0.2089, train_err=8.3574, test_H1=0.3328, test_L2=0.3328
[318] time=3.39, avg_loss=0.2105, train_err=8.4189, test_H1=0.3144, test_L2=0.3144
[321] time=3.39, avg_loss=0.1922, train_err=7.6887, test_H1=0.3131, test_L2=0.3131
[324] time=3.25, avg_loss=0.1872, train_err=7.4895, test_H1=0.3019, test_L2=0.3019
[327] time=3.38, avg_loss=0.1875, train_err=7.4999, test_H1=0.2974, test_L2=0.2974
[330] time=3.31, avg_loss=0.1949, train_err=7.7965, test_H1=0.2951, test_L2=0.2951
[333] time=3.24, avg_loss=0.1833, train_err=7.3311, test_H1=0.2860, test_L2=0.2860
[336] time=3.37, avg_loss=0.1803, train_err=7.2123, test_H1=0.2835, test_L2=0.2835
[339] time=3.23, avg_loss=0.1812, train_err=7.2495, test_H1=0.2919, test_L2=0.2919
[342] time=3.38, avg_loss=0.1916, train_err=7.6644, test_H1=0.2848, test_L2=0.2848
[345] time=3.39, avg_loss=0.1915, train_err=7.6598, test_H1=0.2893, test_L2=0.2893
[348] time=3.23, avg_loss=0.1832, train_err=7.3268, test_H1=0.2903, test_L2=0.2903
[351] time=3.38, avg_loss=0.1770, train_err=7.0797, test_H1=0.2835, test_L2=0.2835
[354] time=3.38, avg_loss=0.1737, train_err=6.9496, test_H1=0.2832, test_L2=0.2832
[357] time=3.24, avg_loss=0.1762, train_err=7.0485, test_H1=0.2794, test_L2=0.2794
[360] time=3.38, avg_loss=0.1870, train_err=7.4810, test_H1=0.2896, test_L2=0.2896
[363] time=3.24, avg_loss=0.1775, train_err=7.1006, test_H1=0.2873, test_L2=0.2873
[366] time=3.24, avg_loss=0.1730, train_err=6.9190, test_H1=0.2745, test_L2=0.2745
[369] time=3.38, avg_loss=0.1675, train_err=6.7000, test_H1=0.2711, test_L2=0.2711
[372] time=3.23, avg_loss=0.1718, train_err=6.8728, test_H1=0.2789, test_L2=0.2789
[375] time=3.38, avg_loss=0.1741, train_err=6.9633, test_H1=0.2741, test_L2=0.2741
[378] time=3.39, avg_loss=0.1752, train_err=7.0077, test_H1=0.2768, test_L2=0.2768
[381] time=3.24, avg_loss=0.1916, train_err=7.6635, test_H1=0.2935, test_L2=0.2935
[384] time=3.38, avg_loss=0.1643, train_err=6.5724, test_H1=0.2689, test_L2=0.2689
[387] time=3.34, avg_loss=0.1676, train_err=6.7052, test_H1=0.2904, test_L2=0.2904
[390] time=3.24, avg_loss=0.1646, train_err=6.5845, test_H1=0.2651, test_L2=0.2651
[393] time=3.38, avg_loss=0.1754, train_err=7.0178, test_H1=0.2732, test_L2=0.2732
[396] time=3.25, avg_loss=0.1596, train_err=6.3859, test_H1=0.2666, test_L2=0.2666
[399] time=3.29, avg_loss=0.1585, train_err=6.3409, test_H1=0.2666, test_L2=0.2666
[402] time=3.37, avg_loss=0.1617, train_err=6.4698, test_H1=0.2694, test_L2=0.2694
[405] time=3.23, avg_loss=0.1580, train_err=6.3203, test_H1=0.2605, test_L2=0.2605
[408] time=3.37, avg_loss=0.1710, train_err=6.8409, test_H1=0.2993, test_L2=0.2993
[411] time=3.38, avg_loss=0.1557, train_err=6.2267, test_H1=0.2586, test_L2=0.2586
[414] time=3.25, avg_loss=0.1552, train_err=6.2080, test_H1=0.2603, test_L2=0.2603
[417] time=3.38, avg_loss=0.1623, train_err=6.4922, test_H1=0.2594, test_L2=0.2594
[420] time=3.28, avg_loss=0.1593, train_err=6.3711, test_H1=0.2667, test_L2=0.2667
[423] time=3.23, avg_loss=0.1596, train_err=6.3824, test_H1=0.2566, test_L2=0.2566
[426] time=3.38, avg_loss=0.1602, train_err=6.4072, test_H1=0.2635, test_L2=0.2635
[429] time=3.25, avg_loss=0.1593, train_err=6.3705, test_H1=0.2849, test_L2=0.2849
[432] time=3.35, avg_loss=0.1661, train_err=6.6456, test_H1=0.2753, test_L2=0.2753
[435] time=3.38, avg_loss=0.1539, train_err=6.1560, test_H1=0.2595, test_L2=0.2595
[438] time=3.22, avg_loss=0.1498, train_err=5.9914, test_H1=0.2553, test_L2=0.2553
[441] time=3.37, avg_loss=0.1501, train_err=6.0024, test_H1=0.2545, test_L2=0.2545
[444] time=3.38, avg_loss=0.1519, train_err=6.0771, test_H1=0.2526, test_L2=0.2526
[447] time=3.23, avg_loss=0.1510, train_err=6.0406, test_H1=0.2538, test_L2=0.2538
[450] time=3.38, avg_loss=0.1435, train_err=5.7380, test_H1=0.2506, test_L2=0.2506
[453] time=3.24, avg_loss=0.1474, train_err=5.8958, test_H1=0.2529, test_L2=0.2529
[456] time=3.24, avg_loss=0.1648, train_err=6.5919, test_H1=0.2622, test_L2=0.2622
[459] time=3.38, avg_loss=0.1504, train_err=6.0171, test_H1=0.2572, test_L2=0.2572
[462] time=3.24, avg_loss=0.1500, train_err=5.9996, test_H1=0.2567, test_L2=0.2567
[465] time=3.38, avg_loss=0.1473, train_err=5.8902, test_H1=0.2525, test_L2=0.2525
[468] time=3.38, avg_loss=0.1409, train_err=5.6371, test_H1=0.2590, test_L2=0.2590
[471] time=3.24, avg_loss=0.1420, train_err=5.6799, test_H1=0.2486, test_L2=0.2486
[474] time=3.38, avg_loss=0.1450, train_err=5.8017, test_H1=0.2633, test_L2=0.2633
[477] time=3.37, avg_loss=0.1494, train_err=5.9747, test_H1=0.2532, test_L2=0.2532
[480] time=3.24, avg_loss=0.1399, train_err=5.5978, test_H1=0.2485, test_L2=0.2485
[483] time=3.38, avg_loss=0.1502, train_err=6.0063, test_H1=0.2532, test_L2=0.2532
[486] time=3.24, avg_loss=0.1608, train_err=6.4302, test_H1=0.2714, test_L2=0.2714
[489] time=3.29, avg_loss=0.1443, train_err=5.7732, test_H1=0.2435, test_L2=0.2435
[492] time=3.38, avg_loss=0.1507, train_err=6.0297, test_H1=0.2510, test_L2=0.2510
[495] time=3.24, avg_loss=0.1391, train_err=5.5648, test_H1=0.2447, test_L2=0.2447
[498] time=3.37, avg_loss=0.1451, train_err=5.8058, test_H1=0.2539, test_L2=0.2539
[501] time=3.38, avg_loss=0.1331, train_err=5.3249, test_H1=0.2424, test_L2=0.2424
[504] time=3.23, avg_loss=0.1365, train_err=5.4610, test_H1=0.2556, test_L2=0.2556
[507] time=3.37, avg_loss=0.1362, train_err=5.4473, test_H1=0.2397, test_L2=0.2397
[510] time=3.34, avg_loss=0.1329, train_err=5.3159, test_H1=0.2411, test_L2=0.2411
[513] time=3.24, avg_loss=0.1325, train_err=5.3000, test_H1=0.2402, test_L2=0.2402
[516] time=3.38, avg_loss=0.1368, train_err=5.4715, test_H1=0.2438, test_L2=0.2438
[519] time=3.25, avg_loss=0.1332, train_err=5.3286, test_H1=0.2402, test_L2=0.2402
[522] time=3.37, avg_loss=0.1381, train_err=5.5237, test_H1=0.2424, test_L2=0.2424
[525] time=3.38, avg_loss=0.1358, train_err=5.4308, test_H1=0.2411, test_L2=0.2411
[528] time=3.25, avg_loss=0.1399, train_err=5.5948, test_H1=0.2477, test_L2=0.2477
[531] time=3.39, avg_loss=0.1408, train_err=5.6305, test_H1=0.2516, test_L2=0.2516
[534] time=3.38, avg_loss=0.1335, train_err=5.3411, test_H1=0.2383, test_L2=0.2383
[537] time=3.24, avg_loss=0.1323, train_err=5.2919, test_H1=0.2398, test_L2=0.2398
[540] time=3.37, avg_loss=0.1284, train_err=5.1355, test_H1=0.2353, test_L2=0.2353
[543] time=3.27, avg_loss=0.1313, train_err=5.2506, test_H1=0.2388, test_L2=0.2388
[546] time=3.26, avg_loss=0.1373, train_err=5.4921, test_H1=0.2448, test_L2=0.2448
[549] time=3.38, avg_loss=0.1500, train_err=5.9989, test_H1=0.2490, test_L2=0.2490
[552] time=3.25, avg_loss=0.1416, train_err=5.6655, test_H1=0.2439, test_L2=0.2439
[555] time=3.38, avg_loss=0.1315, train_err=5.2583, test_H1=0.2453, test_L2=0.2453
[558] time=3.39, avg_loss=0.1307, train_err=5.2281, test_H1=0.2362, test_L2=0.2362
[561] time=3.23, avg_loss=0.1291, train_err=5.1641, test_H1=0.2375, test_L2=0.2375
[564] time=3.39, avg_loss=0.1387, train_err=5.5489, test_H1=0.2396, test_L2=0.2396
[567] time=3.37, avg_loss=0.1330, train_err=5.3192, test_H1=0.2386, test_L2=0.2386
[570] time=3.23, avg_loss=0.1279, train_err=5.1173, test_H1=0.2339, test_L2=0.2339
[573] time=3.38, avg_loss=0.1288, train_err=5.1512, test_H1=0.2436, test_L2=0.2436
[576] time=3.23, avg_loss=0.1328, train_err=5.3104, test_H1=0.2360, test_L2=0.2360
[579] time=3.31, avg_loss=0.1294, train_err=5.1748, test_H1=0.2354, test_L2=0.2354
[582] time=3.38, avg_loss=0.1260, train_err=5.0384, test_H1=0.2419, test_L2=0.2419
[585] time=3.25, avg_loss=0.1323, train_err=5.2901, test_H1=0.2356, test_L2=0.2356
[588] time=3.39, avg_loss=0.1242, train_err=4.9660, test_H1=0.2299, test_L2=0.2299
[591] time=3.38, avg_loss=0.1237, train_err=4.9466, test_H1=0.2337, test_L2=0.2337
[594] time=3.24, avg_loss=0.1277, train_err=5.1096, test_H1=0.2469, test_L2=0.2469
[597] time=3.38, avg_loss=0.1313, train_err=5.2519, test_H1=0.2356, test_L2=0.2356
[600] time=3.30, avg_loss=0.1255, train_err=5.0210, test_H1=0.2309, test_L2=0.2309
[603] time=3.23, avg_loss=0.1211, train_err=4.8454, test_H1=0.2282, test_L2=0.2282
[606] time=3.38, avg_loss=0.1236, train_err=4.9432, test_H1=0.2313, test_L2=0.2313
[609] time=3.23, avg_loss=0.1223, train_err=4.8908, test_H1=0.2291, test_L2=0.2291
[612] time=3.38, avg_loss=0.1246, train_err=4.9835, test_H1=0.2407, test_L2=0.2407
[615] time=3.38, avg_loss=0.1256, train_err=5.0221, test_H1=0.2322, test_L2=0.2322
[618] time=3.24, avg_loss=0.1339, train_err=5.3545, test_H1=0.2389, test_L2=0.2389
[621] time=3.39, avg_loss=0.1263, train_err=5.0524, test_H1=0.2289, test_L2=0.2289
[624] time=3.39, avg_loss=0.1238, train_err=4.9504, test_H1=0.2371, test_L2=0.2371
[627] time=3.25, avg_loss=0.1215, train_err=4.8605, test_H1=0.2265, test_L2=0.2265
[630] time=3.39, avg_loss=0.1348, train_err=5.3918, test_H1=0.2476, test_L2=0.2476
[633] time=3.25, avg_loss=0.1222, train_err=4.8865, test_H1=0.2289, test_L2=0.2289
[636] time=3.22, avg_loss=0.1192, train_err=4.7677, test_H1=0.2264, test_L2=0.2264
[639] time=3.37, avg_loss=0.1256, train_err=5.0230, test_H1=0.2291, test_L2=0.2291
[642] time=3.24, avg_loss=0.1207, train_err=4.8266, test_H1=0.2268, test_L2=0.2268
[645] time=3.38, avg_loss=0.1212, train_err=4.8471, test_H1=0.2299, test_L2=0.2299
[648] time=3.39, avg_loss=0.1231, train_err=4.9228, test_H1=0.2285, test_L2=0.2285
[651] time=3.24, avg_loss=0.1187, train_err=4.7466, test_H1=0.2253, test_L2=0.2253
[654] time=3.37, avg_loss=0.1190, train_err=4.7584, test_H1=0.2275, test_L2=0.2275
[657] time=3.34, avg_loss=0.1175, train_err=4.6989, test_H1=0.2255, test_L2=0.2255
[660] time=3.25, avg_loss=0.1173, train_err=4.6920, test_H1=0.2273, test_L2=0.2273
[663] time=3.38, avg_loss=0.1184, train_err=4.7341, test_H1=0.2293, test_L2=0.2293
[666] time=3.24, avg_loss=0.1153, train_err=4.6124, test_H1=0.2235, test_L2=0.2235
[669] time=3.28, avg_loss=0.1253, train_err=5.0104, test_H1=0.2473, test_L2=0.2473
[672] time=3.37, avg_loss=0.1163, train_err=4.6520, test_H1=0.2233, test_L2=0.2233
[675] time=3.23, avg_loss=0.1147, train_err=4.5899, test_H1=0.2228, test_L2=0.2228
[678] time=3.38, avg_loss=0.1180, train_err=4.7209, test_H1=0.2259, test_L2=0.2259
[681] time=3.38, avg_loss=0.1205, train_err=4.8209, test_H1=0.2280, test_L2=0.2280
[684] time=3.26, avg_loss=0.1255, train_err=5.0194, test_H1=0.2284, test_L2=0.2284
[687] time=3.39, avg_loss=0.1151, train_err=4.6048, test_H1=0.2227, test_L2=0.2227
[690] time=3.28, avg_loss=0.1178, train_err=4.7132, test_H1=0.2288, test_L2=0.2288
[693] time=3.25, avg_loss=0.1157, train_err=4.6261, test_H1=0.2223, test_L2=0.2223
[696] time=3.38, avg_loss=0.1145, train_err=4.5817, test_H1=0.2239, test_L2=0.2239
[699] time=3.25, avg_loss=0.1232, train_err=4.9295, test_H1=0.2247, test_L2=0.2247
[702] time=3.35, avg_loss=0.1134, train_err=4.5357, test_H1=0.2267, test_L2=0.2267
[705] time=3.37, avg_loss=0.1187, train_err=4.7479, test_H1=0.2250, test_L2=0.2250
[708] time=3.24, avg_loss=0.1107, train_err=4.4270, test_H1=0.2199, test_L2=0.2199
[711] time=3.38, avg_loss=0.1128, train_err=4.5116, test_H1=0.2238, test_L2=0.2238
[714] time=3.37, avg_loss=0.1167, train_err=4.6679, test_H1=0.2245, test_L2=0.2245
[717] time=3.25, avg_loss=0.1125, train_err=4.5018, test_H1=0.2199, test_L2=0.2199
[720] time=3.38, avg_loss=0.1230, train_err=4.9217, test_H1=0.2236, test_L2=0.2236
[723] time=3.24, avg_loss=0.1125, train_err=4.5001, test_H1=0.2389, test_L2=0.2389
[726] time=3.25, avg_loss=0.1199, train_err=4.7941, test_H1=0.2268, test_L2=0.2268
[729] time=3.38, avg_loss=0.1163, train_err=4.6515, test_H1=0.2233, test_L2=0.2233
[732] time=3.25, avg_loss=0.1145, train_err=4.5795, test_H1=0.2213, test_L2=0.2213
[735] time=3.38, avg_loss=0.1102, train_err=4.4086, test_H1=0.2185, test_L2=0.2185
[738] time=3.38, avg_loss=0.1133, train_err=4.5315, test_H1=0.2200, test_L2=0.2200
[741] time=3.25, avg_loss=0.1131, train_err=4.5221, test_H1=0.2204, test_L2=0.2204
[744] time=3.37, avg_loss=0.1124, train_err=4.4947, test_H1=0.2244, test_L2=0.2244
[747] time=3.36, avg_loss=0.1129, train_err=4.5155, test_H1=0.2180, test_L2=0.2180
[750] time=3.25, avg_loss=0.1169, train_err=4.6753, test_H1=0.2256, test_L2=0.2256
[753] time=3.38, avg_loss=0.1088, train_err=4.3517, test_H1=0.2215, test_L2=0.2215
[756] time=3.24, avg_loss=0.1082, train_err=4.3279, test_H1=0.2169, test_L2=0.2169
[759] time=3.31, avg_loss=0.1092, train_err=4.3682, test_H1=0.2182, test_L2=0.2182
[762] time=3.38, avg_loss=0.1111, train_err=4.4442, test_H1=0.2178, test_L2=0.2178
[765] time=3.24, avg_loss=0.1101, train_err=4.4035, test_H1=0.2203, test_L2=0.2203
[768] time=3.39, avg_loss=0.1115, train_err=4.4618, test_H1=0.2207, test_L2=0.2207
[771] time=3.38, avg_loss=0.1095, train_err=4.3794, test_H1=0.2200, test_L2=0.2200
[774] time=3.24, avg_loss=0.1124, train_err=4.4959, test_H1=0.2238, test_L2=0.2238
[777] time=3.37, avg_loss=0.1104, train_err=4.4163, test_H1=0.2200, test_L2=0.2200
[780] time=3.30, avg_loss=0.1100, train_err=4.3998, test_H1=0.2189, test_L2=0.2189
[783] time=3.24, avg_loss=0.1086, train_err=4.3440, test_H1=0.2169, test_L2=0.2169
[786] time=3.38, avg_loss=0.1075, train_err=4.3014, test_H1=0.2178, test_L2=0.2178
[789] time=3.24, avg_loss=0.1126, train_err=4.5037, test_H1=0.2170, test_L2=0.2170
[792] time=3.38, avg_loss=0.1096, train_err=4.3833, test_H1=0.2187, test_L2=0.2187
[795] time=3.38, avg_loss=0.1084, train_err=4.3341, test_H1=0.2299, test_L2=0.2299
[798] time=3.25, avg_loss=0.1085, train_err=4.3411, test_H1=0.2158, test_L2=0.2158
[801] time=3.38, avg_loss=0.1058, train_err=4.2336, test_H1=0.2167, test_L2=0.2167
[804] time=3.39, avg_loss=0.1055, train_err=4.2220, test_H1=0.2155, test_L2=0.2155
[807] time=3.24, avg_loss=0.1062, train_err=4.2488, test_H1=0.2140, test_L2=0.2140
[810] time=3.38, avg_loss=0.1042, train_err=4.1692, test_H1=0.2162, test_L2=0.2162
[813] time=3.23, avg_loss=0.1044, train_err=4.1757, test_H1=0.2140, test_L2=0.2140
[816] time=3.31, avg_loss=0.1102, train_err=4.4071, test_H1=0.2242, test_L2=0.2242
[819] time=3.38, avg_loss=0.1064, train_err=4.2561, test_H1=0.2162, test_L2=0.2162
[822] time=3.26, avg_loss=0.1099, train_err=4.3949, test_H1=0.2156, test_L2=0.2156
[825] time=3.37, avg_loss=0.1074, train_err=4.2968, test_H1=0.2166, test_L2=0.2166
[828] time=3.38, avg_loss=0.1118, train_err=4.4738, test_H1=0.2179, test_L2=0.2179
[831] time=3.24, avg_loss=0.1031, train_err=4.1233, test_H1=0.2146, test_L2=0.2146
[834] time=3.38, avg_loss=0.1063, train_err=4.2501, test_H1=0.2166, test_L2=0.2166
[837] time=3.34, avg_loss=0.1032, train_err=4.1296, test_H1=0.2146, test_L2=0.2146
[840] time=3.24, avg_loss=0.1077, train_err=4.3087, test_H1=0.2169, test_L2=0.2169
[843] time=3.37, avg_loss=0.1067, train_err=4.2668, test_H1=0.2190, test_L2=0.2190
[846] time=3.24, avg_loss=0.1042, train_err=4.1687, test_H1=0.2152, test_L2=0.2152
[849] time=3.36, avg_loss=0.1053, train_err=4.2102, test_H1=0.2153, test_L2=0.2153
[852] time=3.38, avg_loss=0.1021, train_err=4.0850, test_H1=0.2154, test_L2=0.2154
[855] time=3.25, avg_loss=0.1052, train_err=4.2099, test_H1=0.2207, test_L2=0.2207
[858] time=3.39, avg_loss=0.1034, train_err=4.1359, test_H1=0.2129, test_L2=0.2129
[861] time=3.39, avg_loss=0.1035, train_err=4.1420, test_H1=0.2134, test_L2=0.2134
[864] time=3.25, avg_loss=0.1051, train_err=4.2052, test_H1=0.2180, test_L2=0.2180
[867] time=3.38, avg_loss=0.1047, train_err=4.1899, test_H1=0.2156, test_L2=0.2156
[870] time=3.26, avg_loss=0.1028, train_err=4.1127, test_H1=0.2147, test_L2=0.2147
[873] time=3.24, avg_loss=0.1023, train_err=4.0916, test_H1=0.2127, test_L2=0.2127
[876] time=3.37, avg_loss=0.1027, train_err=4.1081, test_H1=0.2133, test_L2=0.2133
[879] time=3.24, avg_loss=0.1023, train_err=4.0923, test_H1=0.2126, test_L2=0.2126
[882] time=3.37, avg_loss=0.1042, train_err=4.1679, test_H1=0.2156, test_L2=0.2156
[885] time=3.39, avg_loss=0.1012, train_err=4.0477, test_H1=0.2115, test_L2=0.2115
[888] time=3.26, avg_loss=0.1011, train_err=4.0449, test_H1=0.2113, test_L2=0.2113
[891] time=3.38, avg_loss=0.1032, train_err=4.1262, test_H1=0.2169, test_L2=0.2169
[894] time=3.34, avg_loss=0.1031, train_err=4.1250, test_H1=0.2154, test_L2=0.2154
[897] time=3.25, avg_loss=0.1027, train_err=4.1090, test_H1=0.2123, test_L2=0.2123
[900] time=3.38, avg_loss=0.1019, train_err=4.0771, test_H1=0.2207, test_L2=0.2207
[903] time=3.25, avg_loss=0.1013, train_err=4.0522, test_H1=0.2119, test_L2=0.2119
[906] time=3.31, avg_loss=0.1005, train_err=4.0218, test_H1=0.2142, test_L2=0.2142
[909] time=3.38, avg_loss=0.1024, train_err=4.0942, test_H1=0.2140, test_L2=0.2140
[912] time=3.24, avg_loss=0.1008, train_err=4.0322, test_H1=0.2107, test_L2=0.2107
[915] time=3.37, avg_loss=0.1012, train_err=4.0473, test_H1=0.2111, test_L2=0.2111
[918] time=3.38, avg_loss=0.1002, train_err=4.0097, test_H1=0.2124, test_L2=0.2124
[921] time=3.26, avg_loss=0.1068, train_err=4.2722, test_H1=0.2236, test_L2=0.2236
[924] time=3.38, avg_loss=0.1016, train_err=4.0638, test_H1=0.2137, test_L2=0.2137
[927] time=3.24, avg_loss=0.1005, train_err=4.0216, test_H1=0.2112, test_L2=0.2112
[930] time=3.25, avg_loss=0.1042, train_err=4.1695, test_H1=0.2128, test_L2=0.2128
[933] time=3.37, avg_loss=0.1014, train_err=4.0560, test_H1=0.2147, test_L2=0.2147
[936] time=3.24, avg_loss=0.1008, train_err=4.0337, test_H1=0.2120, test_L2=0.2120
[939] time=3.36, avg_loss=0.1013, train_err=4.0528, test_H1=0.2125, test_L2=0.2125
[942] time=3.38, avg_loss=0.0988, train_err=3.9526, test_H1=0.2103, test_L2=0.2103
[945] time=3.26, avg_loss=0.1001, train_err=4.0025, test_H1=0.2121, test_L2=0.2121
[948] time=3.38, avg_loss=0.0979, train_err=3.9159, test_H1=0.2133, test_L2=0.2133
[951] time=3.37, avg_loss=0.0978, train_err=3.9128, test_H1=0.2106, test_L2=0.2106
[954] time=3.24, avg_loss=0.0983, train_err=3.9304, test_H1=0.2104, test_L2=0.2104
[957] time=3.38, avg_loss=0.0991, train_err=3.9650, test_H1=0.2111, test_L2=0.2111
[960] time=3.24, avg_loss=0.0990, train_err=3.9583, test_H1=0.2092, test_L2=0.2092
[963] time=3.25, avg_loss=0.0974, train_err=3.8959, test_H1=0.2112, test_L2=0.2112
[966] time=3.39, avg_loss=0.0989, train_err=3.9573, test_H1=0.2139, test_L2=0.2139
[969] time=3.24, avg_loss=0.0984, train_err=3.9343, test_H1=0.2089, test_L2=0.2089
[972] time=3.38, avg_loss=0.0992, train_err=3.9687, test_H1=0.2109, test_L2=0.2109
[975] time=3.37, avg_loss=0.0970, train_err=3.8801, test_H1=0.2105, test_L2=0.2105
[978] time=3.23, avg_loss=0.0976, train_err=3.9042, test_H1=0.2089, test_L2=0.2089
[981] time=3.37, avg_loss=0.0980, train_err=3.9213, test_H1=0.2109, test_L2=0.2109
[984] time=3.37, avg_loss=0.0994, train_err=3.9755, test_H1=0.2115, test_L2=0.2115
[987] time=3.24, avg_loss=0.0973, train_err=3.8937, test_H1=0.2103, test_L2=0.2103
[990] time=3.39, avg_loss=0.0990, train_err=3.9583, test_H1=0.2092, test_L2=0.2092
[993] time=3.25, avg_loss=0.0967, train_err=3.8688, test_H1=0.2099, test_L2=0.2099
[996] time=3.31, avg_loss=0.0972, train_err=3.8874, test_H1=0.2126, test_L2=0.2126
[999] time=3.38, avg_loss=0.0985, train_err=3.9411, test_H1=0.2107, test_L2=0.2107
(1000, 4, 2048) (1000, 1, 2048)
Total number of samples: 1000
Input data shape: (1000, 4, 2048)
Output series shape: (1000, 1, 2048)
Batch input series shape: torch.Size([32, 4, 2048])
Batch output series shape: torch.Size([32, 1, 2048])
Dtype torch.complex64 torch.complex64

Our model has 33919746 parameters.
torch.Size([8, 128, 1]) torch.Size([128, 128, 128, 2]) torch.Size([128, 128, 128, 2]) torch.Size([128, 128, 128, 2])

### MODEL ###
 FNO(
  (positional_embedding): GridEmbeddingND()
  (fno_blocks): FNOBlocks(
    (convs): SpectralConv(
      (weight): ModuleList(
        (0-7): 8 x ComplexDenseTensor(shape=torch.Size([128, 128, 128]), rank=None)
      )
    )
    (fno_skips): ModuleList(
      (0-7): 8 x ComplexValued(
        (fr): Conv1d(128, 128, kernel_size=(1,), stride=(1,), bias=False)
        (fi): Conv1d(128, 128, kernel_size=(1,), stride=(1,), bias=False)
      )
    )
  )
  (lifting): ComplexValued(
    (fr): ChannelMLP(
      (fcs): ModuleList(
        (0): Conv1d(5, 256, kernel_size=(1,), stride=(1,))
        (1): Conv1d(256, 128, kernel_size=(1,), stride=(1,))
      )
    )
    (fi): ChannelMLP(
      (fcs): ModuleList(
        (0): Conv1d(5, 256, kernel_size=(1,), stride=(1,))
        (1): Conv1d(256, 128, kernel_size=(1,), stride=(1,))
      )
    )
  )
  (projection): ComplexValued(
    (fr): ChannelMLP(
      (fcs): ModuleList(
        (0): Conv1d(128, 128, kernel_size=(1,), stride=(1,))
        (1): Conv1d(128, 1, kernel_size=(1,), stride=(1,))
      )
    )
    (fi): ChannelMLP(
      (fcs): ModuleList(
        (0): Conv1d(128, 128, kernel_size=(1,), stride=(1,))
        (1): Conv1d(128, 1, kernel_size=(1,), stride=(1,))
      )
    )
  )
)

### OPTIMIZER ###
 AdamW (
Parameter Group 0
    betas: (0.9, 0.999)
    correct_bias: True
    eps: 1e-06
    initial_lr: 0.001
    lr: 0.001
    weight_decay: 2e-06

Parameter Group 1
    betas: (0.9, 0.999)
    correct_bias: True
    dim: 5
    eps: 1e-06
    initial_lr: 0.001
    lr: 0.001
    proj_type: std
    rank: 0.1
    scale: 1.0
    type: tucker
    update_proj_gap: 1
    weight_decay: 2e-06
)

### SCHEDULER ###
 <torch.optim.lr_scheduler.StepLR object at 0x7f7a77904f10>

### LOSSES ###

 * Train: <neuralop.losses.data_losses.H1Loss object at 0x7f7a77905210>

 * Test: {'H1': <neuralop.losses.data_losses.H1Loss object at 0x7f7a77905210>, 'L2': <neuralop.losses.data_losses.LpLoss object at 0x7f7a779057e0>}
using standard method to load data to device.
using standard method to compute loss.
self.override_load_to_device=False
self.overrides_loss=False
Training on 800 samples
Testing on [200] samples         on resolutions ['test'].
Raw outputs of size out.shape=torch.Size([32, 1, 2048])
[0] time=8.39, avg_loss=0.7996, train_err=31.9837, test_H1=0.9991, test_L2=0.9991
[3] time=3.26, avg_loss=0.7975, train_err=31.8989, test_H1=0.9975, test_L2=0.9975
[6] time=3.37, avg_loss=0.7925, train_err=31.6996, test_H1=0.9949, test_L2=0.9949
[9] time=3.36, avg_loss=0.7860, train_err=31.4418, test_H1=0.9953, test_L2=0.9953
[12] time=3.25, avg_loss=0.7854, train_err=31.4143, test_H1=0.9819, test_L2=0.9819
[15] time=3.39, avg_loss=0.7839, train_err=31.3542, test_H1=0.9819, test_L2=0.9819
[18] time=3.26, avg_loss=0.7647, train_err=30.5894, test_H1=0.9698, test_L2=0.9698
[21] time=3.35, avg_loss=0.7618, train_err=30.4722, test_H1=0.9706, test_L2=0.9706
[24] time=3.40, avg_loss=0.7534, train_err=30.1362, test_H1=0.9431, test_L2=0.9431
[27] time=3.26, avg_loss=0.7506, train_err=30.0249, test_H1=0.9447, test_L2=0.9447
[30] time=3.40, avg_loss=0.7575, train_err=30.2995, test_H1=0.9363, test_L2=0.9363
[33] time=3.39, avg_loss=0.7241, train_err=28.9647, test_H1=0.9218, test_L2=0.9218
[36] time=3.24, avg_loss=0.7116, train_err=28.4646, test_H1=0.9099, test_L2=0.9099
[39] time=3.39, avg_loss=0.7238, train_err=28.9526, test_H1=0.8852, test_L2=0.8852
[42] time=3.33, avg_loss=0.6971, train_err=27.8856, test_H1=0.9249, test_L2=0.9249
[45] time=3.26, avg_loss=0.6832, train_err=27.3269, test_H1=0.8769, test_L2=0.8769
[48] time=3.39, avg_loss=0.6661, train_err=26.6431, test_H1=0.9079, test_L2=0.9079
[51] time=3.27, avg_loss=0.6485, train_err=25.9389, test_H1=0.7964, test_L2=0.7964
[54] time=3.39, avg_loss=0.5701, train_err=22.8059, test_H1=0.6875, test_L2=0.6875
[57] time=3.39, avg_loss=0.5607, train_err=22.4284, test_H1=0.6750, test_L2=0.6750
[60] time=3.27, avg_loss=0.5018, train_err=20.0724, test_H1=0.6577, test_L2=0.6577
[63] time=3.40, avg_loss=0.5266, train_err=21.0622, test_H1=0.6677, test_L2=0.6677
[66] time=3.41, avg_loss=0.4440, train_err=17.7618, test_H1=0.5494, test_L2=0.5494
[69] time=3.26, avg_loss=0.4402, train_err=17.6081, test_H1=0.5148, test_L2=0.5148
[72] time=3.39, avg_loss=0.4405, train_err=17.6204, test_H1=0.5342, test_L2=0.5342
[75] time=3.26, avg_loss=0.3936, train_err=15.7455, test_H1=0.4826, test_L2=0.4826
[78] time=3.30, avg_loss=0.3565, train_err=14.2591, test_H1=0.4982, test_L2=0.4982
[81] time=3.40, avg_loss=0.4197, train_err=16.7884, test_H1=0.5272, test_L2=0.5272
[84] time=3.27, avg_loss=0.3349, train_err=13.3971, test_H1=0.4474, test_L2=0.4474
[87] time=3.40, avg_loss=0.3574, train_err=14.2961, test_H1=0.4546, test_L2=0.4546
[90] time=3.40, avg_loss=0.3191, train_err=12.7636, test_H1=0.4288, test_L2=0.4288
[93] time=3.26, avg_loss=0.3186, train_err=12.7434, test_H1=0.4553, test_L2=0.4553
[96] time=3.40, avg_loss=0.3118, train_err=12.4729, test_H1=0.4175, test_L2=0.4175
[99] time=3.37, avg_loss=0.3544, train_err=14.1773, test_H1=0.4809, test_L2=0.4809
[102] time=3.25, avg_loss=0.3044, train_err=12.1767, test_H1=0.4309, test_L2=0.4309
[105] time=3.39, avg_loss=0.3014, train_err=12.0541, test_H1=0.3958, test_L2=0.3958
[108] time=3.27, avg_loss=0.2947, train_err=11.7870, test_H1=0.3904, test_L2=0.3904
[111] time=3.37, avg_loss=0.3359, train_err=13.4351, test_H1=0.4316, test_L2=0.4316
[114] time=3.40, avg_loss=0.3032, train_err=12.1265, test_H1=0.4297, test_L2=0.4297
[117] time=3.26, avg_loss=0.2881, train_err=11.5241, test_H1=0.3886, test_L2=0.3886
[120] time=3.39, avg_loss=0.2903, train_err=11.6122, test_H1=0.4330, test_L2=0.4330
[123] time=3.40, avg_loss=0.2790, train_err=11.1585, test_H1=0.4033, test_L2=0.4033
[126] time=3.26, avg_loss=0.3245, train_err=12.9793, test_H1=0.3998, test_L2=0.3998
[129] time=3.40, avg_loss=0.2764, train_err=11.0551, test_H1=0.3816, test_L2=0.3816
[132] time=3.30, avg_loss=0.2819, train_err=11.2753, test_H1=0.3970, test_L2=0.3970
[135] time=3.26, avg_loss=0.2824, train_err=11.2978, test_H1=0.3730, test_L2=0.3730
[138] time=3.37, avg_loss=0.2784, train_err=11.1370, test_H1=0.3733, test_L2=0.3733
[141] time=3.25, avg_loss=0.2733, train_err=10.9329, test_H1=0.3883, test_L2=0.3883
[144] time=3.39, avg_loss=0.2823, train_err=11.2909, test_H1=0.3853, test_L2=0.3853
[147] time=3.40, avg_loss=0.2803, train_err=11.2117, test_H1=0.3874, test_L2=0.3874
[150] time=3.26, avg_loss=0.2824, train_err=11.2964, test_H1=0.3737, test_L2=0.3737
[153] time=3.40, avg_loss=0.2505, train_err=10.0199, test_H1=0.3545, test_L2=0.3545
[156] time=3.40, avg_loss=0.2493, train_err=9.9720, test_H1=0.3483, test_L2=0.3483
[159] time=3.27, avg_loss=0.2443, train_err=9.7704, test_H1=0.3404, test_L2=0.3404
[162] time=3.40, avg_loss=0.2505, train_err=10.0206, test_H1=0.3764, test_L2=0.3764
[165] time=3.26, avg_loss=0.2718, train_err=10.8706, test_H1=0.3694, test_L2=0.3694
[168] time=3.30, avg_loss=0.2387, train_err=9.5471, test_H1=0.3426, test_L2=0.3426
[171] time=3.39, avg_loss=0.2334, train_err=9.3357, test_H1=0.3323, test_L2=0.3323
[174] time=2.49, avg_loss=0.2507, train_err=10.0282, test_H1=0.3452, test_L2=0.3452
[177] time=1.55, avg_loss=0.2395, train_err=9.5796, test_H1=0.3527, test_L2=0.3527
[180] time=1.55, avg_loss=0.2219, train_err=8.8747, test_H1=0.3279, test_L2=0.3279
[183] time=1.55, avg_loss=0.2241, train_err=8.9631, test_H1=0.3221, test_L2=0.3221
[186] time=2.97, avg_loss=0.2360, train_err=9.4388, test_H1=0.3345, test_L2=0.3345
[189] time=3.41, avg_loss=0.2254, train_err=9.0152, test_H1=0.3300, test_L2=0.3300
[192] time=3.27, avg_loss=0.2251, train_err=9.0036, test_H1=0.3289, test_L2=0.3289
[195] time=3.27, avg_loss=0.2303, train_err=9.2110, test_H1=0.3328, test_L2=0.3328
[198] time=3.40, avg_loss=0.2376, train_err=9.5045, test_H1=0.3382, test_L2=0.3382
[201] time=3.27, avg_loss=0.2172, train_err=8.6864, test_H1=0.3203, test_L2=0.3203
[204] time=3.40, avg_loss=0.2173, train_err=8.6924, test_H1=0.3264, test_L2=0.3264
[207] time=3.40, avg_loss=0.2170, train_err=8.6809, test_H1=0.3227, test_L2=0.3227
[210] time=3.26, avg_loss=0.2146, train_err=8.5846, test_H1=0.3282, test_L2=0.3282
[213] time=3.39, avg_loss=0.2094, train_err=8.3773, test_H1=0.3177, test_L2=0.3177
[216] time=3.32, avg_loss=0.2375, train_err=9.4984, test_H1=0.3954, test_L2=0.3954
[219] time=3.27, avg_loss=0.2116, train_err=8.4635, test_H1=0.3040, test_L2=0.3040
[222] time=3.40, avg_loss=0.2072, train_err=8.2865, test_H1=0.3037, test_L2=0.3037
[225] time=3.26, avg_loss=0.2157, train_err=8.6291, test_H1=0.3027, test_L2=0.3027
[228] time=3.40, avg_loss=0.2123, train_err=8.4934, test_H1=0.3102, test_L2=0.3102
[231] time=3.41, avg_loss=0.2110, train_err=8.4388, test_H1=0.3021, test_L2=0.3021
[234] time=3.27, avg_loss=0.1932, train_err=7.7279, test_H1=0.3011, test_L2=0.3011
[237] time=3.40, avg_loss=0.2516, train_err=10.0635, test_H1=0.3452, test_L2=0.3452
[240] time=3.35, avg_loss=0.2210, train_err=8.8392, test_H1=0.3196, test_L2=0.3196
[243] time=3.26, avg_loss=0.1942, train_err=7.7685, test_H1=0.2951, test_L2=0.2951
[246] time=3.39, avg_loss=0.1856, train_err=7.4238, test_H1=0.2972, test_L2=0.2972
[249] time=3.25, avg_loss=0.1844, train_err=7.3775, test_H1=0.2859, test_L2=0.2859
[252] time=3.40, avg_loss=0.1998, train_err=7.9914, test_H1=0.3390, test_L2=0.3390
[255] time=3.40, avg_loss=0.1813, train_err=7.2537, test_H1=0.2875, test_L2=0.2875
[258] time=3.25, avg_loss=0.1807, train_err=7.2270, test_H1=0.2817, test_L2=0.2817
[261] time=3.41, avg_loss=0.1887, train_err=7.5473, test_H1=0.2959, test_L2=0.2959
[264] time=3.41, avg_loss=0.1851, train_err=7.4050, test_H1=0.2988, test_L2=0.2988
[267] time=3.27, avg_loss=0.1823, train_err=7.2930, test_H1=0.2830, test_L2=0.2830
[270] time=3.41, avg_loss=0.1808, train_err=7.2331, test_H1=0.2881, test_L2=0.2881
[273] time=3.26, avg_loss=0.1875, train_err=7.4982, test_H1=0.2933, test_L2=0.2933
[276] time=3.27, avg_loss=0.1815, train_err=7.2599, test_H1=0.2783, test_L2=0.2783
[279] time=3.39, avg_loss=0.1826, train_err=7.3054, test_H1=0.3217, test_L2=0.3217
[282] time=3.25, avg_loss=0.1781, train_err=7.1224, test_H1=0.2834, test_L2=0.2834
[285] time=3.40, avg_loss=0.1950, train_err=7.8018, test_H1=0.2979, test_L2=0.2979
[288] time=3.40, avg_loss=0.1838, train_err=7.3520, test_H1=0.2835, test_L2=0.2835
[291] time=3.26, avg_loss=0.1979, train_err=7.9166, test_H1=0.2842, test_L2=0.2842
[294] time=3.40, avg_loss=0.1770, train_err=7.0796, test_H1=0.2786, test_L2=0.2786
[297] time=3.33, avg_loss=0.1762, train_err=7.0480, test_H1=0.2855, test_L2=0.2855
[300] time=3.26, avg_loss=0.1639, train_err=6.5562, test_H1=0.2667, test_L2=0.2667
[303] time=3.41, avg_loss=0.1597, train_err=6.3883, test_H1=0.2672, test_L2=0.2672
[306] time=3.26, avg_loss=0.1628, train_err=6.5139, test_H1=0.2679, test_L2=0.2679
[309] time=3.34, avg_loss=0.1855, train_err=7.4185, test_H1=0.2905, test_L2=0.2905
[312] time=3.41, avg_loss=0.1702, train_err=6.8068, test_H1=0.2766, test_L2=0.2766
[315] time=3.25, avg_loss=0.1671, train_err=6.6851, test_H1=0.2654, test_L2=0.2654
[318] time=3.39, avg_loss=0.1578, train_err=6.3110, test_H1=0.2655, test_L2=0.2655
[321] time=3.40, avg_loss=0.1604, train_err=6.4159, test_H1=0.2784, test_L2=0.2784
[324] time=3.27, avg_loss=0.1644, train_err=6.5744, test_H1=0.2636, test_L2=0.2636
[327] time=3.41, avg_loss=0.1566, train_err=6.2637, test_H1=0.2653, test_L2=0.2653
[330] time=3.27, avg_loss=0.1625, train_err=6.4993, test_H1=0.2678, test_L2=0.2678
[333] time=3.33, avg_loss=0.1528, train_err=6.1119, test_H1=0.2688, test_L2=0.2688
[336] time=3.40, avg_loss=0.1975, train_err=7.8989, test_H1=0.3313, test_L2=0.3313
[339] time=3.27, avg_loss=0.1558, train_err=6.2316, test_H1=0.2680, test_L2=0.2680
[342] time=3.40, avg_loss=0.1657, train_err=6.6296, test_H1=0.2666, test_L2=0.2666
[345] time=3.41, avg_loss=0.1613, train_err=6.4522, test_H1=0.2617, test_L2=0.2617
[348] time=3.27, avg_loss=0.1584, train_err=6.3364, test_H1=0.2631, test_L2=0.2631
[351] time=3.39, avg_loss=0.1675, train_err=6.6990, test_H1=0.2673, test_L2=0.2673
[354] time=3.26, avg_loss=0.1500, train_err=5.9990, test_H1=0.2885, test_L2=0.2885
[357] time=3.38, avg_loss=0.1449, train_err=5.7942, test_H1=0.2554, test_L2=0.2554
[360] time=3.41, avg_loss=0.1484, train_err=5.9371, test_H1=0.2606, test_L2=0.2606
[363] time=3.27, avg_loss=0.1569, train_err=6.2749, test_H1=0.2601, test_L2=0.2601
[366] time=3.40, avg_loss=0.1632, train_err=6.5277, test_H1=0.2777, test_L2=0.2777
[369] time=3.38, avg_loss=0.1472, train_err=5.8888, test_H1=0.2679, test_L2=0.2679
[372] time=3.27, avg_loss=0.1482, train_err=5.9275, test_H1=0.2650, test_L2=0.2650
[375] time=3.40, avg_loss=0.1524, train_err=6.0963, test_H1=0.2608, test_L2=0.2608
[378] time=3.26, avg_loss=0.1443, train_err=5.7726, test_H1=0.2564, test_L2=0.2564
[381] time=3.33, avg_loss=0.1712, train_err=6.8474, test_H1=0.3336, test_L2=0.3336
[384] time=3.40, avg_loss=0.1571, train_err=6.2853, test_H1=0.2820, test_L2=0.2820
[387] time=3.26, avg_loss=0.1480, train_err=5.9181, test_H1=0.2607, test_L2=0.2607
[390] time=3.40, avg_loss=0.1506, train_err=6.0224, test_H1=0.2550, test_L2=0.2550
[393] time=3.40, avg_loss=0.1410, train_err=5.6381, test_H1=0.2524, test_L2=0.2524
[396] time=3.26, avg_loss=0.1433, train_err=5.7338, test_H1=0.2506, test_L2=0.2506
[399] time=3.41, avg_loss=0.1442, train_err=5.7665, test_H1=0.2530, test_L2=0.2530
[402] time=3.28, avg_loss=0.1388, train_err=5.5521, test_H1=0.2502, test_L2=0.2502
[405] time=3.34, avg_loss=0.1456, train_err=5.8253, test_H1=0.2614, test_L2=0.2614
[408] time=3.40, avg_loss=0.1430, train_err=5.7196, test_H1=0.2555, test_L2=0.2555
[411] time=3.26, avg_loss=0.1427, train_err=5.7090, test_H1=0.2497, test_L2=0.2497
[414] time=3.40, avg_loss=0.1358, train_err=5.4337, test_H1=0.2470, test_L2=0.2470
[417] time=3.41, avg_loss=0.1357, train_err=5.4273, test_H1=0.2530, test_L2=0.2530
[420] time=3.26, avg_loss=0.1369, train_err=5.4745, test_H1=0.2510, test_L2=0.2510
[423] time=3.39, avg_loss=0.1389, train_err=5.5560, test_H1=0.2490, test_L2=0.2490
[426] time=3.26, avg_loss=0.1781, train_err=7.1232, test_H1=0.2660, test_L2=0.2660
[429] time=3.33, avg_loss=0.1498, train_err=5.9927, test_H1=0.2613, test_L2=0.2613
[432] time=3.40, avg_loss=0.1326, train_err=5.3042, test_H1=0.2465, test_L2=0.2465
[435] time=3.25, avg_loss=0.1327, train_err=5.3089, test_H1=0.2434, test_L2=0.2434
[438] time=3.40, avg_loss=0.1363, train_err=5.4522, test_H1=0.2539, test_L2=0.2539
[441] time=3.40, avg_loss=0.1371, train_err=5.4831, test_H1=0.2509, test_L2=0.2509
[444] time=3.26, avg_loss=0.1363, train_err=5.4515, test_H1=0.2418, test_L2=0.2418
[447] time=3.41, avg_loss=0.1371, train_err=5.4843, test_H1=0.2461, test_L2=0.2461
[450] time=3.29, avg_loss=0.1328, train_err=5.3103, test_H1=0.2444, test_L2=0.2444
[453] time=3.25, avg_loss=0.1282, train_err=5.1286, test_H1=0.2488, test_L2=0.2488
[456] time=3.39, avg_loss=0.1260, train_err=5.0419, test_H1=0.2398, test_L2=0.2398
[459] time=3.25, avg_loss=0.1307, train_err=5.2289, test_H1=0.2419, test_L2=0.2419
[462] time=3.39, avg_loss=0.1355, train_err=5.4215, test_H1=0.2626, test_L2=0.2626
[465] time=3.41, avg_loss=0.1605, train_err=6.4214, test_H1=0.2804, test_L2=0.2804
[468] time=3.27, avg_loss=0.1303, train_err=5.2119, test_H1=0.2429, test_L2=0.2429
[471] time=3.41, avg_loss=0.1230, train_err=4.9207, test_H1=0.2431, test_L2=0.2431
[474] time=3.34, avg_loss=0.1244, train_err=4.9745, test_H1=0.2438, test_L2=0.2438
[477] time=3.26, avg_loss=0.1277, train_err=5.1083, test_H1=0.2460, test_L2=0.2460
[480] time=3.41, avg_loss=0.1286, train_err=5.1445, test_H1=0.2549, test_L2=0.2549
[483] time=3.27, avg_loss=0.1284, train_err=5.1340, test_H1=0.2455, test_L2=0.2455
[486] time=3.35, avg_loss=0.1283, train_err=5.1301, test_H1=0.2371, test_L2=0.2371
[489] time=3.39, avg_loss=0.1222, train_err=4.8865, test_H1=0.2357, test_L2=0.2357
[492] time=3.25, avg_loss=0.1345, train_err=5.3815, test_H1=0.2406, test_L2=0.2406
[495] time=3.39, avg_loss=0.1335, train_err=5.3410, test_H1=0.2489, test_L2=0.2489
[498] time=3.39, avg_loss=0.1305, train_err=5.2208, test_H1=0.2501, test_L2=0.2501
[501] time=3.25, avg_loss=0.1221, train_err=4.8841, test_H1=0.2326, test_L2=0.2326
[504] time=3.40, avg_loss=0.1195, train_err=4.7809, test_H1=0.2329, test_L2=0.2329
[507] time=3.28, avg_loss=0.1227, train_err=4.9089, test_H1=0.2661, test_L2=0.2661
[510] time=3.32, avg_loss=0.1183, train_err=4.7336, test_H1=0.2349, test_L2=0.2349
[513] time=3.41, avg_loss=0.1216, train_err=4.8650, test_H1=0.2353, test_L2=0.2353
[516] time=3.28, avg_loss=0.1201, train_err=4.8034, test_H1=0.2348, test_L2=0.2348
[519] time=3.41, avg_loss=0.1198, train_err=4.7910, test_H1=0.2341, test_L2=0.2341
[522] time=3.40, avg_loss=0.1156, train_err=4.6259, test_H1=0.2302, test_L2=0.2302
[525] time=3.26, avg_loss=0.1200, train_err=4.7986, test_H1=0.2362, test_L2=0.2362
[528] time=3.40, avg_loss=0.1197, train_err=4.7890, test_H1=0.2369, test_L2=0.2369
[531] time=3.26, avg_loss=0.1266, train_err=5.0648, test_H1=0.2490, test_L2=0.2490
[534] time=3.31, avg_loss=0.1199, train_err=4.7953, test_H1=0.2335, test_L2=0.2335
[537] time=3.41, avg_loss=0.1378, train_err=5.5129, test_H1=0.2507, test_L2=0.2507
[540] time=3.25, avg_loss=0.1134, train_err=4.5343, test_H1=0.2283, test_L2=0.2283
[543] time=3.41, avg_loss=0.1142, train_err=4.5673, test_H1=0.2294, test_L2=0.2294
[546] time=3.40, avg_loss=0.1183, train_err=4.7321, test_H1=0.2304, test_L2=0.2304
[549] time=3.27, avg_loss=0.1159, train_err=4.6365, test_H1=0.2361, test_L2=0.2361
[552] time=3.40, avg_loss=0.1126, train_err=4.5041, test_H1=0.2282, test_L2=0.2282
[555] time=3.32, avg_loss=0.1114, train_err=4.4541, test_H1=0.2334, test_L2=0.2334
[558] time=3.26, avg_loss=0.1074, train_err=4.2943, test_H1=0.2268, test_L2=0.2268
[561] time=3.39, avg_loss=0.1081, train_err=4.3246, test_H1=0.2253, test_L2=0.2253
[564] time=3.26, avg_loss=0.1236, train_err=4.9420, test_H1=0.2296, test_L2=0.2296
[567] time=3.40, avg_loss=0.1095, train_err=4.3786, test_H1=0.2296, test_L2=0.2296
[570] time=3.40, avg_loss=0.1080, train_err=4.3200, test_H1=0.2224, test_L2=0.2224
[573] time=3.27, avg_loss=0.1150, train_err=4.5997, test_H1=0.2272, test_L2=0.2272
[576] time=3.40, avg_loss=0.1092, train_err=4.3664, test_H1=0.2266, test_L2=0.2266
[579] time=3.34, avg_loss=0.1104, train_err=4.4143, test_H1=0.2290, test_L2=0.2290
[582] time=3.26, avg_loss=0.1074, train_err=4.2961, test_H1=0.2302, test_L2=0.2302
[585] time=3.40, avg_loss=0.1114, train_err=4.4560, test_H1=0.2503, test_L2=0.2503
[588] time=3.26, avg_loss=0.1110, train_err=4.4415, test_H1=0.2271, test_L2=0.2271
[591] time=3.35, avg_loss=0.1060, train_err=4.2400, test_H1=0.2245, test_L2=0.2245
[594] time=3.40, avg_loss=0.1114, train_err=4.4556, test_H1=0.2256, test_L2=0.2256
[597] time=3.25, avg_loss=0.1059, train_err=4.2366, test_H1=0.2248, test_L2=0.2248
[600] time=3.40, avg_loss=0.1036, train_err=4.1432, test_H1=0.2207, test_L2=0.2207
[603] time=3.40, avg_loss=0.1044, train_err=4.1745, test_H1=0.2252, test_L2=0.2252
[606] time=3.27, avg_loss=0.1004, train_err=4.0171, test_H1=0.2238, test_L2=0.2238
[609] time=3.40, avg_loss=0.1013, train_err=4.0514, test_H1=0.2220, test_L2=0.2220
[612] time=3.25, avg_loss=0.1038, train_err=4.1519, test_H1=0.2273, test_L2=0.2273
[615] time=3.32, avg_loss=0.1076, train_err=4.3038, test_H1=0.2209, test_L2=0.2209
[618] time=3.40, avg_loss=0.1032, train_err=4.1267, test_H1=0.2258, test_L2=0.2258
[621] time=3.26, avg_loss=0.1034, train_err=4.1374, test_H1=0.2291, test_L2=0.2291
[624] time=3.41, avg_loss=0.1004, train_err=4.0158, test_H1=0.2223, test_L2=0.2223
[627] time=3.40, avg_loss=0.1078, train_err=4.3114, test_H1=0.2241, test_L2=0.2241
[630] time=3.25, avg_loss=0.1085, train_err=4.3400, test_H1=0.2301, test_L2=0.2301
[633] time=3.40, avg_loss=0.1004, train_err=4.0176, test_H1=0.2176, test_L2=0.2176
[636] time=3.29, avg_loss=0.1045, train_err=4.1802, test_H1=0.2226, test_L2=0.2226
[639] time=3.29, avg_loss=0.1081, train_err=4.3242, test_H1=0.2225, test_L2=0.2225
[642] time=3.41, avg_loss=0.1165, train_err=4.6607, test_H1=0.2769, test_L2=0.2769
[645] time=3.28, avg_loss=0.1041, train_err=4.1626, test_H1=0.2205, test_L2=0.2205
[648] time=3.39, avg_loss=0.0972, train_err=3.8891, test_H1=0.2174, test_L2=0.2174
[651] time=3.41, avg_loss=0.0964, train_err=3.8573, test_H1=0.2153, test_L2=0.2153
[654] time=3.27, avg_loss=0.0951, train_err=3.8023, test_H1=0.2172, test_L2=0.2172
[657] time=3.40, avg_loss=0.1009, train_err=4.0362, test_H1=0.2220, test_L2=0.2220
[660] time=3.41, avg_loss=0.0968, train_err=3.8729, test_H1=0.2173, test_L2=0.2173
[663] time=3.26, avg_loss=0.0950, train_err=3.8006, test_H1=0.2137, test_L2=0.2137
[666] time=3.39, avg_loss=0.0982, train_err=3.9277, test_H1=0.2166, test_L2=0.2166
[669] time=3.26, avg_loss=0.1005, train_err=4.0218, test_H1=0.2165, test_L2=0.2165
[672] time=3.35, avg_loss=0.0982, train_err=3.9286, test_H1=0.2164, test_L2=0.2164
[675] time=3.40, avg_loss=0.0945, train_err=3.7792, test_H1=0.2139, test_L2=0.2139
[678] time=3.25, avg_loss=0.0976, train_err=3.9053, test_H1=0.2191, test_L2=0.2191
[681] time=3.40, avg_loss=0.1138, train_err=4.5527, test_H1=0.2221, test_L2=0.2221
[684] time=3.40, avg_loss=0.0921, train_err=3.6841, test_H1=0.2125, test_L2=0.2125
[687] time=3.27, avg_loss=0.0964, train_err=3.8550, test_H1=0.2189, test_L2=0.2189
[690] time=3.40, avg_loss=0.0914, train_err=3.6570, test_H1=0.2127, test_L2=0.2127
[693] time=3.30, avg_loss=0.0956, train_err=3.8253, test_H1=0.2203, test_L2=0.2203
[696] time=3.27, avg_loss=0.0916, train_err=3.6626, test_H1=0.2112, test_L2=0.2112
[699] time=3.39, avg_loss=0.0956, train_err=3.8247, test_H1=0.2154, test_L2=0.2154
[702] time=3.25, avg_loss=0.0892, train_err=3.5690, test_H1=0.2127, test_L2=0.2127
[705] time=3.39, avg_loss=0.0909, train_err=3.6344, test_H1=0.2107, test_L2=0.2107
[708] time=3.40, avg_loss=0.0890, train_err=3.5615, test_H1=0.2114, test_L2=0.2114
[711] time=3.26, avg_loss=0.0946, train_err=3.7834, test_H1=0.2184, test_L2=0.2184
[714] time=3.40, avg_loss=0.0937, train_err=3.7488, test_H1=0.2127, test_L2=0.2127
[717] time=3.35, avg_loss=0.0886, train_err=3.5458, test_H1=0.2118, test_L2=0.2118
[720] time=3.27, avg_loss=0.0903, train_err=3.6125, test_H1=0.2140, test_L2=0.2140
[723] time=3.40, avg_loss=0.0948, train_err=3.7921, test_H1=0.2151, test_L2=0.2151
[726] time=3.27, avg_loss=0.0892, train_err=3.5682, test_H1=0.2156, test_L2=0.2156
[729] time=3.35, avg_loss=0.0919, train_err=3.6745, test_H1=0.2115, test_L2=0.2115
[732] time=3.40, avg_loss=0.0884, train_err=3.5348, test_H1=0.2103, test_L2=0.2103
[735] time=3.26, avg_loss=0.0872, train_err=3.4880, test_H1=0.2084, test_L2=0.2084
[738] time=3.40, avg_loss=0.0874, train_err=3.4960, test_H1=0.2087, test_L2=0.2087
[741] time=3.40, avg_loss=0.0876, train_err=3.5040, test_H1=0.2117, test_L2=0.2117
[744] time=3.26, avg_loss=0.0901, train_err=3.6038, test_H1=0.2109, test_L2=0.2109
[747] time=3.41, avg_loss=0.0873, train_err=3.4908, test_H1=0.2107, test_L2=0.2107
[750] time=3.26, avg_loss=0.0847, train_err=3.3875, test_H1=0.2093, test_L2=0.2093
[753] time=3.33, avg_loss=0.0838, train_err=3.3516, test_H1=0.2082, test_L2=0.2082
[756] time=3.41, avg_loss=0.0864, train_err=3.4547, test_H1=0.2089, test_L2=0.2089
[759] time=3.26, avg_loss=0.0877, train_err=3.5081, test_H1=0.2089, test_L2=0.2089
[762] time=3.41, avg_loss=0.0954, train_err=3.8171, test_H1=0.2120, test_L2=0.2120
[765] time=3.40, avg_loss=0.0858, train_err=3.4319, test_H1=0.2074, test_L2=0.2074
[768] time=3.26, avg_loss=0.0839, train_err=3.3543, test_H1=0.2110, test_L2=0.2110
[771] time=3.40, avg_loss=0.0871, train_err=3.4852, test_H1=0.2105, test_L2=0.2105
[774] time=3.28, avg_loss=0.0841, train_err=3.3658, test_H1=0.2096, test_L2=0.2096
[777] time=3.31, avg_loss=0.0836, train_err=3.3423, test_H1=0.2084, test_L2=0.2084
[780] time=3.41, avg_loss=0.0866, train_err=3.4646, test_H1=0.2132, test_L2=0.2132
[783] time=3.26, avg_loss=0.0842, train_err=3.3694, test_H1=0.2086, test_L2=0.2086
[786] time=3.40, avg_loss=0.0816, train_err=3.2645, test_H1=0.2060, test_L2=0.2060
[789] time=3.40, avg_loss=0.0846, train_err=3.3841, test_H1=0.2070, test_L2=0.2070
[792] time=3.26, avg_loss=0.0859, train_err=3.4368, test_H1=0.2070, test_L2=0.2070
[795] time=3.40, avg_loss=0.0878, train_err=3.5121, test_H1=0.2103, test_L2=0.2103
[798] time=3.34, avg_loss=0.0805, train_err=3.2208, test_H1=0.2047, test_L2=0.2047
[801] time=3.25, avg_loss=0.0810, train_err=3.2413, test_H1=0.2067, test_L2=0.2067
[804] time=3.40, avg_loss=0.0812, train_err=3.2497, test_H1=0.2087, test_L2=0.2087
[807] time=3.26, avg_loss=0.0802, train_err=3.2083, test_H1=0.2073, test_L2=0.2073
[810] time=3.40, avg_loss=0.0810, train_err=3.2383, test_H1=0.2058, test_L2=0.2058
[813] time=3.40, avg_loss=0.0874, train_err=3.4972, test_H1=0.2088, test_L2=0.2088
[816] time=3.26, avg_loss=0.0822, train_err=3.2895, test_H1=0.2072, test_L2=0.2072
[819] time=3.41, avg_loss=0.0816, train_err=3.2646, test_H1=0.2087, test_L2=0.2087
[822] time=3.37, avg_loss=0.0786, train_err=3.1432, test_H1=0.2060, test_L2=0.2060
[825] time=3.27, avg_loss=0.0843, train_err=3.3710, test_H1=0.2068, test_L2=0.2068
[828] time=3.41, avg_loss=0.0831, train_err=3.3240, test_H1=0.2162, test_L2=0.2162
[831] time=3.26, avg_loss=0.0813, train_err=3.2516, test_H1=0.2090, test_L2=0.2090
[834] time=3.33, avg_loss=0.0770, train_err=3.0808, test_H1=0.2060, test_L2=0.2060
[837] time=3.40, avg_loss=0.0815, train_err=3.2581, test_H1=0.2046, test_L2=0.2046
[840] time=3.25, avg_loss=0.0786, train_err=3.1459, test_H1=0.2038, test_L2=0.2038
[843] time=3.40, avg_loss=0.0779, train_err=3.1158, test_H1=0.2055, test_L2=0.2055
[846] time=3.40, avg_loss=0.0957, train_err=3.8293, test_H1=0.2254, test_L2=0.2254
[849] time=3.26, avg_loss=0.0779, train_err=3.1143, test_H1=0.2036, test_L2=0.2036
[852] time=3.40, avg_loss=0.0766, train_err=3.0644, test_H1=0.2049, test_L2=0.2049
[855] time=3.27, avg_loss=0.0779, train_err=3.1159, test_H1=0.2026, test_L2=0.2026
[858] time=3.28, avg_loss=0.0757, train_err=3.0285, test_H1=0.2024, test_L2=0.2024
[861] time=3.40, avg_loss=0.0777, train_err=3.1082, test_H1=0.2040, test_L2=0.2040
[864] time=3.26, avg_loss=0.0755, train_err=3.0194, test_H1=0.2036, test_L2=0.2036
[867] time=3.40, avg_loss=0.0757, train_err=3.0298, test_H1=0.2022, test_L2=0.2022
[870] time=3.40, avg_loss=0.0774, train_err=3.0942, test_H1=0.2054, test_L2=0.2054
[873] time=3.24, avg_loss=0.0768, train_err=3.0706, test_H1=0.2045, test_L2=0.2045
[876] time=3.39, avg_loss=0.0774, train_err=3.0946, test_H1=0.2030, test_L2=0.2030
[879] time=3.34, avg_loss=0.0770, train_err=3.0815, test_H1=0.2025, test_L2=0.2025
[882] time=3.26, avg_loss=0.0748, train_err=2.9909, test_H1=0.2019, test_L2=0.2019
[885] time=3.41, avg_loss=0.0751, train_err=3.0033, test_H1=0.2029, test_L2=0.2029
[888] time=3.26, avg_loss=0.0781, train_err=3.1232, test_H1=0.2035, test_L2=0.2035
[891] time=3.41, avg_loss=0.0843, train_err=3.3704, test_H1=0.2165, test_L2=0.2165
[894] time=3.41, avg_loss=0.0742, train_err=2.9685, test_H1=0.2024, test_L2=0.2024
[897] time=3.27, avg_loss=0.0737, train_err=2.9481, test_H1=0.2022, test_L2=0.2022
[900] time=3.40, avg_loss=0.0736, train_err=2.9430, test_H1=0.2025, test_L2=0.2025
[903] time=3.39, avg_loss=0.0726, train_err=2.9058, test_H1=0.2022, test_L2=0.2022
[906] time=3.26, avg_loss=0.0735, train_err=2.9389, test_H1=0.2003, test_L2=0.2003
[909] time=3.39, avg_loss=0.0734, train_err=2.9372, test_H1=0.2015, test_L2=0.2015
[912] time=3.25, avg_loss=0.0758, train_err=3.0333, test_H1=0.2044, test_L2=0.2044
[915] time=3.39, avg_loss=0.0776, train_err=3.1051, test_H1=0.2049, test_L2=0.2049
[918] time=3.41, avg_loss=0.0726, train_err=2.9047, test_H1=0.1999, test_L2=0.1999
[921] time=3.26, avg_loss=0.0714, train_err=2.8559, test_H1=0.2018, test_L2=0.2018
[924] time=3.40, avg_loss=0.0749, train_err=2.9972, test_H1=0.2016, test_L2=0.2016
[927] time=3.40, avg_loss=0.0745, train_err=2.9812, test_H1=0.2019, test_L2=0.2019
[930] time=3.26, avg_loss=0.0726, train_err=2.9050, test_H1=0.2017, test_L2=0.2017
[933] time=3.40, avg_loss=0.0739, train_err=2.9557, test_H1=0.2015, test_L2=0.2015
[936] time=3.27, avg_loss=0.0718, train_err=2.8723, test_H1=0.2013, test_L2=0.2013
[939] time=3.28, avg_loss=0.0709, train_err=2.8354, test_H1=0.2006, test_L2=0.2006
[942] time=3.39, avg_loss=0.0728, train_err=2.9118, test_H1=0.2023, test_L2=0.2023
[945] time=3.26, avg_loss=0.0756, train_err=3.0244, test_H1=0.2021, test_L2=0.2021
[948] time=3.40, avg_loss=0.0731, train_err=2.9235, test_H1=0.2036, test_L2=0.2036
[951] time=3.40, avg_loss=0.0713, train_err=2.8526, test_H1=0.2003, test_L2=0.2003
[954] time=3.26, avg_loss=0.0695, train_err=2.7797, test_H1=0.2003, test_L2=0.2003
[957] time=3.39, avg_loss=0.0701, train_err=2.8030, test_H1=0.2020, test_L2=0.2020
[960] time=3.29, avg_loss=0.0697, train_err=2.7886, test_H1=0.2027, test_L2=0.2027
[963] time=3.27, avg_loss=0.0685, train_err=2.7417, test_H1=0.1994, test_L2=0.1994
[966] time=3.41, avg_loss=0.0720, train_err=2.8799, test_H1=0.2023, test_L2=0.2023
[969] time=3.26, avg_loss=0.0697, train_err=2.7882, test_H1=0.2007, test_L2=0.2007
[972] time=3.40, avg_loss=0.0691, train_err=2.7633, test_H1=0.2004, test_L2=0.2004
[975] time=3.40, avg_loss=0.0735, train_err=2.9397, test_H1=0.2032, test_L2=0.2032
[978] time=3.25, avg_loss=0.0726, train_err=2.9026, test_H1=0.2002, test_L2=0.2002
[981] time=3.40, avg_loss=0.0702, train_err=2.8066, test_H1=0.1994, test_L2=0.1994
[984] time=3.38, avg_loss=0.0706, train_err=2.8242, test_H1=0.2014, test_L2=0.2014
[987] time=3.26, avg_loss=0.0703, train_err=2.8111, test_H1=0.2026, test_L2=0.2026
[990] time=3.41, avg_loss=0.0693, train_err=2.7701, test_H1=0.2000, test_L2=0.2000
[993] time=3.26, avg_loss=0.0701, train_err=2.8041, test_H1=0.2016, test_L2=0.2016
[996] time=3.38, avg_loss=0.0676, train_err=2.7032, test_H1=0.1995, test_L2=0.1995
[999] time=3.41, avg_loss=0.0699, train_err=2.7964, test_H1=0.2008, test_L2=0.2008
(1000, 4, 2048) (1000, 1, 2048)
Total number of samples: 1000
Input data shape: (1000, 4, 2048)
Output series shape: (1000, 1, 2048)
Batch input series shape: torch.Size([32, 4, 2048])
Batch output series shape: torch.Size([32, 1, 2048])
Dtype torch.complex64 torch.complex64

Our model has 33919746 parameters.
torch.Size([8, 128, 1]) torch.Size([128, 128, 128, 2]) torch.Size([128, 128, 128, 2]) torch.Size([128, 128, 128, 2])

### MODEL ###
 FNO(
  (positional_embedding): GridEmbeddingND()
  (fno_blocks): FNOBlocks(
    (convs): SpectralConv(
      (weight): ModuleList(
        (0-7): 8 x ComplexDenseTensor(shape=torch.Size([128, 128, 128]), rank=None)
      )
    )
    (fno_skips): ModuleList(
      (0-7): 8 x ComplexValued(
        (fr): Conv1d(128, 128, kernel_size=(1,), stride=(1,), bias=False)
        (fi): Conv1d(128, 128, kernel_size=(1,), stride=(1,), bias=False)
      )
    )
  )
  (lifting): ComplexValued(
    (fr): ChannelMLP(
      (fcs): ModuleList(
        (0): Conv1d(5, 256, kernel_size=(1,), stride=(1,))
        (1): Conv1d(256, 128, kernel_size=(1,), stride=(1,))
      )
    )
    (fi): ChannelMLP(
      (fcs): ModuleList(
        (0): Conv1d(5, 256, kernel_size=(1,), stride=(1,))
        (1): Conv1d(256, 128, kernel_size=(1,), stride=(1,))
      )
    )
  )
  (projection): ComplexValued(
    (fr): ChannelMLP(
      (fcs): ModuleList(
        (0): Conv1d(128, 128, kernel_size=(1,), stride=(1,))
        (1): Conv1d(128, 1, kernel_size=(1,), stride=(1,))
      )
    )
    (fi): ChannelMLP(
      (fcs): ModuleList(
        (0): Conv1d(128, 128, kernel_size=(1,), stride=(1,))
        (1): Conv1d(128, 1, kernel_size=(1,), stride=(1,))
      )
    )
  )
)

### OPTIMIZER ###
 AdamW (
Parameter Group 0
    betas: (0.9, 0.999)
    correct_bias: True
    eps: 1e-06
    initial_lr: 0.001
    lr: 0.001
    weight_decay: 2e-06

Parameter Group 1
    betas: (0.9, 0.999)
    correct_bias: True
    dim: 5
    eps: 1e-06
    initial_lr: 0.001
    lr: 0.001
    proj_type: std
    rank: 0.25
    scale: 1.0
    type: tucker
    update_proj_gap: 1
    weight_decay: 2e-06
)

### SCHEDULER ###
 <torch.optim.lr_scheduler.StepLR object at 0x7f3ed5cfd750>

### LOSSES ###

 * Train: <neuralop.losses.data_losses.H1Loss object at 0x7f3ed5cfd1b0>

 * Test: {'H1': <neuralop.losses.data_losses.H1Loss object at 0x7f3ed5cfd1b0>, 'L2': <neuralop.losses.data_losses.LpLoss object at 0x7f3ed5cfd0f0>}
using standard method to load data to device.
using standard method to compute loss.
self.override_load_to_device=False
self.overrides_loss=False
Training on 800 samples
Testing on [200] samples         on resolutions ['test'].
Raw outputs of size out.shape=torch.Size([32, 1, 2048])
[0] time=9.10, avg_loss=0.7996, train_err=31.9850, test_H1=0.9971, test_L2=0.9971
[3] time=3.47, avg_loss=0.7979, train_err=31.9176, test_H1=0.9940, test_L2=0.9940
[6] time=3.44, avg_loss=0.7956, train_err=31.8245, test_H1=1.0013, test_L2=1.0013
[9] time=3.30, avg_loss=0.7895, train_err=31.5793, test_H1=0.9686, test_L2=0.9686
[12] time=3.46, avg_loss=0.7739, train_err=30.9541, test_H1=0.9675, test_L2=0.9675
[15] time=3.31, avg_loss=0.7820, train_err=31.2782, test_H1=0.9721, test_L2=0.9721
[18] time=3.46, avg_loss=0.7760, train_err=31.0409, test_H1=0.9749, test_L2=0.9749
[21] time=3.47, avg_loss=0.7576, train_err=30.3052, test_H1=0.9502, test_L2=0.9502
[24] time=3.30, avg_loss=0.7392, train_err=29.5666, test_H1=0.9397, test_L2=0.9397
[27] time=3.46, avg_loss=0.7352, train_err=29.4098, test_H1=0.9305, test_L2=0.9305
[30] time=3.31, avg_loss=0.7402, train_err=29.6079, test_H1=0.9343, test_L2=0.9343
[33] time=3.47, avg_loss=0.7136, train_err=28.5440, test_H1=0.9108, test_L2=0.9108
[36] time=3.46, avg_loss=0.7222, train_err=28.8871, test_H1=0.8967, test_L2=0.8967
[39] time=3.31, avg_loss=0.7007, train_err=28.0260, test_H1=0.8934, test_L2=0.8934
[42] time=3.46, avg_loss=0.6914, train_err=27.6554, test_H1=0.8760, test_L2=0.8760
[45] time=3.31, avg_loss=0.6909, train_err=27.6357, test_H1=0.9111, test_L2=0.9111
[48] time=3.47, avg_loss=0.6822, train_err=27.2886, test_H1=0.8959, test_L2=0.8959
[51] time=3.45, avg_loss=0.6427, train_err=25.7082, test_H1=0.8549, test_L2=0.8549
[54] time=3.29, avg_loss=0.6610, train_err=26.4418, test_H1=0.8503, test_L2=0.8503
[57] time=3.45, avg_loss=0.6122, train_err=24.4883, test_H1=0.7830, test_L2=0.7830
[60] time=3.29, avg_loss=0.5308, train_err=21.2313, test_H1=0.7215, test_L2=0.7215
[63] time=3.45, avg_loss=0.4902, train_err=19.6100, test_H1=0.5904, test_L2=0.5904
[66] time=3.45, avg_loss=0.4455, train_err=17.8190, test_H1=0.7063, test_L2=0.7063
[69] time=3.33, avg_loss=0.4458, train_err=17.8306, test_H1=0.5274, test_L2=0.5274
[72] time=3.46, avg_loss=0.3780, train_err=15.1204, test_H1=0.4780, test_L2=0.4780
[75] time=3.30, avg_loss=0.3639, train_err=14.5574, test_H1=0.6006, test_L2=0.6006
[78] time=3.46, avg_loss=0.4047, train_err=16.1861, test_H1=0.5338, test_L2=0.5338
[81] time=3.45, avg_loss=0.3382, train_err=13.5272, test_H1=0.4461, test_L2=0.4461
[84] time=3.32, avg_loss=0.3393, train_err=13.5712, test_H1=0.4372, test_L2=0.4372
[87] time=3.47, avg_loss=0.3220, train_err=12.8789, test_H1=0.4736, test_L2=0.4736
[90] time=3.31, avg_loss=0.3560, train_err=14.2416, test_H1=0.5020, test_L2=0.5020
[93] time=3.45, avg_loss=0.3494, train_err=13.9778, test_H1=0.5158, test_L2=0.5158
[96] time=3.46, avg_loss=0.3364, train_err=13.4575, test_H1=0.4360, test_L2=0.4360
[99] time=3.30, avg_loss=0.3158, train_err=12.6313, test_H1=0.4153, test_L2=0.4153
[102] time=3.44, avg_loss=0.3238, train_err=12.9515, test_H1=0.4164, test_L2=0.4164
[105] time=3.31, avg_loss=0.2918, train_err=11.6703, test_H1=0.3906, test_L2=0.3906
[108] time=3.46, avg_loss=0.3009, train_err=12.0366, test_H1=0.4118, test_L2=0.4118
[111] time=3.46, avg_loss=0.3101, train_err=12.4049, test_H1=0.3987, test_L2=0.3987
[114] time=3.31, avg_loss=0.3212, train_err=12.8497, test_H1=0.4112, test_L2=0.4112
[117] time=3.46, avg_loss=0.3087, train_err=12.3500, test_H1=0.4074, test_L2=0.4074
[120] time=3.33, avg_loss=0.2834, train_err=11.3347, test_H1=0.3786, test_L2=0.3786
[123] time=3.45, avg_loss=0.2745, train_err=10.9789, test_H1=0.3898, test_L2=0.3898
[126] time=3.45, avg_loss=0.2884, train_err=11.5357, test_H1=0.3897, test_L2=0.3897
[129] time=3.31, avg_loss=0.2734, train_err=10.9371, test_H1=0.3785, test_L2=0.3785
[132] time=3.45, avg_loss=0.2616, train_err=10.4644, test_H1=0.3928, test_L2=0.3928
[135] time=3.32, avg_loss=0.2593, train_err=10.3713, test_H1=0.3733, test_L2=0.3733
[138] time=3.42, avg_loss=0.2602, train_err=10.4093, test_H1=0.3503, test_L2=0.3503
[141] time=3.46, avg_loss=0.2441, train_err=9.7638, test_H1=0.3750, test_L2=0.3750
[144] time=3.30, avg_loss=0.2521, train_err=10.0841, test_H1=0.3637, test_L2=0.3637
[147] time=3.45, avg_loss=0.2435, train_err=9.7386, test_H1=0.3317, test_L2=0.3317
[150] time=3.29, avg_loss=0.2400, train_err=9.6015, test_H1=0.3379, test_L2=0.3379
[153] time=3.44, avg_loss=0.2233, train_err=8.9335, test_H1=0.3583, test_L2=0.3583
[156] time=3.44, avg_loss=0.2150, train_err=8.6004, test_H1=0.3210, test_L2=0.3210
[159] time=3.31, avg_loss=0.2636, train_err=10.5444, test_H1=0.3941, test_L2=0.3941
[162] time=3.45, avg_loss=0.2190, train_err=8.7588, test_H1=0.3802, test_L2=0.3802
[165] time=3.31, avg_loss=0.2241, train_err=8.9640, test_H1=0.3379, test_L2=0.3379
[168] time=3.41, avg_loss=0.2241, train_err=8.9639, test_H1=0.3317, test_L2=0.3317
[171] time=3.45, avg_loss=0.2393, train_err=9.5735, test_H1=0.3362, test_L2=0.3362
[174] time=3.31, avg_loss=0.2063, train_err=8.2501, test_H1=0.2991, test_L2=0.2991
[177] time=3.45, avg_loss=0.2268, train_err=9.0714, test_H1=0.3183, test_L2=0.3183
[180] time=3.32, avg_loss=0.1950, train_err=7.8003, test_H1=0.2965, test_L2=0.2965
[183] time=3.41, avg_loss=0.2226, train_err=8.9060, test_H1=0.4410, test_L2=0.4410
[186] time=3.45, avg_loss=0.1991, train_err=7.9622, test_H1=0.2934, test_L2=0.2934
[189] time=3.31, avg_loss=0.2003, train_err=8.0101, test_H1=0.3084, test_L2=0.3084
[192] time=3.44, avg_loss=0.1989, train_err=7.9566, test_H1=0.3013, test_L2=0.3013
[195] time=3.30, avg_loss=0.2094, train_err=8.3778, test_H1=0.2910, test_L2=0.2910
[198] time=3.45, avg_loss=0.1880, train_err=7.5182, test_H1=0.2899, test_L2=0.2899
[201] time=3.47, avg_loss=0.1951, train_err=7.8020, test_H1=0.3112, test_L2=0.3112
[204] time=3.33, avg_loss=0.1901, train_err=7.6034, test_H1=0.2954, test_L2=0.2954
[207] time=3.48, avg_loss=0.1842, train_err=7.3665, test_H1=0.2885, test_L2=0.2885
[210] time=3.33, avg_loss=0.1804, train_err=7.2164, test_H1=0.2875, test_L2=0.2875
[213] time=3.44, avg_loss=0.2497, train_err=9.9864, test_H1=0.3151, test_L2=0.3151
[216] time=3.46, avg_loss=0.1867, train_err=7.4674, test_H1=0.2852, test_L2=0.2852
[219] time=3.30, avg_loss=0.1814, train_err=7.2559, test_H1=0.2907, test_L2=0.2907
[222] time=3.46, avg_loss=0.2257, train_err=9.0294, test_H1=0.2959, test_L2=0.2959
[225] time=3.32, avg_loss=0.1753, train_err=7.0102, test_H1=0.2763, test_L2=0.2763
[228] time=3.41, avg_loss=0.1723, train_err=6.8936, test_H1=0.2769, test_L2=0.2769
[231] time=3.45, avg_loss=0.1758, train_err=7.0339, test_H1=0.2758, test_L2=0.2758
[234] time=3.30, avg_loss=0.1792, train_err=7.1682, test_H1=0.2862, test_L2=0.2862
[237] time=3.46, avg_loss=0.2144, train_err=8.5744, test_H1=0.3037, test_L2=0.3037
[240] time=3.31, avg_loss=0.1988, train_err=7.9511, test_H1=0.2924, test_L2=0.2924
[243] time=3.44, avg_loss=0.2535, train_err=10.1409, test_H1=0.2925, test_L2=0.2925
[246] time=3.47, avg_loss=0.1774, train_err=7.0966, test_H1=0.2750, test_L2=0.2750
[249] time=3.32, avg_loss=0.1740, train_err=6.9613, test_H1=0.2763, test_L2=0.2763
[252] time=3.45, avg_loss=0.1607, train_err=6.4266, test_H1=0.2650, test_L2=0.2650
[255] time=3.32, avg_loss=0.1635, train_err=6.5383, test_H1=0.2759, test_L2=0.2759
[258] time=3.40, avg_loss=0.1646, train_err=6.5823, test_H1=0.2664, test_L2=0.2664
[261] time=1.57, avg_loss=0.1618, train_err=6.4734, test_H1=0.2637, test_L2=0.2637
[264] time=1.56, avg_loss=0.1679, train_err=6.7178, test_H1=0.2805, test_L2=0.2805
[267] time=1.56, avg_loss=0.1726, train_err=6.9041, test_H1=0.2775, test_L2=0.2775
[270] time=1.56, avg_loss=0.1669, train_err=6.6745, test_H1=0.2675, test_L2=0.2675
[273] time=3.43, avg_loss=0.1840, train_err=7.3609, test_H1=0.2851, test_L2=0.2851
[276] time=3.45, avg_loss=0.1584, train_err=6.3369, test_H1=0.2616, test_L2=0.2616
[279] time=3.32, avg_loss=0.1781, train_err=7.1243, test_H1=0.2995, test_L2=0.2995
[282] time=3.46, avg_loss=0.1662, train_err=6.6470, test_H1=0.2662, test_L2=0.2662
[285] time=3.31, avg_loss=0.1558, train_err=6.2331, test_H1=0.2610, test_L2=0.2610
[288] time=3.46, avg_loss=0.1570, train_err=6.2807, test_H1=0.2610, test_L2=0.2610
[291] time=3.34, avg_loss=0.1658, train_err=6.6327, test_H1=0.2679, test_L2=0.2679
[294] time=3.30, avg_loss=0.1586, train_err=6.3425, test_H1=0.2621, test_L2=0.2621
[297] time=3.45, avg_loss=0.1555, train_err=6.2206, test_H1=0.2611, test_L2=0.2611
[300] time=3.31, avg_loss=0.1602, train_err=6.4064, test_H1=0.2764, test_L2=0.2764
[303] time=3.47, avg_loss=0.1550, train_err=6.2003, test_H1=0.2615, test_L2=0.2615
[306] time=3.38, avg_loss=0.1518, train_err=6.0720, test_H1=0.2645, test_L2=0.2645
[309] time=3.31, avg_loss=0.1471, train_err=5.8832, test_H1=0.2569, test_L2=0.2569
[312] time=3.45, avg_loss=0.1498, train_err=5.9926, test_H1=0.2515, test_L2=0.2515
(1000, 4, 2048) (1000, 1, 2048)
Total number of samples: 1000
Input data shape: (1000, 4, 2048)
Output series shape: (1000, 1, 2048)
Batch input series shape: torch.Size([32, 4, 2048])
Batch output series shape: torch.Size([32, 1, 2048])
Dtype torch.complex64 torch.complex64

Our model has 33919746 parameters.
torch.Size([8, 128, 1]) torch.Size([128, 128, 128, 2]) torch.Size([128, 128, 128, 2]) torch.Size([128, 128, 128, 2])

### MODEL ###
 FNO(
  (positional_embedding): GridEmbeddingND()
  (fno_blocks): FNOBlocks(
    (convs): SpectralConv(
      (weight): ModuleList(
        (0-7): 8 x ComplexDenseTensor(shape=torch.Size([128, 128, 128]), rank=None)
      )
    )
    (fno_skips): ModuleList(
      (0-7): 8 x ComplexValued(
        (fr): Conv1d(128, 128, kernel_size=(1,), stride=(1,), bias=False)
        (fi): Conv1d(128, 128, kernel_size=(1,), stride=(1,), bias=False)
      )
    )
  )
  (lifting): ComplexValued(
    (fr): ChannelMLP(
      (fcs): ModuleList(
        (0): Conv1d(5, 256, kernel_size=(1,), stride=(1,))
        (1): Conv1d(256, 128, kernel_size=(1,), stride=(1,))
      )
    )
    (fi): ChannelMLP(
      (fcs): ModuleList(
        (0): Conv1d(5, 256, kernel_size=(1,), stride=(1,))
        (1): Conv1d(256, 128, kernel_size=(1,), stride=(1,))
      )
    )
  )
  (projection): ComplexValued(
    (fr): ChannelMLP(
      (fcs): ModuleList(
        (0): Conv1d(128, 128, kernel_size=(1,), stride=(1,))
        (1): Conv1d(128, 1, kernel_size=(1,), stride=(1,))
      )
    )
    (fi): ChannelMLP(
      (fcs): ModuleList(
        (0): Conv1d(128, 128, kernel_size=(1,), stride=(1,))
        (1): Conv1d(128, 1, kernel_size=(1,), stride=(1,))
      )
    )
  )
)

### OPTIMIZER ###
 AdamW (
Parameter Group 0
    betas: (0.9, 0.999)
    correct_bias: True
    eps: 1e-06
    initial_lr: 0.001
    lr: 0.001
    weight_decay: 2e-06

Parameter Group 1
    betas: (0.9, 0.999)
    correct_bias: True
    dim: 5
    eps: 1e-06
    initial_lr: 0.001
    lr: 0.001
    proj_type: std
    rank: 0.5
    scale: 1.0
    type: tucker
    update_proj_gap: 1
    weight_decay: 2e-06
)

### SCHEDULER ###
 <torch.optim.lr_scheduler.StepLR object at 0x7f64d2308ee0>

### LOSSES ###

 * Train: <neuralop.losses.data_losses.H1Loss object at 0x7f64d23091e0>

 * Test: {'H1': <neuralop.losses.data_losses.H1Loss object at 0x7f64d23091e0>, 'L2': <neuralop.losses.data_losses.LpLoss object at 0x7f64d23097b0>}
using standard method to load data to device.
using standard method to compute loss.
self.override_load_to_device=False
self.overrides_loss=False
Training on 800 samples
Testing on [200] samples         on resolutions ['test'].
Raw outputs of size out.shape=torch.Size([32, 1, 2048])
[0] time=39.32, avg_loss=0.7997, train_err=31.9885, test_H1=0.9986, test_L2=0.9986
[3] time=3.47, avg_loss=0.7962, train_err=31.8490, test_H1=1.0001, test_L2=1.0001
[6] time=3.61, avg_loss=0.7886, train_err=31.5427, test_H1=0.9954, test_L2=0.9954
[9] time=3.47, avg_loss=0.7968, train_err=31.8722, test_H1=0.9904, test_L2=0.9904
[12] time=3.60, avg_loss=0.7858, train_err=31.4309, test_H1=0.9792, test_L2=0.9792
[15] time=3.51, avg_loss=0.7717, train_err=30.8671, test_H1=0.9755, test_L2=0.9755
[18] time=3.51, avg_loss=0.7714, train_err=30.8571, test_H1=0.9626, test_L2=0.9626
[21] time=3.60, avg_loss=0.7620, train_err=30.4815, test_H1=0.9442, test_L2=0.9442
[24] time=3.47, avg_loss=0.7392, train_err=29.5684, test_H1=0.9437, test_L2=0.9437
[27] time=3.60, avg_loss=0.7392, train_err=29.5669, test_H1=0.9248, test_L2=0.9248
[30] time=3.47, avg_loss=0.7468, train_err=29.8724, test_H1=0.9250, test_L2=0.9250
[33] time=3.60, avg_loss=0.7307, train_err=29.2279, test_H1=0.9136, test_L2=0.9136
[36] time=3.47, avg_loss=0.7211, train_err=28.8426, test_H1=0.9440, test_L2=0.9440
[39] time=3.59, avg_loss=0.7111, train_err=28.4439, test_H1=0.8715, test_L2=0.8715
[42] time=3.53, avg_loss=0.6962, train_err=27.8483, test_H1=0.9147, test_L2=0.9147
[45] time=3.52, avg_loss=0.6758, train_err=27.0332, test_H1=0.8279, test_L2=0.8279
[48] time=3.61, avg_loss=0.6459, train_err=25.8345, test_H1=0.8072, test_L2=0.8072
[51] time=3.47, avg_loss=0.6210, train_err=24.8384, test_H1=0.8626, test_L2=0.8626
[54] time=3.61, avg_loss=0.5667, train_err=22.6694, test_H1=0.7415, test_L2=0.7415
[57] time=3.47, avg_loss=0.5200, train_err=20.8008, test_H1=0.6919, test_L2=0.6919
[60] time=3.60, avg_loss=0.4987, train_err=19.9482, test_H1=0.6985, test_L2=0.6985
[63] time=3.46, avg_loss=0.4172, train_err=16.6878, test_H1=0.5591, test_L2=0.5591
[66] time=3.61, avg_loss=0.4151, train_err=16.6050, test_H1=0.5156, test_L2=0.5156
[69] time=3.56, avg_loss=0.3908, train_err=15.6330, test_H1=0.5088, test_L2=0.5088
[72] time=3.46, avg_loss=0.4174, train_err=16.6972, test_H1=0.5507, test_L2=0.5507
[75] time=3.61, avg_loss=0.3523, train_err=14.0923, test_H1=0.5210, test_L2=0.5210
[78] time=3.47, avg_loss=0.3554, train_err=14.2167, test_H1=0.4448, test_L2=0.4448
[81] time=3.60, avg_loss=0.3769, train_err=15.0779, test_H1=0.5562, test_L2=0.5562
[84] time=3.46, avg_loss=0.3467, train_err=13.8670, test_H1=0.5427, test_L2=0.5427
[87] time=3.61, avg_loss=0.3301, train_err=13.2048, test_H1=0.4441, test_L2=0.4441
[90] time=3.48, avg_loss=0.3054, train_err=12.2160, test_H1=0.4076, test_L2=0.4076
[93] time=3.58, avg_loss=0.3396, train_err=13.5849, test_H1=0.4569, test_L2=0.4569
[96] time=3.61, avg_loss=0.2954, train_err=11.8153, test_H1=0.3980, test_L2=0.3980
[99] time=3.47, avg_loss=0.2950, train_err=11.7997, test_H1=0.4157, test_L2=0.4157
[102] time=3.60, avg_loss=0.2850, train_err=11.4015, test_H1=0.4076, test_L2=0.4076
[105] time=3.46, avg_loss=0.3278, train_err=13.1132, test_H1=0.4016, test_L2=0.4016
[108] time=3.60, avg_loss=0.2904, train_err=11.6154, test_H1=0.4132, test_L2=0.4132
[111] time=3.46, avg_loss=0.2735, train_err=10.9393, test_H1=0.3654, test_L2=0.3654
[114] time=3.60, avg_loss=0.2982, train_err=11.9281, test_H1=0.4036, test_L2=0.4036
[117] time=3.47, avg_loss=0.2682, train_err=10.7276, test_H1=0.3628, test_L2=0.3628
[120] time=3.59, avg_loss=0.2634, train_err=10.5372, test_H1=0.3544, test_L2=0.3544
[123] time=3.61, avg_loss=0.2773, train_err=11.0907, test_H1=0.4415, test_L2=0.4415
[126] time=3.47, avg_loss=0.2832, train_err=11.3277, test_H1=0.3964, test_L2=0.3964
[129] time=3.60, avg_loss=0.2922, train_err=11.6864, test_H1=0.4473, test_L2=0.4473
[132] time=3.47, avg_loss=0.2673, train_err=10.6928, test_H1=0.3670, test_L2=0.3670
[135] time=3.61, avg_loss=0.2292, train_err=9.1670, test_H1=0.3126, test_L2=0.3126
[138] time=3.48, avg_loss=0.2169, train_err=8.6741, test_H1=0.3073, test_L2=0.3073
[141] time=3.61, avg_loss=0.2152, train_err=8.6071, test_H1=0.3268, test_L2=0.3268
[144] time=3.47, avg_loss=0.2314, train_err=9.2547, test_H1=0.3961, test_L2=0.3961
[147] time=3.60, avg_loss=0.2192, train_err=8.7661, test_H1=0.3152, test_L2=0.3152
[150] time=3.60, avg_loss=0.2150, train_err=8.6003, test_H1=0.3181, test_L2=0.3181
[153] time=3.47, avg_loss=0.2004, train_err=8.0149, test_H1=0.3106, test_L2=0.3106
[156] time=3.61, avg_loss=0.1975, train_err=7.8980, test_H1=0.3015, test_L2=0.3015
[159] time=3.47, avg_loss=0.2101, train_err=8.4040, test_H1=0.3065, test_L2=0.3065
[162] time=3.60, avg_loss=0.2130, train_err=8.5214, test_H1=0.3166, test_L2=0.3166
[165] time=3.47, avg_loss=0.1921, train_err=7.6858, test_H1=0.2858, test_L2=0.2858
[168] time=3.61, avg_loss=0.2004, train_err=8.0148, test_H1=0.2932, test_L2=0.2932
[171] time=3.47, avg_loss=0.1992, train_err=7.9685, test_H1=0.2941, test_L2=0.2941
[174] time=3.55, avg_loss=0.1825, train_err=7.2983, test_H1=0.2848, test_L2=0.2848
[177] time=3.60, avg_loss=0.1889, train_err=7.5549, test_H1=0.2959, test_L2=0.2959
[180] time=3.46, avg_loss=0.2279, train_err=9.1168, test_H1=0.3251, test_L2=0.3251
[183] time=3.60, avg_loss=0.1819, train_err=7.2751, test_H1=0.2843, test_L2=0.2843
[186] time=3.46, avg_loss=0.1909, train_err=7.6344, test_H1=0.2857, test_L2=0.2857
[189] time=3.60, avg_loss=0.1864, train_err=7.4542, test_H1=0.2746, test_L2=0.2746
[192] time=3.47, avg_loss=0.1816, train_err=7.2659, test_H1=0.2859, test_L2=0.2859
[195] time=3.60, avg_loss=0.1721, train_err=6.8858, test_H1=0.2731, test_L2=0.2731
[198] time=3.48, avg_loss=0.2146, train_err=8.5850, test_H1=0.3257, test_L2=0.3257
[201] time=3.58, avg_loss=0.1725, train_err=6.9003, test_H1=0.2685, test_L2=0.2685
[204] time=3.60, avg_loss=0.1610, train_err=6.4397, test_H1=0.2593, test_L2=0.2593
[207] time=3.47, avg_loss=0.2360, train_err=9.4385, test_H1=0.3379, test_L2=0.3379
[210] time=3.62, avg_loss=0.1620, train_err=6.4815, test_H1=0.2614, test_L2=0.2614
[213] time=3.47, avg_loss=0.1788, train_err=7.1524, test_H1=0.2926, test_L2=0.2926
[216] time=3.61, avg_loss=0.1962, train_err=7.8472, test_H1=0.2770, test_L2=0.2770
[219] time=3.48, avg_loss=0.1808, train_err=7.2309, test_H1=0.2726, test_L2=0.2726
[222] time=3.61, avg_loss=0.1508, train_err=6.0322, test_H1=0.2542, test_L2=0.2542
[225] time=3.47, avg_loss=0.1538, train_err=6.1518, test_H1=0.2675, test_L2=0.2675
[228] time=3.54, avg_loss=0.1718, train_err=6.8711, test_H1=0.2714, test_L2=0.2714
[231] time=3.60, avg_loss=0.1536, train_err=6.1451, test_H1=0.2715, test_L2=0.2715
[234] time=3.47, avg_loss=0.1672, train_err=6.6870, test_H1=0.2606, test_L2=0.2606
[237] time=3.60, avg_loss=0.1424, train_err=5.6976, test_H1=0.2528, test_L2=0.2528
[240] time=3.48, avg_loss=0.1731, train_err=6.9256, test_H1=0.3132, test_L2=0.3132
[243] time=3.60, avg_loss=0.1473, train_err=5.8925, test_H1=0.2641, test_L2=0.2641
[246] time=3.48, avg_loss=0.1516, train_err=6.0643, test_H1=0.2784, test_L2=0.2784
[249] time=3.60, avg_loss=0.1877, train_err=7.5096, test_H1=0.2824, test_L2=0.2824
[252] time=3.50, avg_loss=0.1339, train_err=5.3558, test_H1=0.2350, test_L2=0.2350
[255] time=3.53, avg_loss=0.1393, train_err=5.5724, test_H1=0.2428, test_L2=0.2428
[258] time=3.61, avg_loss=0.1340, train_err=5.3600, test_H1=0.2378, test_L2=0.2378
[261] time=3.45, avg_loss=0.1470, train_err=5.8805, test_H1=0.2510, test_L2=0.2510
[264] time=3.60, avg_loss=0.1559, train_err=6.2376, test_H1=0.2613, test_L2=0.2613
[267] time=3.46, avg_loss=0.1284, train_err=5.1380, test_H1=0.2353, test_L2=0.2353
[270] time=3.61, avg_loss=0.1255, train_err=5.0184, test_H1=0.2318, test_L2=0.2318
[273] time=3.46, avg_loss=0.1281, train_err=5.1224, test_H1=0.2613, test_L2=0.2613
[276] time=3.59, avg_loss=0.1265, train_err=5.0608, test_H1=0.2482, test_L2=0.2482
[279] time=3.53, avg_loss=0.1331, train_err=5.3226, test_H1=0.2430, test_L2=0.2430
[282] time=3.54, avg_loss=0.1281, train_err=5.1249, test_H1=0.2401, test_L2=0.2401
[285] time=3.61, avg_loss=0.1310, train_err=5.2392, test_H1=0.2374, test_L2=0.2374
[288] time=3.48, avg_loss=0.1264, train_err=5.0576, test_H1=0.2374, test_L2=0.2374
[291] time=3.61, avg_loss=0.1231, train_err=4.9230, test_H1=0.2522, test_L2=0.2522
[294] time=3.48, avg_loss=0.1212, train_err=4.8493, test_H1=0.2312, test_L2=0.2312
[297] time=3.60, avg_loss=0.1339, train_err=5.3577, test_H1=0.2429, test_L2=0.2429
[300] time=3.48, avg_loss=0.1186, train_err=4.7435, test_H1=0.2262, test_L2=0.2262
[303] time=3.61, avg_loss=0.1220, train_err=4.8806, test_H1=0.2261, test_L2=0.2261
[306] time=3.51, avg_loss=0.1035, train_err=4.1387, test_H1=0.2139, test_L2=0.2139
[309] time=3.51, avg_loss=0.1158, train_err=4.6304, test_H1=0.2279, test_L2=0.2279
[312] time=3.61, avg_loss=0.1120, train_err=4.4818, test_H1=0.2366, test_L2=0.2366
[315] time=3.48, avg_loss=0.1141, train_err=4.5652, test_H1=0.2347, test_L2=0.2347
[318] time=3.61, avg_loss=0.1128, train_err=4.5130, test_H1=0.2699, test_L2=0.2699
[321] time=3.47, avg_loss=0.1091, train_err=4.3640, test_H1=0.2099, test_L2=0.2099
[324] time=3.61, avg_loss=0.1032, train_err=4.1282, test_H1=0.2121, test_L2=0.2121
[327] time=3.48, avg_loss=0.1135, train_err=4.5393, test_H1=0.2245, test_L2=0.2245
[330] time=3.61, avg_loss=0.0974, train_err=3.8944, test_H1=0.2103, test_L2=0.2103
[333] time=3.53, avg_loss=0.1039, train_err=4.1542, test_H1=0.2247, test_L2=0.2247
[336] time=3.48, avg_loss=0.1054, train_err=4.2164, test_H1=0.2201, test_L2=0.2201
[339] time=3.60, avg_loss=0.1103, train_err=4.4135, test_H1=0.2264, test_L2=0.2264
[342] time=3.44, avg_loss=0.1151, train_err=4.6027, test_H1=0.2250, test_L2=0.2250
[345] time=3.59, avg_loss=0.1088, train_err=4.3517, test_H1=0.2114, test_L2=0.2114
[348] time=3.47, avg_loss=0.1117, train_err=4.4695, test_H1=0.2178, test_L2=0.2178
[351] time=3.60, avg_loss=0.1062, train_err=4.2468, test_H1=0.2096, test_L2=0.2096
[354] time=3.46, avg_loss=0.0914, train_err=3.6554, test_H1=0.1993, test_L2=0.1993
[357] time=3.60, avg_loss=0.0884, train_err=3.5350, test_H1=0.1975, test_L2=0.1975
[360] time=3.60, avg_loss=0.0835, train_err=3.3410, test_H1=0.1968, test_L2=0.1968
[363] time=3.48, avg_loss=0.1065, train_err=4.2581, test_H1=0.2275, test_L2=0.2275
[366] time=3.60, avg_loss=0.0882, train_err=3.5286, test_H1=0.1992, test_L2=0.1992
[369] time=3.46, avg_loss=0.0962, train_err=3.8479, test_H1=0.2006, test_L2=0.2006
[372] time=3.61, avg_loss=0.0999, train_err=3.9965, test_H1=0.2558, test_L2=0.2558
[375] time=3.46, avg_loss=0.0910, train_err=3.6395, test_H1=0.1967, test_L2=0.1967
[378] time=3.61, avg_loss=0.0840, train_err=3.3612, test_H1=0.2181, test_L2=0.2181
[381] time=3.48, avg_loss=0.0846, train_err=3.3836, test_H1=0.1948, test_L2=0.1948
[384] time=3.61, avg_loss=0.0828, train_err=3.3138, test_H1=0.2118, test_L2=0.2118
[387] time=3.59, avg_loss=0.0800, train_err=3.1986, test_H1=0.1969, test_L2=0.1969
[390] time=3.48, avg_loss=0.0893, train_err=3.5715, test_H1=0.1970, test_L2=0.1970
[393] time=3.60, avg_loss=0.0863, train_err=3.4514, test_H1=0.2304, test_L2=0.2304
[396] time=3.49, avg_loss=0.1002, train_err=4.0067, test_H1=0.2051, test_L2=0.2051
[399] time=3.60, avg_loss=0.0792, train_err=3.1698, test_H1=0.1928, test_L2=0.1928
[402] time=3.48, avg_loss=0.0762, train_err=3.0461, test_H1=0.1873, test_L2=0.1873
[405] time=3.61, avg_loss=0.0794, train_err=3.1774, test_H1=0.2019, test_L2=0.2019
[408] time=3.47, avg_loss=0.1229, train_err=4.9147, test_H1=0.2330, test_L2=0.2330
[411] time=3.57, avg_loss=0.0854, train_err=3.4173, test_H1=0.1987, test_L2=0.1987
[414] time=3.60, avg_loss=0.0814, train_err=3.2552, test_H1=0.1933, test_L2=0.1933
[417] time=3.47, avg_loss=0.0837, train_err=3.3476, test_H1=0.2068, test_L2=0.2068
[420] time=3.60, avg_loss=0.0741, train_err=2.9647, test_H1=0.1907, test_L2=0.1907
[423] time=3.47, avg_loss=0.0772, train_err=3.0899, test_H1=0.1881, test_L2=0.1881
[426] time=3.61, avg_loss=0.0775, train_err=3.1000, test_H1=0.2001, test_L2=0.2001
[429] time=3.46, avg_loss=0.0741, train_err=2.9625, test_H1=0.1882, test_L2=0.1882
[432] time=3.60, avg_loss=0.0821, train_err=3.2826, test_H1=0.1898, test_L2=0.1898
[435] time=3.46, avg_loss=0.0789, train_err=3.1572, test_H1=0.2056, test_L2=0.2056
[438] time=3.60, avg_loss=0.0774, train_err=3.0956, test_H1=0.2211, test_L2=0.2211
[441] time=3.60, avg_loss=0.0669, train_err=2.6771, test_H1=0.1876, test_L2=0.1876
[444] time=3.48, avg_loss=0.0857, train_err=3.4287, test_H1=0.1911, test_L2=0.1911
[447] time=3.60, avg_loss=0.0820, train_err=3.2792, test_H1=0.1963, test_L2=0.1963
[450] time=3.46, avg_loss=0.0701, train_err=2.8036, test_H1=0.1950, test_L2=0.1950
[453] time=3.61, avg_loss=0.0625, train_err=2.5003, test_H1=0.1835, test_L2=0.1835
[456] time=3.47, avg_loss=0.0762, train_err=3.0481, test_H1=0.1967, test_L2=0.1967
[459] time=3.60, avg_loss=0.0731, train_err=2.9255, test_H1=0.1925, test_L2=0.1925
[462] time=3.48, avg_loss=0.0862, train_err=3.4496, test_H1=0.2143, test_L2=0.2143
[465] time=3.58, avg_loss=0.0669, train_err=2.6750, test_H1=0.1839, test_L2=0.1839
[468] time=3.60, avg_loss=0.0700, train_err=2.8015, test_H1=0.1860, test_L2=0.1860
[471] time=3.48, avg_loss=0.0618, train_err=2.4713, test_H1=0.1834, test_L2=0.1834
[474] time=3.61, avg_loss=0.0618, train_err=2.4733, test_H1=0.1904, test_L2=0.1904
[477] time=3.48, avg_loss=0.0629, train_err=2.5176, test_H1=0.1796, test_L2=0.1796
[480] time=3.60, avg_loss=0.0632, train_err=2.5292, test_H1=0.1832, test_L2=0.1832
[483] time=3.47, avg_loss=0.0631, train_err=2.5228, test_H1=0.1797, test_L2=0.1797
[486] time=3.61, avg_loss=0.0602, train_err=2.4068, test_H1=0.1822, test_L2=0.1822
[489] time=3.47, avg_loss=0.0668, train_err=2.6722, test_H1=0.1832, test_L2=0.1832
[492] time=3.54, avg_loss=0.0591, train_err=2.3628, test_H1=0.1817, test_L2=0.1817
[495] time=3.60, avg_loss=0.0838, train_err=3.3502, test_H1=0.1984, test_L2=0.1984
[498] time=3.47, avg_loss=0.0822, train_err=3.2861, test_H1=0.1921, test_L2=0.1921
[501] time=3.59, avg_loss=0.0590, train_err=2.3588, test_H1=0.1833, test_L2=0.1833
[504] time=3.47, avg_loss=0.0629, train_err=2.5180, test_H1=0.1869, test_L2=0.1869
[507] time=3.60, avg_loss=0.0689, train_err=2.7577, test_H1=0.1869, test_L2=0.1869
[510] time=3.47, avg_loss=0.0548, train_err=2.1913, test_H1=0.1769, test_L2=0.1769
[513] time=3.60, avg_loss=0.0654, train_err=2.6173, test_H1=0.2279, test_L2=0.2279
[516] time=3.49, avg_loss=0.0559, train_err=2.2341, test_H1=0.1836, test_L2=0.1836
[519] time=3.59, avg_loss=0.0562, train_err=2.2469, test_H1=0.1779, test_L2=0.1779
[522] time=3.60, avg_loss=0.0651, train_err=2.6034, test_H1=0.1881, test_L2=0.1881
[525] time=3.47, avg_loss=0.0562, train_err=2.2467, test_H1=0.1782, test_L2=0.1782
[528] time=3.60, avg_loss=0.0723, train_err=2.8927, test_H1=0.1831, test_L2=0.1831
[531] time=3.47, avg_loss=0.0680, train_err=2.7216, test_H1=0.1867, test_L2=0.1867
[534] time=3.63, avg_loss=0.0527, train_err=2.1075, test_H1=0.1759, test_L2=0.1759
[537] time=3.47, avg_loss=0.0701, train_err=2.8057, test_H1=0.1841, test_L2=0.1841
[540] time=3.61, avg_loss=0.0600, train_err=2.4013, test_H1=0.1780, test_L2=0.1780
[543] time=3.48, avg_loss=0.0507, train_err=2.0275, test_H1=0.1762, test_L2=0.1762
[546] time=3.54, avg_loss=0.0555, train_err=2.2200, test_H1=0.1887, test_L2=0.1887
[549] time=3.61, avg_loss=0.0717, train_err=2.8688, test_H1=0.1901, test_L2=0.1901
[552] time=3.47, avg_loss=0.0570, train_err=2.2790, test_H1=0.1760, test_L2=0.1760
[555] time=3.61, avg_loss=0.0618, train_err=2.4716, test_H1=0.1841, test_L2=0.1841
[558] time=3.48, avg_loss=0.0519, train_err=2.0768, test_H1=0.1733, test_L2=0.1733
[561] time=3.61, avg_loss=0.0516, train_err=2.0624, test_H1=0.1737, test_L2=0.1737
[564] time=3.47, avg_loss=0.0488, train_err=1.9536, test_H1=0.1723, test_L2=0.1723
[567] time=3.61, avg_loss=0.0563, train_err=2.2532, test_H1=0.1838, test_L2=0.1838
[570] time=3.51, avg_loss=0.0499, train_err=1.9962, test_H1=0.1764, test_L2=0.1764
[573] time=3.52, avg_loss=0.0508, train_err=2.0305, test_H1=0.1744, test_L2=0.1744
[576] time=3.60, avg_loss=0.0475, train_err=1.8991, test_H1=0.1798, test_L2=0.1798
[579] time=3.46, avg_loss=0.0527, train_err=2.1090, test_H1=0.1793, test_L2=0.1793
[582] time=3.61, avg_loss=0.0598, train_err=2.3936, test_H1=0.1773, test_L2=0.1773
[585] time=3.47, avg_loss=0.0484, train_err=1.9343, test_H1=0.1744, test_L2=0.1744
[588] time=3.60, avg_loss=0.0505, train_err=2.0192, test_H1=0.1742, test_L2=0.1742
[591] time=3.47, avg_loss=0.0499, train_err=1.9940, test_H1=0.1724, test_L2=0.1724
[594] time=3.60, avg_loss=0.0512, train_err=2.0492, test_H1=0.1809, test_L2=0.1809
[597] time=3.53, avg_loss=0.0491, train_err=1.9653, test_H1=0.1707, test_L2=0.1707
[600] time=3.54, avg_loss=0.0479, train_err=1.9165, test_H1=0.1703, test_L2=0.1703
[603] time=3.60, avg_loss=0.0498, train_err=1.9939, test_H1=0.1705, test_L2=0.1705
[606] time=3.48, avg_loss=0.0520, train_err=2.0796, test_H1=0.1709, test_L2=0.1709
[609] time=3.60, avg_loss=0.0488, train_err=1.9521, test_H1=0.1723, test_L2=0.1723
[612] time=3.46, avg_loss=0.0473, train_err=1.8931, test_H1=0.1715, test_L2=0.1715
[615] time=3.61, avg_loss=0.0437, train_err=1.7472, test_H1=0.1711, test_L2=0.1711
[618] time=3.47, avg_loss=0.0443, train_err=1.7712, test_H1=0.1700, test_L2=0.1700
[621] time=3.61, avg_loss=0.0510, train_err=2.0390, test_H1=0.1741, test_L2=0.1741
[624] time=3.52, avg_loss=0.0453, train_err=1.8122, test_H1=0.1705, test_L2=0.1705
[627] time=3.49, avg_loss=0.0448, train_err=1.7917, test_H1=0.1773, test_L2=0.1773
[630] time=3.61, avg_loss=0.0443, train_err=1.7723, test_H1=0.1716, test_L2=0.1716
[633] time=3.47, avg_loss=0.0427, train_err=1.7090, test_H1=0.1714, test_L2=0.1714
[636] time=1.64, avg_loss=0.0495, train_err=1.9819, test_H1=0.1734, test_L2=0.1734
[639] time=3.48, avg_loss=0.0564, train_err=2.2545, test_H1=0.1748, test_L2=0.1748
[642] time=3.61, avg_loss=0.0481, train_err=1.9224, test_H1=0.1710, test_L2=0.1710
[645] time=3.47, avg_loss=0.0438, train_err=1.7526, test_H1=0.1701, test_L2=0.1701
[648] time=3.61, avg_loss=0.0542, train_err=2.1689, test_H1=0.1768, test_L2=0.1768
[651] time=3.48, avg_loss=0.0556, train_err=2.2235, test_H1=0.1841, test_L2=0.1841
[654] time=3.55, avg_loss=0.0408, train_err=1.6303, test_H1=0.1702, test_L2=0.1702
[657] time=3.61, avg_loss=0.0428, train_err=1.7116, test_H1=0.1731, test_L2=0.1731
[660] time=3.49, avg_loss=0.0378, train_err=1.5121, test_H1=0.1679, test_L2=0.1679
[663] time=3.60, avg_loss=0.0377, train_err=1.5067, test_H1=0.1682, test_L2=0.1682
[666] time=3.47, avg_loss=0.0436, train_err=1.7457, test_H1=0.1718, test_L2=0.1718
[669] time=3.61, avg_loss=0.0450, train_err=1.8017, test_H1=0.1703, test_L2=0.1703
[672] time=3.47, avg_loss=0.0477, train_err=1.9090, test_H1=0.1762, test_L2=0.1762
[675] time=3.62, avg_loss=0.0517, train_err=2.0683, test_H1=0.1718, test_L2=0.1718
[678] time=3.48, avg_loss=0.0397, train_err=1.5892, test_H1=0.1776, test_L2=0.1776
[681] time=3.54, avg_loss=0.0401, train_err=1.6028, test_H1=0.1779, test_L2=0.1779
[684] time=3.60, avg_loss=0.0451, train_err=1.8048, test_H1=0.1696, test_L2=0.1696
