# Hardware Implemented LSS+HQ

Code for hardware implementation LSS and HQ operator.

## INSTALL

Tested with PyTorch 1.12.1 + CUDA 11.3, on an Nvidia T4 GPU.

> Note: This cuda program is based on [Nvidia cutlass](https://github.com/NVIDIA/cutlass) version 2.10. You need to pull down the corresponding version library. Besides, in quantize_forward_HQ/setup_easy.py and quantize_grad_weight_LSS/setup.py, you need to change the path of `include_dirs` into the absolute path on your own computer, only in this way can it work normally.

### CutLass

```
git clone git@github.com:NVIDIA/cutlass.git 
#checkout branch
git checkout feature/2.10/updates_before_tagging
```

### LSS

```
cd quantize_grad_weight_LSS
python setup.py install
```

### HQ

```
cd quantize_forward_HQ
python setup_easy.py install
```



## Tflops

Operator speed between FP16-GEMM, LssWeight, LssInput and HQ operator. Metric is tflops when the shapes of input matrices vary.

```
cd quantize_grad_weight_LSS
python test.py
```

The result is shown in quantize_grad_weight_LSS/image/plot_flops.pdf



## Proportion of time

Proportion of time for each part of computation in LssWeight, LssInput and HQ operator

### LSS

```
cd quantize_grad_weight_LSS
python test.py
```

The result of LssWeight is shown in quantize_grad_weight_LSS/image/plot_time.pdf

The result of LssInput is shown in quantize_grad_weight_LSS/image/plot_time2.pdf

### HQ

```
cd quantize_forward_HQ
python test.py
```

The result is shown in quantize_forward_HQ/image/plot_time.pdf