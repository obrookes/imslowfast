# Official code for "Understanding Background Importance in Wildlife Behaviour Recognition via the PanAf-FGBG Dataset"

This repository based on [SlowFast codebase](https://github.com/facebookresearch/SlowFast)

## SlowFast Installation
Please refer to [INSTALL.md](https://github.com/facebookresearch/SlowFast/blob/main/INSTALL.md) for installation instructions.


## MVIT-V2 Models
| Models | Config |
| -------- | ------- |
| MVIT-V2 Baseline | config/MVIT_B_16x4.yaml |
| MVIT-V2 with Background subtraction | config/MVIT_B_16x4_BGFG_MIXUP.yaml |
| Dual-MVIT-V2 with Background subtraction | config/Dual_MVIT_B_16x4_BGFG_MIXUP.yaml |
| MVIT-V2 with Background concatenation on IDD | config/MVIT_B_16x4_CONCAT_BG_SUB.yaml |
| MVIT-V2 with Background concatenation on ODD | config/Dual_MVIT_B_16x4_CONCAT_BG_SUB_excl.yaml |

## ResNet-50 Models
| Models | Config |
| -------- | ------- |
| ResNet-50 Baseline | config/SLOW_8x8_R50.yaml |
| ResNet-50 with Background subtraction | config/ResNet_50_BGFG_MIXUP.yaml |
| Dual-ResNet-50 with Background subtraction | config/Dual_ResNet_50_BGFG_MIXUP.yaml |
| ResNet-50 with Background concatenation on IDD | config/ResNet_50_CONCAT_BG_SUB.yaml |
| ResNet-50 with Background concatenation on ODD | config/ResNet_50_CONCAT_BG_SUB_excl.yaml |


## Running Experiments 
To run experiments run e.g. for Dual_MVIT_B_16x4_BGFG_MIXUP model:
```bash
python tools/run_net.py --cfg configs/Dual_MVIT_B_16x4_BGFG_MIXUP.yaml
```

## Generate Synthetic Background Videos
* To generate synthetic background videos using Frame Difference run:
```bash
python bg_sub_fd.py
```
* To generate synthetic background videos using Gaussian Mixture Models run:
```bash
python bg_sub_gmm.py
```
* To generate synthetic background videos using SAM2 run:
```bash
python blabla.py
```