# Lightning-NeRF ICRA 2024

:page_facing_up: Lightning NeRF: Efficient Hybrid Scene Representation for Autonomous Driving

:boy: Junyi Cao, Zhichao Li, Naiyan Wang, Chao Ma

**Please consider citing our paper if you find it interesting or helpful to your research.**
```
TO BE DONE
```

---

### Introduction

This repository provides code to integrate the Lightning NeRF into [NeRFStudio](https://docs.nerf.studio/en/latest). Lightning NeRF is an efficient novel view synthesis framework for outdoor scenes that integrates point clouds and images.

### Dependencies
- [PyTorch](https://pytorch.org/get-started/previous-versions) 1.13.1
- [NeRFAcc](https://github.com/KAIR-BAIR/nerfacc) 0.5.2
- [Tiny CUDA Neural Networks](https://github.com/NVlabs/tiny-cuda-nn)
- [nr3d_lib](https://github.com/PJLab-ADG/nr3d_lib) 0.3.1
- [NeRFStudio](https://github.com/nerfstudio-project/nerfstudio) 0.2.2

### Installation
0. Make sure the dependencies are resolved.
1. Clone the repository:
    ```bash
    git clone https://github.com/VISION-SJTU/Lightning-NeRF.git
    ```
1. Install Lightning NeRF:
    ```bash
    cd Lightning-NeRF
    pip install -e .
    ```

### Data
0. **Use our data pack.** You may skip the following steps 1 and 2 by downloading the data pack used in our experiments. We will provide the download link soon.
1. **Download source data.** We use [KITTI-360](https://www.cvlibs.net/datasets/kitti-360/index.php) and [Argoverse2 (Sensor Dataset)](https://argoverse.github.io/user-guide/datasets/sensor.html) for experiments. Please download the original data from the offical webpages. Here, we list the chosen scenes presented in our paper.

<div align=center>

<img 
    src="https://s1.locimg.com/2023/09/22/0af88b2bd34b7.png" 
    title="KITTI-360"
    alter="KITTI-360" 
    width="400px">

<img 
    src="https://s1.locimg.com/2023/09/22/8b05a58c1ff77.png" 
    title="Argoverse2"
    alter="Argoverse2" 
    width="400px">

</div>

2. **Preprocess the data.** You need to extract camera poses, RGB images, and LiDAR pointcloud from the original data. 

3. **Implement the dataloader.** You need to create the corresponding `dataparser` script for loading the datasets in NeRFStudio. We will provide our implementation soon.



### Training

To train the model with default parameters, run the following command in the console:

```bash
ns-train lightning_nerf \
    --mixed-precision True \
    --pipeline.model.point-cloud-path path/to/pcd.ply \
    --pipeline.model.frontal-axis x \
    --pipeline.model.init-density-value 10.0 \
    --pipeline.model.density-grid-base-res 256 \
    --pipeline.model.density-log2-hashmap-size 24 \
    --pipeline.model.bg-density-grid-res 32 \
    --pipeline.model.bg-density-log2-hashmap-size 18 \
    --pipeline.model.near-plane 0.01 \
    --pipeline.model.far-plane 10.0 \
    --pipeline.model.vi-mlp-num-layers 3 \
    --pipeline.model.vi-mlp-hidden-size 64 \
    --pipeline.model.vd-mlp-num-layers 2 \
    --pipeline.model.vd-mlp-hidden-size 32 \
    --pipeline.model.color-grid-base-res 128 \
    --pipeline.model.color-grid-max-res 2048 \
    --pipeline.model.color-grid-fpl 2 \
    --pipeline.model.color-grid-num-levels 8 \
    --pipeline.model.bg-color-grid-base-res 32 \
    --pipeline.model.bg-color-grid-max-res 128 \
    --pipeline.model.bg-color-log2-hashmap-size 16 \
    --pipeline.model.alpha-thre 0.02 \
    --pipeline.model.occ-grid-base-res 256 \
    --pipeline.model.occ-grid-num-levels 4 \
    --pipeline.model.occ-num-samples-per-ray 750 \
    --pipeline.model.occ-grid-update-warmup-step 2 \
    --pipeline.model.pdf-num-samples-per-ray 8 \
    --pipeline.model.pdf-samples-warmup-step 1000 \
    --pipeline.model.pdf-samples-fixed-step 3000 \
    --pipeline.model.pdf-samples-fixed-ratio 0.5 \
    --pipeline.model.appearance-embedding-dim 0 \
    ${dataparser_name} \
    --data <data-folder> \
    --orientation-method none
```

You can run `ns-train lightning_nerf --help` to see detailed information of optional arguments.

### Evaluation
To evaluate a model, run the following command in the console:
```bash
ns-eval --load-config=${PATH_TO_CONFIG} --output-path=${PATH_TO_RESULT}.json
```

**Note:** There are differences in the calculation of `SSIM` across NeRF variants. We by default adopt the NeRFStuidio version (i.e., implementation from `torchmetrics`) in our experiments. However, in Table 1 of the manuscript, some results are cited from [DNMP](https://arxiv.org/abs/2307.10776). For fairness, we adopt the DNMP version (i.e., implementation from `skimage`) for comparing `SSIM` in this table. See the discussion [here](https://github.com/DNMP/DNMP/issues/16) for details.

### More
Since Lightning NeRF is integrated into the NeRFStudio project, you may refer to [docs.nerf.studio](https://docs.nerf.studio/) for more functional supports.

