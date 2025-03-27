# Fuse My Cells Challenge
This repository contains the codebase we used for the participation (algorithm name `RWTH-GE`) in the [Fuse My Cells Challenge](https://fusemycells.grand-challenge.org/) held at ISBI 2025 in Houston, USA. The repository was adapted from our previous work on image synthesis and semantic segmentation, which was primarily developed by Dennis Eschweiler and can be found in the following repositories: 

- [Diffusion Models for Image Synthesis](https://github.com/stegmaierj/DiffusionModelsForImageSynthesis)
- [GAN-based Cell Synthesis](https://github.com/stegmaierj/CellSynthesis)
- [Cellpose3D](https://github.com/stegmaierj/Cellpose3D)
- [HarmonicNet](https://github.com/stegmaierj/HarmonicNet)


Please cite the respective papers if you reuse the code for your own projects (see full references below).

## Algorithm Description
We used a 3D CNN to approach the task of reconstructing a missing view for a multi-view fusion of two complementary views in 3D light-sheet microscopy on the provided challenge data set. The architecture is a relatively shallow 3D U-Net style network (see `Source/models/module_UNet3D.py` for details). Our primary strategy was to augment the input image (the respective `*_angle.tif` image) with some additional information that was supposed to facilitate the reconstruction of the `*_fused.tif` ground truth image. We tested the following additional channels:

- Distance Map: In line with our previous work on patch-based image synthesis with GANs [3], we added an input channel that encodes the Euclidean distance to the surface of the specimen. To this end, we computed a binary threshold of the input image and then identified the 3D convex hull of the specimen. Subsequently, a signed Euclidean distance map was computed with a value of `0` at the surface of the specimen, increasing positive values in the interior of the specimen and deacreasing negative values on the outside of the object. 
- Light Map: We tried to estimate an intensity gradient in the foreground regions by comparing the weighted centroid to the centroid, assuming that the direction vector between the two would indicate a bright to dark direction that could be used as an approximated illumination direction. The channel contains a linearly increasing intensity value, starting at the `0` level at the location of the guessed light position and increasing in steps of `1` towards the opposite image border.

In the preliminary performed experiments, however, we could not find a positive impact of the light map and thus discarded it for the final submission. Supplementing the raw input image with the distance map image, on the other hand, consistently yielded superior results compared to solely using the raw image as input. As we perform a patch-based processing, we hypothesize that the distance map channel serves as a positional encoding and thus allows learning global / specimen-dependent intensity patterns despite the patch-based processing.

## Data Preparation
For faster data access during training and inference, we converted all input images into the H5 format. Moreover, distance maps and light maps were precomputed to have them readily available whenever needed. The preprocessing can be performed for a single input image using the script `Source/convert_images.py` with the parameters `input_path` (absolute path to the input image) and `output_path` (output path to store the converted input image).

## Training Details
Training can be performed with the script `Source/train_script.py`. The training requires that the file names of the training, validation and test sets are listed in three separate `*.csv` files. Each line in these files is expected to contain the path to an `*_angle.tif` image and the target image (`*_fused.tif`). For instance the file could look like:

```
Study1_H5/image_19_membrane_angle.h5;Study1_H5/image_19_membrane_fused.h5
Study1_H5/image_20_membrane_angle.h5;Study1_H5/image_20_membrane_fused.h5
Study1_H5/image_25_membrane_angle.h5;Study1_H5/image_25_membrane_fused.h5
...
```

Depending on the number of input channels that should be used during training, the input parameter `image_groups` needs to be adjusted accordingly. Valid options are `data/raw_image`, `data/surface_distance`, `data/light_map`. The number of input channels of the network also has to be adapted accordingly using the `in_channels` parameter. We tested two different loss functions:

1. SSIM + L1: This loss minimizes the sum of the negative SSIM and L1 distance of input and target image.
2. The second loss directly implements the evaluation metric of the Fuse My Cells Challenge ([N_SSIM](https://fusemycells.grand-challenge.org/evaluation-metrics/)) and is also regularized using the L1 distance of input and target image.

In our experiments, we consistently found that optimizing the SSIM + L1 led to better results. Training was performed in a patch-based manner using a patch size of `64x256x256` (ZYX order of the dimensions). We trained separate models for each study (1-5) and image content (nucleus or membrane).

## Application Details
The application of a trained model to a new input image can be performed using the script `Source/apply_script.py`. Input files are again provided in the form of `*.csv` files as detailed in the previous section on training. Due to GPU memory constraints, a processing at the full resolution was not possible in one go and we thus follow a patch-based reconstruction strategy. Overlapping patches are predicted sequentially and assembled to a full-resolution image using a weighted average of overlapping regions. Depending on the training success and the presence of valid foreground structures this approach works well in practice. It has, however, some difficulties in reconstructing large background areas and in turn the result images may exhibit a chessboard-like pattern for some images.

## Challenge Submission
The folder `Docker` contains all files that are used to create a Docker image for the challenge. Based on the template provided by the organizers, we implemented an interface to the apply script mentioned in the previous section. Based on the meta data extracted from the input images, we select the most appropriate model among the pretrained ones. As information about membrane vs. nucleus is always contained in the meta data, this constrains the number of models to select from. If the study number does not match any of the pretrained models, we guess the most appropriate model based on the image size similarity to the images of the training set. Of course, this is a very error-prone guess and most likely a more elaborate selection of the most suitable model would improve our currently suboptimal results. The script `Docker/Build_Docker_ForGC.sh` packs all relevant resources to a Docker image that can be submitted to the FMC challenge. Pretrained model checkpoints are provided in the folder `Docker/resources/weights/Study{1,2,3,4,5}_{Nucleus, Membrane}`.

## References
1. Eschweiler, D., Yilmaz, R., Baumann, M., Laube, I., Roy, R., Jose, A., ... & Stegmaier, J. (2024). Denoising diffusion probabilistic models for generation of realistic fully-annotated microscopy image datasets. PLOS Computational Biology, 20(2), e1011890.
2. Eschweiler, D., Smith, R. S., & Stegmaier, J. (2022). Robust 3D cell segmentation: extending the view of cellpose. In 2022 IEEE International Conference on Image Processing (ICIP) (pp. 191-195).
3. Eschweiler, D., Rethwisch, M., Jarchow, M., Koppers, S., & Stegmaier, J. (2021). 3D fluorescence microscopy data synthesis for segmentation and benchmarking. PLOS ONE, 16(12), e0260509.
4. Eschweiler, D., Rethwisch, M., Koppers, S., & Stegmaier, J. (2021). Spherical harmonics for shape-constrained 3D cell segmentation. In 2021 IEEE 18th International Symposium on Biomedical Imaging (ISBI) (pp. 792-796).