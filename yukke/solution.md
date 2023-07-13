### Summary

- 3D encoder and 2D/1D encoder

  - 1/2 or 1/4 resolution prediction

  - very simple decoder

### Data preprocessing

- split the 2nd fragment into two image vertically: 4 folds

### Model

- 3D CNN encoder and 2D Encoder
  - encoder: based on @samfc10 's Notebook: [Vesuvius Challenge - 3D ResNet Training](https://www.kaggle.com/code/samfc10/vesuvius-challenge-3d-resnet-training)
    - remove maxpooling after the 1st CNN
    - use attention before reduce D-dim
    - use resnet18 and resnet34
  - decoder
    - a single 2D CNN layer
    - upsample with a nearest interpolation
  - output resolution is downsampled to 1/2. then upsample with a bilinear interpolation
- 3D transformer encoder and linear decoder
  - encoder: use [MViTv2-s](https://pytorch.org/vision/main/models/video_mvit.html) of the PyTorch official implementation and a pre-trained model
    - modify forward function to get each scale output
    - replace MaxPool3d into MaxPool2d for the reproducibility
  - decoder: a single linear and patch expanding to upscale low resolutions 3D images
    - patch expanding is from [Swin-Unet](https://arxiv.org/abs/2105.05537)
  - output resolution is downsampled to 1/2 or 1/4, then upsample with a bilinear interpolation

### Training

- amp
- torch.compile
- label smoothing
- cutout
- cutmix
- mixup
- data augmentation
  - referred @ts 's notebook: [2.5d segmentaion baseline [training]](https://www.kaggle.com/code/tanakar/2-5d-segmentaion-baseline-training)
  - referred @tattaka and @mipypf 's
- patch_size=224 and stride=112
  - stride=75 or stride=56 didn't work

### Inference

- fp16 inference
- stride=75
  - better than stride=112
- ignore edge of output prediction
  - use only the red area of prediction (figure below)
