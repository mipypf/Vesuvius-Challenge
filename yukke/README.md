# Vesuvius Challenge - Ink Detection

<https://www.kaggle.com/competitions/vesuvius-challenge-ink-detection/overview>

## Environment

- Docker: 23.0.1
- [rocker](https://github.com/osrf/rocker): 0.2.10
- [off-your-rocker](https://github.com/sloretz/off-your-rocker)

## Setup

Build a docker image.

```shell
make build-docker
pipx install git+https://github.com/sloretz/off-your-rocker --include-deps
```

Download data from [the competition page](https://www.kaggle.com/competitions/vesuvius-challenge-ink-detection/overview).

```shell
$ tree /path_to_input -L 2
/path_to_input
└── vesuvius-challenge-ink-detection
    ├── sample_submission.csv
    ├── test
    └── train
```

Download pre-trained models of 3D-ResNet.

```shell
gdown 12FxrQY2hX-bINbmSrN9q2Z5zJguJhy6C
gdown 1K9oiny9ENYODFxjFBTKdGeoDvOzPu4qQ
mkdir checkpoints
mv r3d18_KM_200ep.pth checkpoints
mv r3d34_KM_200ep.pth checkpoints
```

## Experiments

```shell
make run-docker LOCAL_DATA_PATH=/path_to_input
```

### Preprocess

```shell
python -m inkdet.tools.preprocess
```

### Train

```shell
train <CONFIG_PATH>
```

Run shell scripts in the `scripts` directory to train models for the submission.

### Reference

- <https://github.com/kenshohara/3D-ResNets-PyTorch>
