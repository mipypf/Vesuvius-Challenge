#!/bin/bash

train configs/encdec_exp027.yaml \
    --cfg-options train_augmentations.3.p=0 \
    train_augmentations.4.rotate_limit=45

# train configs/encdec_exp027.yaml \
#     --valid-id 2a \
#     --cfg-options train_augmentations.3.p=0 \
#     train_augmentations.4.rotate_limit=45

train configs/encdec_exp027.yaml \
    --valid-id 2b \
    --cfg-options train_augmentations.3.p=0 \
    train_augmentations.4.rotate_limit=45

# train configs/encdec_exp027.yaml \
#     --valid-id 3 \
#     --cfg-options train_augmentations.3.p=0 \
#     train_augmentations.4.rotate_limit=45
