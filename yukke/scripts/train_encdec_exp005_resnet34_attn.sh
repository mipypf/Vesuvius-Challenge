#!/bin/bash

train configs/encdec_exp005.yaml \
    --valid-id 1 \
    --cfg-options \
    model_params.use_lateral_attention=True \
    model_params.encoder_depth=34 checkpoint_path=checkpoints/r3d34_KM_200ep.pth

# train configs/encdec_exp005.yaml \
#     --valid-id 2a \
#     --cfg-options \
#     model_params.use_lateral_attention=True \
#     model_params.encoder_depth=34 checkpoint_path=checkpoints/r3d34_KM_200ep.pth

train configs/encdec_exp005.yaml \
    --valid-id 2b \
    --cfg-options \
    model_params.use_lateral_attention=True \
    model_params.encoder_depth=34 checkpoint_path=checkpoints/r3d34_KM_200ep.pth

# train configs/encdec_exp005.yaml \
#     --valid-id 3 \
#     --cfg-options \
#     model_params.use_lateral_attention=True \
#     model_params.encoder_depth=34 checkpoint_path=checkpoints/r3d34_KM_200ep.pth
