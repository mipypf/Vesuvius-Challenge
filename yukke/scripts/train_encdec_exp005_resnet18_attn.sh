#!/bin/bash

train configs/encdec_exp005.yaml \
    --valid-id 1 \
    --cfg-options \
    model_params.use_lateral_attention=True

# train configs/encdec_exp005.yaml \
#     --valid-id 2a \
#     --cfg-options \
#         model_params.use_lateral_attention=True

train configs/encdec_exp005.yaml \
    --valid-id 2b \
    --cfg-options \
    model_params.use_lateral_attention=True

# train configs/encdec_exp005.yaml \
#     --valid-id 3 \
#     --cfg-options \
#         model_params.use_lateral_attention=True
