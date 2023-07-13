#!/bin/bash

train configs/encdec_exp008.yaml \
    --valid-id 1 \
    --cfg-options \
    encoder_name=mvit_v2_s \
    epochs=20

# train configs/encdec_exp008.yaml \
#     --valid-id 2a \
#     --cfg-options \
#     encoder_name=mvit_v2_s \
#     epochs=20

train configs/encdec_exp008.yaml \
    --valid-id 2b \
    --cfg-options \
    encoder_name=mvit_v2_s \
    epochs=20

# train configs/encdec_exp008.yaml \
#     --valid-id 3 \
#     --cfg-options \
#     encoder_name=mvit_v2_s \
#     epochs=20
