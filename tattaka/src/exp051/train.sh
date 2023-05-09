# python train.py --seed 2022 --model_name resnetrs50 --drop_rate 0.4 --drop_path_rate 0.2 --num_3d_layer 3 --in_chans 5 \
#     --mixup_p 0.5 --mixup_alpha 0.2  --lr 1e-3 --backbone_lr 1e-4  --batch_size 16 --image_size 256 --start_z 15 --end_z -15 --shift_z 2 \
#     --fold 0 --gpus 4 --epochs 40 --logdir resnetrs50_split3d5x7csn_mixup --num_workers 6
# python eval.py --seed 2022 --batch_size 16 --image_size 256 --fold 0 --logdir resnetrs50_split3d5x7csn_mixup --num_workers 16
# python eval_tta_pp.py --seed 2022 --batch_size 16 --image_size 256 --fold 0 --logdir resnetrs50_split3d5x7csn_mixup --num_workers 16
# python make_oof.py --seed 2022 --batch_size 16 --image_size 256 --fold 0 --logdir resnetrs50_split3d5x7csn_mixup --num_workers 16

# python train.py --seed 2024 --model_name resnetrs50 --drop_rate 0.4 --drop_path_rate 0.2 --num_3d_layer 3 --in_chans 5 \
#     --mixup_p 0.5 --mixup_alpha 0.2  --lr 1e-3 --backbone_lr 1e-4  --batch_size 16 --image_size 256 --start_z 15 --end_z -15 --shift_z 2 \
#     --fold 2 --gpus 4 --epochs 40 --logdir resnetrs50_split3d5x7csn_mixup --num_workers 6
# python eval.py --seed 2024 --batch_size 16 --image_size 256 --fold 2 --logdir resnetrs50_split3d5x7csn_mixup --num_workers 16
# python eval_tta_pp.py --seed 2024 --batch_size 16 --image_size 256 --fold 2 --logdir resnetrs50_split3d5x7csn_mixup --num_workers 16
# python make_oof.py --seed 2024 --batch_size 16 --image_size 256 --fold 2 --logdir resnetrs50_split3d5x7csn_mixup --num_workers 16

# python train.py --seed 2023 --model_name resnetrs50 --drop_rate 0.4 --drop_path_rate 0.2 --num_3d_layer 3 --in_chans 5 \
#     --mixup_p 0.5 --mixup_alpha 0.2  --lr 1e-3 --backbone_lr 1e-4  --batch_size 16 --image_size 256 --start_z 15 --end_z -15 --shift_z 2 \
#     --fold 1 --gpus 4 --epochs 40 --logdir resnetrs50_split3d5x7csn_mixup --num_workers 6
# python eval.py --seed 2023 --batch_size 16 --image_size 256 --fold 1 --logdir resnetrs50_split3d5x7csn_mixup --num_workers 16
# python eval_tta_pp.py --seed 2023 --batch_size 16 --image_size 256 --fold 1 --logdir resnetrs50_split3d5x7csn_mixup --num_workers 16
# python make_oof.py --seed 2023 --batch_size 16 --image_size 256 --fold 1 --logdir resnetrs50_split3d5x7csn_mixup --num_workers 16

# python train.py --seed 2025 --model_name resnetrs50 --drop_rate 0.4 --drop_path_rate 0.2 --num_3d_layer 3 --in_chans 5 \
#     --mixup_p 0.5 --mixup_alpha 0.2  --lr 1e-3 --backbone_lr 1e-4  --batch_size 16 --image_size 256 --start_z 15 --end_z -15 --shift_z 2 \
#     --fold 3 --gpus 4 --epochs 40 --logdir resnetrs50_split3d5x7csn_mixup --num_workers 6
# python eval.py --seed 2025 --batch_size 16 --image_size 256 --fold 3 --logdir resnetrs50_split3d5x7csn_mixup --num_workers 16
# python eval_tta_pp.py --seed 2025 --batch_size 16 --image_size 256 --fold 3 --logdir resnetrs50_split3d5x7csn_mixup --num_workers 16
# python make_oof.py --seed 2025 --batch_size 16 --image_size 256 --fold 3 --logdir resnetrs50_split3d5x7csn_mixup --num_workers 16

# python train.py --seed 2026 --model_name resnetrs50 --drop_rate 0.4 --drop_path_rate 0.2 --num_3d_layer 3 --in_chans 5 \
#     --mixup_p 0.5 --mixup_alpha 0.2  --lr 1e-3 --backbone_lr 1e-4  --batch_size 16 --image_size 256 --start_z 15 --end_z -15 --shift_z 2 \
#     --fold 4 --gpus 4 --epochs 40 --logdir resnetrs50_split3d5x7csn_mixup --num_workers 6
# python eval.py --seed 2026 --batch_size 16 --image_size 256 --fold 4 --logdir resnetrs50_split3d5x7csn_mixup --num_workers 16
# python eval_tta_pp.py --seed 2026 --batch_size 16 --image_size 256 --fold 4 --logdir resnetrs50_split3d5x7csn_mixup --num_workers 16
# python make_oof.py --seed 2026 --batch_size 16 --image_size 256 --fold 4 --logdir resnetrs50_split3d5x7csn_mixup --num_workers 16



# python train.py --seed 2022 --model_name resnetrs50 --drop_rate 0.4 --drop_path_rate 0.2 --num_3d_layer 6 --in_chans 3 \
#     --mixup_p 0.5 --mixup_alpha 0.2  --lr 1e-3 --backbone_lr 1e-4  --batch_size 16 --image_size 256 --start_z 19 --end_z -19 --shift_z 2 \
#     --fold 0 --gpus 4 --epochs 40 --logdir resnetrs50_split3d3x9csn_l6_mixup --num_workers 6
# python eval.py --seed 2022 --batch_size 16 --image_size 256 --fold 0 --logdir resnetrs50_split3d3x9csn_l6_mixup --num_workers 16
# python eval_tta_pp.py --seed 2022 --batch_size 16 --image_size 256 --fold 0 --logdir resnetrs50_split3d3x9csn_l6_mixup --num_workers 16
# python make_oof.py --seed 2022 --batch_size 16 --image_size 256 --fold 0 --logdir resnetrs50_split3d3x9csn_l6_mixup --num_workers 16

# python train.py --seed 2024 --model_name resnetrs50 --drop_rate 0.4 --drop_path_rate 0.2 --num_3d_layer 6 --in_chans 3 \
#     --mixup_p 0.5 --mixup_alpha 0.2  --lr 1e-3 --backbone_lr 1e-4  --batch_size 16 --image_size 256 --start_z 19 --end_z -19 --shift_z 2 \
#     --fold 2 --gpus 4 --epochs 40 --logdir resnetrs50_split3d3x9csn_l6_mixup --num_workers 6
# python eval.py --seed 2024 --batch_size 16 --image_size 256 --fold 2 --logdir resnetrs50_split3d3x9csn_l6_mixup --num_workers 16
# python eval_tta_pp.py --seed 2024 --batch_size 16 --image_size 256 --fold 2 --logdir resnetrs50_split3d3x9csn_l6_mixup --num_workers 16
# python make_oof.py --seed 2024 --batch_size 16 --image_size 256 --fold 2 --logdir resnetrs50_split3d3x9csn_l6_mixup --num_workers 16

# python train.py --seed 2023 --model_name resnetrs50 --drop_rate 0.4 --drop_path_rate 0.2 --num_3d_layer 6 --in_chans 3 \
#     --mixup_p 0.5 --mixup_alpha 0.2  --lr 1e-3 --backbone_lr 1e-4  --batch_size 16 --image_size 256 --start_z 19 --end_z -19 --shift_z 2 \
#     --fold 1 --gpus 4 --epochs 40 --logdir resnetrs50_split3d3x9csn_l6_mixup --num_workers 6
# python eval.py --seed 2023 --batch_size 16 --image_size 256 --fold 1 --logdir resnetrs50_split3d3x9csn_l6_mixup --num_workers 16
# python eval_tta_pp.py --seed 2023 --batch_size 16 --image_size 256 --fold 1 --logdir resnetrs50_split3d3x9csn_l6_mixup --num_workers 16
# python make_oof.py --seed 2023 --batch_size 16 --image_size 256 --fold 1 --logdir resnetrs50_split3d3x9csn_l6_mixup --num_workers 16

# python train.py --seed 2025 --model_name resnetrs50 --drop_rate 0.4 --drop_path_rate 0.2 --num_3d_layer 6 --in_chans 3 \
#     --mixup_p 0.5 --mixup_alpha 0.2  --lr 1e-3 --backbone_lr 1e-4  --batch_size 16 --image_size 256 --start_z 19 --end_z -19 --shift_z 2 \
#     --fold 3 --gpus 4 --epochs 40 --logdir resnetrs50_split3d3x9csn_l6_mixup --num_workers 6
# python eval.py --seed 2025 --batch_size 16 --image_size 256 --fold 3 --logdir resnetrs50_split3d3x9csn_l6_mixup --num_workers 16
# python eval_tta_pp.py --seed 2025 --batch_size 16 --image_size 256 --fold 3 --logdir resnetrs50_split3d3x9csn_l6_mixup --num_workers 16
# python make_oof.py --seed 2025 --batch_size 16 --image_size 256 --fold 3 --logdir resnetrs50_split3d3x9csn_l6_mixup --num_workers 16

# python train.py --seed 2026 --model_name resnetrs50 --drop_rate 0.4 --drop_path_rate 0.2 --num_3d_layer 6 --in_chans 3 \
#     --mixup_p 0.5 --mixup_alpha 0.2  --lr 1e-3 --backbone_lr 1e-4  --batch_size 16 --image_size 256 --start_z 19 --end_z -19 --shift_z 2 \
#     --fold 4 --gpus 4 --epochs 40 --logdir resnetrs50_split3d3x9csn_l6_mixup --num_workers 6
# python eval.py --seed 2026 --batch_size 16 --image_size 256 --fold 4 --logdir resnetrs50_split3d3x9csn_l6_mixup --num_workers 16
# python eval_tta_pp.py --seed 2026 --batch_size 16 --image_size 256 --fold 4 --logdir resnetrs50_split3d3x9csn_l6_mixup --num_workers 16
# python make_oof.py --seed 2026 --batch_size 16 --image_size 256 --fold 4 --logdir resnetrs50_split3d3x9csn_l6_mixup --num_workers 16




# python train.py --seed 2022 --model_name resnetrs50 --drop_rate 0.4 --drop_path_rate 0.2 --num_3d_layer 3 --in_chans 5 \
#     --mixup_p 0.5 --mixup_alpha 0.2  --lr 1e-3 --backbone_lr 1e-4  --batch_size 16 --image_size 256 --start_z 15 --end_z -15 --shift_z 2 \
#     --fold 0 --gpus 4 --epochs 25 --logdir resnetrs50_split3d5x7csn_mixup_ep25 --num_workers 6
# python eval_tta_pp.py --seed 2022 --batch_size 16 --image_size 256 --fold 0 --logdir resnetrs50_split3d5x7csn_mixup_ep25 --num_workers 16

# python train.py --seed 2024 --model_name resnetrs50 --drop_rate 0.4 --drop_path_rate 0.2 --num_3d_layer 3 --in_chans 5 \
#     --mixup_p 0.5 --mixup_alpha 0.2  --lr 1e-3 --backbone_lr 1e-4  --batch_size 16 --image_size 256 --start_z 15 --end_z -15 --shift_z 2 \
#     --fold 2 --gpus 4 --epochs 25 --logdir resnetrs50_split3d5x7csn_mixup_ep25 --num_workers 6
# python eval_tta_pp.py --seed 2024 --batch_size 16 --image_size 256 --fold 2 --logdir resnetrs50_split3d5x7csn_mixup_ep25 --num_workers 16


# python train.py --seed 2022 --model_name resnetrs50 --drop_rate 0.4 --drop_path_rate 0.2 --num_3d_layer 6 --in_chans 3 \
#     --mixup_p 0.5 --mixup_alpha 0.2  --lr 1e-3 --backbone_lr 1e-4  --batch_size 16 --image_size 256 --start_z 19 --end_z -19 --shift_z 2 \
#     --fold 0 --gpus 4 --epochs 25 --logdir resnetrs50_split3d3x9csn_l6_mixup_ep25 --num_workers 6
# python eval_tta_pp.py --seed 2022 --batch_size 16 --image_size 256 --fold 0 --logdir resnetrs50_split3d3x9csn_l6_mixup_ep25 --num_workers 16

# python train.py --seed 2024 --model_name resnetrs50 --drop_rate 0.4 --drop_path_rate 0.2 --num_3d_layer 6 --in_chans 3 \
#     --mixup_p 0.5 --mixup_alpha 0.2  --lr 1e-3 --backbone_lr 1e-4  --batch_size 16 --image_size 256 --start_z 19 --end_z -19 --shift_z 2 \
#     --fold 2 --gpus 4 --epochs 25 --logdir resnetrs50_split3d3x9csn_l6_mixup_ep25 --num_workers 6
# python eval_tta_pp.py --seed 2024 --batch_size 16 --image_size 256 --fold 2 --logdir resnetrs50_split3d3x9csn_l6_mixup_ep25 --num_workers 16



# python train.py --seed 2022 --model_name convnext_tiny --drop_rate 0.4 --drop_path_rate 0.2 --num_3d_layer 3 --in_chans 5 \
#     --mixup_p 0.5 --mixup_alpha 0.2  --lr 1e-3 --backbone_lr 1e-4  --batch_size 16 --image_size 256 --start_z 15 --end_z -15 --shift_z 2 \
#     --fold 0 --gpus 4 --epochs 25 --logdir convnext_t_split3d5x7csn_mixup_ep25 --num_workers 6
# python eval_tta_pp.py --seed 2022 --batch_size 16 --image_size 256 --fold 0 --logdir convnext_t_split3d5x7csn_mixup_ep25 --num_workers 16

# python train.py --seed 2024 --model_name convnext_tiny --drop_rate 0.4 --drop_path_rate 0.2 --num_3d_layer 3 --in_chans 5 \
#     --mixup_p 0.5 --mixup_alpha 0.2  --lr 1e-3 --backbone_lr 1e-4  --batch_size 16 --image_size 256 --start_z 15 --end_z -15 --shift_z 2 \
#     --fold 2 --gpus 4 --epochs 25 --logdir convnext_t_split3d5x7csn_mixup_ep25 --num_workers 6
# python eval_tta_pp.py --seed 2024 --batch_size 16 --image_size 256 --fold 2 --logdir convnext_t_split3d5x7csn_mixup_ep25 --num_workers 16


# python train.py --seed 2022 --model_name convnext_tiny --drop_rate 0.4 --drop_path_rate 0.2 --num_3d_layer 6 --in_chans 3 \
#     --mixup_p 0.5 --mixup_alpha 0.2  --lr 1e-3 --backbone_lr 1e-4  --batch_size 16 --image_size 256 --start_z 19 --end_z -19 --shift_z 2 \
#     --fold 0 --gpus 4 --epochs 25 --logdir convnext_t_split3d3x9csn_l6_mixup_ep25 --num_workers 6
# python eval_tta_pp.py --seed 2022 --batch_size 16 --image_size 256 --fold 0 --logdir convnext_t_split3d3x9csn_l6_mixup_ep25 --num_workers 16

# python train.py --seed 2024 --model_name convnext_tiny --drop_rate 0.4 --drop_path_rate 0.2 --num_3d_layer 6 --in_chans 3 \
#     --mixup_p 0.5 --mixup_alpha 0.2  --lr 1e-3 --backbone_lr 1e-4  --batch_size 16 --image_size 256 --start_z 19 --end_z -19 --shift_z 2 \
#     --fold 2 --gpus 4 --epochs 25 --logdir convnext_t_split3d3x9csn_l6_mixup_ep25 --num_workers 6
# python eval_tta_pp.py --seed 2024 --batch_size 16 --image_size 256 --fold 2 --logdir convnext_t_split3d3x9csn_l6_mixup_ep25 --num_workers 16



# python train.py --seed 2022 --model_name convnext_tiny --drop_rate 0.4 --drop_path_rate 0.2 --num_3d_layer 3 --in_chans 5 \
#     --mixup_p 0.5 --mixup_alpha 0.2  --lr 1e-3 --backbone_lr 1e-4  --batch_size 16 --image_size 256 --start_z 15 --end_z -15 --shift_z 2 \
#     --fold 0 --gpus 4 --epochs 40 --logdir convnext_tiny_split3d5x7csn_mixup --num_workers 6
# python eval.py --seed 2022 --batch_size 16 --image_size 256 --fold 0 --logdir convnext_tiny_split3d5x7csn_mixup --num_workers 16
# python eval_tta_pp.py --seed 2022 --batch_size 16 --image_size 256 --fold 0 --logdir convnext_tiny_split3d5x7csn_mixup --num_workers 16
# python make_oof.py --seed 2022 --batch_size 16 --image_size 256 --fold 0 --logdir convnext_tiny_split3d5x7csn_mixup --num_workers 16

# python train.py --seed 2024 --model_name convnext_tiny --drop_rate 0.4 --drop_path_rate 0.2 --num_3d_layer 3 --in_chans 5 \
#     --mixup_p 0.5 --mixup_alpha 0.2  --lr 1e-3 --backbone_lr 1e-4  --batch_size 16 --image_size 256 --start_z 15 --end_z -15 --shift_z 2 \
#     --fold 2 --gpus 4 --epochs 40 --logdir convnext_tiny_split3d5x7csn_mixup --num_workers 6
# python eval.py --seed 2024 --batch_size 16 --image_size 256 --fold 2 --logdir convnext_tiny_split3d5x7csn_mixup --num_workers 16
# python eval_tta_pp.py --seed 2024 --batch_size 16 --image_size 256 --fold 2 --logdir convnext_tiny_split3d5x7csn_mixup --num_workers 16
# python make_oof.py --seed 2024 --batch_size 16 --image_size 256 --fold 2 --logdir convnext_tiny_split3d5x7csn_mixup --num_workers 16

# python train.py --seed 2023 --model_name convnext_tiny --drop_rate 0.4 --drop_path_rate 0.2 --num_3d_layer 3 --in_chans 5 \
#     --mixup_p 0.5 --mixup_alpha 0.2  --lr 1e-3 --backbone_lr 1e-4  --batch_size 16 --image_size 256 --start_z 15 --end_z -15 --shift_z 2 \
#     --fold 1 --gpus 4 --epochs 40 --logdir convnext_tiny_split3d5x7csn_mixup --num_workers 6
# python eval.py --seed 2023 --batch_size 16 --image_size 256 --fold 1 --logdir convnext_tiny_split3d5x7csn_mixup --num_workers 16
# python eval_tta_pp.py --seed 2023 --batch_size 16 --image_size 256 --fold 1 --logdir convnext_tiny_split3d5x7csn_mixup --num_workers 16
# python make_oof.py --seed 2023 --batch_size 16 --image_size 256 --fold 1 --logdir convnext_tiny_split3d5x7csn_mixup --num_workers 16

# python train.py --seed 2025 --model_name convnext_tiny --drop_rate 0.4 --drop_path_rate 0.2 --num_3d_layer 3 --in_chans 5 \
#     --mixup_p 0.5 --mixup_alpha 0.2  --lr 1e-3 --backbone_lr 1e-4  --batch_size 16 --image_size 256 --start_z 15 --end_z -15 --shift_z 2 \
#     --fold 3 --gpus 4 --epochs 40 --logdir convnext_tiny_split3d5x7csn_mixup --num_workers 6
# python eval.py --seed 2025 --batch_size 16 --image_size 256 --fold 3 --logdir convnext_tiny_split3d5x7csn_mixup --num_workers 16
# python eval_tta_pp.py --seed 2025 --batch_size 16 --image_size 256 --fold 3 --logdir convnext_tiny_split3d5x7csn_mixup --num_workers 16
# python make_oof.py --seed 2025 --batch_size 16 --image_size 256 --fold 3 --logdir convnext_tiny_split3d5x7csn_mixup --num_workers 16

# python train.py --seed 2026 --model_name convnext_tiny --drop_rate 0.4 --drop_path_rate 0.2 --num_3d_layer 3 --in_chans 5 \
#     --mixup_p 0.5 --mixup_alpha 0.2  --lr 1e-3 --backbone_lr 1e-4  --batch_size 16 --image_size 256 --start_z 15 --end_z -15 --shift_z 2 \
#     --fold 4 --gpus 4 --epochs 40 --logdir convnext_tiny_split3d5x7csn_mixup --num_workers 6
# python eval.py --seed 2026 --batch_size 16 --image_size 256 --fold 4 --logdir convnext_tiny_split3d5x7csn_mixup --num_workers 16
# python eval_tta_pp.py --seed 2026 --batch_size 16 --image_size 256 --fold 4 --logdir convnext_tiny_split3d5x7csn_mixup --num_workers 16
# python make_oof.py --seed 2026 --batch_size 16 --image_size 256 --fold 4 --logdir convnext_tiny_split3d5x7csn_mixup --num_workers 16



# python train.py --seed 2022 --model_name convnext_tiny --drop_rate 0.4 --drop_path_rate 0.2 --num_3d_layer 6 --in_chans 3 \
#     --mixup_p 0.5 --mixup_alpha 0.2  --lr 1e-3 --backbone_lr 1e-4  --batch_size 16 --image_size 256 --start_z 19 --end_z -19 --shift_z 2 \
#     --fold 0 --gpus 4 --epochs 40 --logdir convnext_tiny_split3d3x9csn_l6_mixup --num_workers 6
# python eval.py --seed 2022 --batch_size 16 --image_size 256 --fold 0 --logdir convnext_tiny_split3d3x9csn_l6_mixup --num_workers 16
# python eval_tta_pp.py --seed 2022 --batch_size 16 --image_size 256 --fold 0 --logdir convnext_tiny_split3d3x9csn_l6_mixup --num_workers 16
# python make_oof.py --seed 2022 --batch_size 16 --image_size 256 --fold 0 --logdir convnext_tiny_split3d3x9csn_l6_mixup --num_workers 16

# python train.py --seed 2024 --model_name convnext_tiny --drop_rate 0.4 --drop_path_rate 0.2 --num_3d_layer 6 --in_chans 3 \
#     --mixup_p 0.5 --mixup_alpha 0.2  --lr 1e-3 --backbone_lr 1e-4  --batch_size 16 --image_size 256 --start_z 19 --end_z -19 --shift_z 2 \
#     --fold 2 --gpus 4 --epochs 40 --logdir convnext_tiny_split3d3x9csn_l6_mixup --num_workers 6
# python eval.py --seed 2024 --batch_size 16 --image_size 256 --fold 2 --logdir convnext_tiny_split3d3x9csn_l6_mixup --num_workers 16
# python eval_tta_pp.py --seed 2024 --batch_size 16 --image_size 256 --fold 2 --logdir convnext_tiny_split3d3x9csn_l6_mixup --num_workers 16
# python make_oof.py --seed 2024 --batch_size 16 --image_size 256 --fold 2 --logdir convnext_tiny_split3d3x9csn_l6_mixup --num_workers 16

# python train.py --seed 2023 --model_name convnext_tiny --drop_rate 0.4 --drop_path_rate 0.2 --num_3d_layer 6 --in_chans 3 \
#     --mixup_p 0.5 --mixup_alpha 0.2  --lr 1e-3 --backbone_lr 1e-4  --batch_size 16 --image_size 256 --start_z 19 --end_z -19 --shift_z 2 \
#     --fold 1 --gpus 4 --epochs 40 --logdir convnext_tiny_split3d3x9csn_l6_mixup --num_workers 6
# python eval.py --seed 2023 --batch_size 16 --image_size 256 --fold 1 --logdir convnext_tiny_split3d3x9csn_l6_mixup --num_workers 16
# python eval_tta_pp.py --seed 2023 --batch_size 16 --image_size 256 --fold 1 --logdir convnext_tiny_split3d3x9csn_l6_mixup --num_workers 16
# python make_oof.py --seed 2023 --batch_size 16 --image_size 256 --fold 1 --logdir convnext_tiny_split3d3x9csn_l6_mixup --num_workers 16

# python train.py --seed 2025 --model_name convnext_tiny --drop_rate 0.4 --drop_path_rate 0.2 --num_3d_layer 6 --in_chans 3 \
#     --mixup_p 0.5 --mixup_alpha 0.2  --lr 1e-3 --backbone_lr 1e-4  --batch_size 16 --image_size 256 --start_z 19 --end_z -19 --shift_z 2 \
#     --fold 3 --gpus 4 --epochs 40 --logdir convnext_tiny_split3d3x9csn_l6_mixup --num_workers 6
# python eval.py --seed 2025 --batch_size 16 --image_size 256 --fold 3 --logdir convnext_tiny_split3d3x9csn_l6_mixup --num_workers 16
# python eval_tta_pp.py --seed 2025 --batch_size 16 --image_size 256 --fold 3 --logdir convnext_tiny_split3d3x9csn_l6_mixup --num_workers 16
# python make_oof.py --seed 2025 --batch_size 16 --image_size 256 --fold 3 --logdir convnext_tiny_split3d3x9csn_l6_mixup --num_workers 16

# python train.py --seed 2026 --model_name convnext_tiny --drop_rate 0.4 --drop_path_rate 0.2 --num_3d_layer 6 --in_chans 3 \
#     --mixup_p 0.5 --mixup_alpha 0.2  --lr 1e-3 --backbone_lr 1e-4  --batch_size 16 --image_size 256 --start_z 19 --end_z -19 --shift_z 2 \
#     --fold 4 --gpus 4 --epochs 40 --logdir convnext_tiny_split3d3x9csn_l6_mixup --num_workers 6
# python eval.py --seed 2026 --batch_size 16 --image_size 256 --fold 4 --logdir convnext_tiny_split3d3x9csn_l6_mixup --num_workers 16
# python eval_tta_pp.py --seed 2026 --batch_size 16 --image_size 256 --fold 4 --logdir convnext_tiny_split3d3x9csn_l6_mixup --num_workers 16
# python make_oof.py --seed 2026 --batch_size 16 --image_size 256 --fold 4 --logdir convnext_tiny_split3d3x9csn_l6_mixup --num_workers 16

