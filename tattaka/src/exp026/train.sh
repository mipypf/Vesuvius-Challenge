# python train.py --seed 2022 --model_name resnetrs50 --drop_path 0.4 --drop_path_rate 0.2 --num_3d_layers 3 --in_chans 7 \
#     --mixup_p 0.5 --mixup_alpha 0.2  --lr 3e-4  --batch_size 16 --image_size 256 --start_z 8 --end_z -8 --shift_z 2 \
#     --fold 0 --gpus 4 --epochs 25 --logdir resnetrs50_split3d7x7csn_mixup --num_workers 6
# python eval.py --seed 2022 --batch_size 16 --image_size 256 --fold 0 --logdir resnetrs50_split3d7x7csn_mixup --num_workers 16
# python eval_tta.py --seed 2022 --batch_size 16 --image_size 256 --fold 0 --logdir resnetrs50_split3d7x7csn_mixup --num_workers 16

# python train.py --seed 2023 --model_name resnetrs50 --drop_path 0.4 --drop_path_rate 0.2 --num_3d_layers 3 --in_chans 7 \
#     --mixup_p 0.5 --mixup_alpha 0.2  --lr 3e-4  --batch_size 16 --image_size 256 --start_z 8 --end_z -8 --shift_z 2 \
#     --fold 1 --gpus 4 --epochs 25 --logdir resnetrs50_split3d7x7csn_mixup --num_workers 6
# python eval.py --seed 2023 --batch_size 16 --image_size 256 --fold 1 --logdir resnetrs50_split3d7x7csn_mixup --num_workers 16
# python eval_tta.py --seed 2023 --batch_size 16 --image_size 256 --fold 1 --logdir resnetrs50_split3d7x7csn_mixup --num_workers 16

# python train.py --seed 2024 --model_name resnetrs50 --drop_path 0.4 --drop_path_rate 0.2 --num_3d_layers 3 --in_chans 7 \
#     --mixup_p 0.5 --mixup_alpha 0.2  --lr 3e-4  --batch_size 16 --image_size 256 --start_z 8 --end_z -8 --shift_z 2 \
#     --fold 2 --gpus 4 --epochs 25 --logdir resnetrs50_split3d7x7csn_mixup --num_workers 6
# python eval.py --seed 2024 --batch_size 16 --image_size 256 --fold 2 --logdir resnetrs50_split3d7x7csn_mixup --num_workers 16
# python eval_tta.py --seed 2024 --batch_size 16 --image_size 256 --fold 2 --logdir resnetrs50_split3d7x7csn_mixup --num_workers 16

# python train.py --seed 2025 --model_name resnetrs50 --drop_path 0.4 --drop_path_rate 0.2 --num_3d_layers 3 --in_chans 7 \
#     --mixup_p 0.5 --mixup_alpha 0.2  --lr 3e-4  --batch_size 16 --image_size 256 --start_z 8 --end_z -8 --shift_z 2 \
#     --fold 3 --gpus 4 --epochs 25 --logdir resnetrs50_split3d7x7csn_mixup --num_workers 6
# python eval.py --seed 2025 --batch_size 16 --image_size 256 --fold 3 --logdir resnetrs50_split3d7x7csn_mixup --num_workers 16
# python eval_tta.py --seed 2025 --batch_size 16 --image_size 256 --fold 3 --logdir resnetrs50_split3d7x7csn_mixup --num_workers 16

# python train.py --seed 2026 --model_name resnetrs50 --drop_path 0.4 --drop_path_rate 0.2 --num_3d_layers 3 --in_chans 7 \
#     --mixup_p 0.5 --mixup_alpha 0.2  --lr 3e-4  --batch_size 16 --image_size 256 --start_z 8 --end_z -8 --shift_z 2 \
#     --fold 4 --gpus 4 --epochs 25 --logdir resnetrs50_split3d7x7csn_mixup --num_workers 6
# python eval.py --seed 2026 --batch_size 16 --image_size 256 --fold 4 --logdir resnetrs50_split3d7x7csn_mixup --num_workers 16
# python eval_tta.py --seed 2026 --batch_size 16 --image_size 256 --fold 4 --logdir resnetrs50_split3d7x7csn_mixup --num_workers 16


# python train.py --seed 2022 --model_name resnetrs50 --drop_path 0.4 --drop_path_rate 0.2 --num_3d_layers 3 --in_chans 6 \
#     --mixup_p 0.5 --mixup_alpha 0.2  --lr 3e-4  --batch_size 16 --image_size 256 --start_z 11 --end_z -12 --shift_z 2 \
#     --fold 0 --gpus 4 --epochs 25 --logdir resnetrs50_split3d6x7csn_mixup --num_workers 6
# python eval.py --seed 2022 --batch_size 16 --image_size 256 --fold 0 --logdir resnetrs50_split3d6x7csn_mixup --num_workers 16
# python eval_tta.py --seed 2022 --batch_size 16 --image_size 256 --fold 0 --logdir resnetrs50_split3d6x7csn_mixup --num_workers 16

# python train.py --seed 2023 --model_name resnetrs50 --drop_path 0.4 --drop_path_rate 0.2 --num_3d_layers 3 --in_chans 6 \
#     --mixup_p 0.5 --mixup_alpha 0.2  --lr 3e-4  --batch_size 16 --image_size 256 --start_z 11 --end_z -12 --shift_z 2 \
#     --fold 1 --gpus 4 --epochs 25 --logdir resnetrs50_split3d6x7csn_mixup --num_workers 6
# python eval.py --seed 2023 --batch_size 16 --image_size 256 --fold 1 --logdir resnetrs50_split3d6x7csn_mixup --num_workers 16
# python eval_tta.py --seed 2023 --batch_size 16 --image_size 256 --fold 1 --logdir resnetrs50_split3d6x7csn_mixup --num_workers 16

# python train.py --seed 2024 --model_name resnetrs50 --drop_path 0.4 --drop_path_rate 0.2 --num_3d_layers 3 --in_chans 6 \
#     --mixup_p 0.5 --mixup_alpha 0.2  --lr 3e-4  --batch_size 16 --image_size 256 --start_z 11 --end_z -12 --shift_z 2 \
#     --fold 2 --gpus 4 --epochs 25 --logdir resnetrs50_split3d6x7csn_mixup --num_workers 6
# python eval.py --seed 2024 --batch_size 16 --image_size 256 --fold 2 --logdir resnetrs50_split3d6x7csn_mixup --num_workers 16
# python eval_tta.py --seed 2024 --batch_size 16 --image_size 256 --fold 2 --logdir resnetrs50_split3d6x7csn_mixup --num_workers 16

# python train.py --seed 2025 --model_name resnetrs50 --drop_path 0.4 --drop_path_rate 0.2 --num_3d_layers 3 --in_chans 6 \
#     --mixup_p 0.5 --mixup_alpha 0.2  --lr 3e-4  --batch_size 16 --image_size 256 --start_z 11 --end_z -12 --shift_z 2 \
#     --fold 3 --gpus 4 --epochs 25 --logdir resnetrs50_split3d6x7csn_mixup --num_workers 6
# python eval.py --seed 2025 --batch_size 16 --image_size 256 --fold 3 --logdir resnetrs50_split3d6x7csn_mixup --num_workers 16
# python eval_tta.py --seed 2025 --batch_size 16 --image_size 256 --fold 3 --logdir resnetrs50_split3d6x7csn_mixup --num_workers 16

# python train.py --seed 2026 --model_name resnetrs50 --drop_path 0.4 --drop_path_rate 0.2 --num_3d_layers 3 --in_chans 6 \
#     --mixup_p 0.5 --mixup_alpha 0.2  --lr 3e-4  --batch_size 16 --image_size 256 --start_z 11 --end_z -12 --shift_z 2 \
#     --fold 4 --gpus 4 --epochs 25 --logdir resnetrs50_split3d6x7csn_mixup --num_workers 6
# python eval.py --seed 2026 --batch_size 16 --image_size 256 --fold 4 --logdir resnetrs50_split3d6x7csn_mixup --num_workers 16
# python eval_tta.py --seed 2026 --batch_size 16 --image_size 256 --fold 4 --logdir resnetrs50_split3d6x7csn_mixup --num_workers 16



# python train.py --seed 2022 --model_name resnetrs50 --drop_path 0.4 --drop_path_rate 0.2 --num_3d_layers 3 --in_chans 6 \
#     --mixup_p 0.5 --mixup_alpha 0.2  --lr 3e-4  --batch_size 16 --image_size 256 --start_z 14 --end_z -15 --shift_z 2 \
#     --fold 0 --gpus 4 --epochs 25 --logdir resnetrs50_split3d6x6csn_mixup --num_workers 6
# python eval.py --seed 2022 --batch_size 16 --image_size 256 --fold 0 --logdir resnetrs50_split3d6x6csn_mixup --num_workers 16
# python eval_tta.py --seed 2022 --batch_size 16 --image_size 256 --fold 0 --logdir resnetrs50_split3d6x6csn_mixup --num_workers 16

# python train.py --seed 2023 --model_name resnetrs50 --drop_path 0.4 --drop_path_rate 0.2 --num_3d_layers 3 --in_chans 6 \
#     --mixup_p 0.5 --mixup_alpha 0.2  --lr 3e-4  --batch_size 16 --image_size 256 --start_z 14 --end_z -15 --shift_z 2 \
#     --fold 1 --gpus 4 --epochs 25 --logdir resnetrs50_split3d6x6csn_mixup --num_workers 6
# python eval.py --seed 2023 --batch_size 16 --image_size 256 --fold 1 --logdir resnetrs50_split3d6x6csn_mixup --num_workers 16
# python eval_tta.py --seed 2023 --batch_size 16 --image_size 256 --fold 1 --logdir resnetrs50_split3d6x6csn_mixup --num_workers 16

# python train.py --seed 2024 --model_name resnetrs50 --drop_path 0.4 --drop_path_rate 0.2 --num_3d_layers 3 --in_chans 6 \
#     --mixup_p 0.5 --mixup_alpha 0.2  --lr 3e-4  --batch_size 16 --image_size 256 --start_z 14 --end_z -15 --shift_z 2 \
#     --fold 2 --gpus 4 --epochs 25 --logdir resnetrs50_split3d6x6csn_mixup --num_workers 6
# python eval.py --seed 2024 --batch_size 16 --image_size 256 --fold 2 --logdir resnetrs50_split3d6x6csn_mixup --num_workers 16
# python eval_tta.py --seed 2024 --batch_size 16 --image_size 256 --fold 2 --logdir resnetrs50_split3d6x6csn_mixup --num_workers 16

# python train.py --seed 2025 --model_name resnetrs50 --drop_path 0.4 --drop_path_rate 0.2 --num_3d_layers 3 --in_chans 6 \
#     --mixup_p 0.5 --mixup_alpha 0.2  --lr 3e-4  --batch_size 16 --image_size 256 --start_z 14 --end_z -15 --shift_z 2 \
#     --fold 3 --gpus 4 --epochs 25 --logdir resnetrs50_split3d6x6csn_mixup --num_workers 6
# python eval.py --seed 2025 --batch_size 16 --image_size 256 --fold 3 --logdir resnetrs50_split3d6x6csn_mixup --num_workers 16
# python eval_tta.py --seed 2025 --batch_size 16 --image_size 256 --fold 3 --logdir resnetrs50_split3d6x6csn_mixup --num_workers 16

# python train.py --seed 2026 --model_name resnetrs50 --drop_path 0.4 --drop_path_rate 0.2 --num_3d_layers 3 --in_chans 6 \
#     --mixup_p 0.5 --mixup_alpha 0.2  --lr 3e-4  --batch_size 16 --image_size 256 --start_z 14 --end_z -15 --shift_z 2 \
#     --fold 4 --gpus 4 --epochs 25 --logdir resnetrs50_split3d6x6csn_mixup --num_workers 6
# python eval.py --seed 2026 --batch_size 16 --image_size 256 --fold 4 --logdir resnetrs50_split3d6x6csn_mixup --num_workers 16
# python eval_tta.py --seed 2026 --batch_size 16 --image_size 256 --fold 4 --logdir resnetrs50_split3d6x6csn_mixup --num_workers 16


# python train.py --seed 2022 --model_name resnetrs50 --drop_path 0.4 --drop_path_rate 0.2 --num_3d_layers 3 --in_chans 5 \
#     --mixup_p 0.5 --mixup_alpha 0.2  --lr 3e-4  --batch_size 16 --image_size 256 --start_z 15 --end_z -15 --shift_z 2 \
#     --fold 0 --gpus 4 --epochs 25 --logdir resnetrs50_split3d5x7csn_mixup --num_workers 6
# python eval.py --seed 2022 --batch_size 16 --image_size 256 --fold 0 --logdir resnetrs50_split3d5x7csn_mixup --num_workers 16
# python eval_tta.py --seed 2022 --batch_size 16 --image_size 256 --fold 0 --logdir resnetrs50_split3d5x7csn_mixup --num_workers 16

# python train.py --seed 2023 --model_name resnetrs50 --drop_path 0.4 --drop_path_rate 0.2 --num_3d_layers 3 --in_chans 5 \
#     --mixup_p 0.5 --mixup_alpha 0.2  --lr 3e-4  --batch_size 16 --image_size 256 --start_z 15 --end_z -15 --shift_z 2 \
#     --fold 1 --gpus 4 --epochs 25 --logdir resnetrs50_split3d5x7csn_mixup --num_workers 6
# python eval.py --seed 2023 --batch_size 16 --image_size 256 --fold 1 --logdir resnetrs50_split3d5x7csn_mixup --num_workers 16
# python eval_tta.py --seed 2023 --batch_size 16 --image_size 256 --fold 1 --logdir resnetrs50_split3d5x7csn_mixup --num_workers 16

# python train.py --seed 2024 --model_name resnetrs50 --drop_path 0.4 --drop_path_rate 0.2 --num_3d_layers 3 --in_chans 5 \
#     --mixup_p 0.5 --mixup_alpha 0.2  --lr 3e-4  --batch_size 16 --image_size 256 --start_z 15 --end_z -15 --shift_z 2 \
#     --fold 2 --gpus 4 --epochs 25 --logdir resnetrs50_split3d5x7csn_mixup --num_workers 6
# python eval.py --seed 2024 --batch_size 16 --image_size 256 --fold 2 --logdir resnetrs50_split3d5x7csn_mixup --num_workers 16
# python eval_tta.py --seed 2024 --batch_size 16 --image_size 256 --fold 2 --logdir resnetrs50_split3d5x7csn_mixup --num_workers 16

# python train.py --seed 2025 --model_name resnetrs50 --drop_path 0.4 --drop_path_rate 0.2 --num_3d_layers 3 --in_chans 5 \
#     --mixup_p 0.5 --mixup_alpha 0.2  --lr 3e-4  --batch_size 16 --image_size 256 --start_z 15 --end_z -15 --shift_z 2 \
#     --fold 3 --gpus 4 --epochs 25 --logdir resnetrs50_split3d5x7csn_mixup --num_workers 6
# python eval.py --seed 2025 --batch_size 16 --image_size 256 --fold 3 --logdir resnetrs50_split3d5x7csn_mixup --num_workers 16
# python eval_tta.py --seed 2025 --batch_size 16 --image_size 256 --fold 3 --logdir resnetrs50_split3d5x7csn_mixup --num_workers 16

# python train.py --seed 2026 --model_name resnetrs50 --drop_path 0.4 --drop_path_rate 0.2 --num_3d_layers 3 --in_chans 5 \
#     --mixup_p 0.5 --mixup_alpha 0.2  --lr 3e-4  --batch_size 16 --image_size 256 --start_z 15 --end_z -15 --shift_z 2 \
#     --fold 4 --gpus 4 --epochs 25 --logdir resnetrs50_split3d5x7csn_mixup --num_workers 6
# python eval.py --seed 2026 --batch_size 16 --image_size 256 --fold 4 --logdir resnetrs50_split3d5x7csn_mixup --num_workers 16
# python eval_tta.py --seed 2026 --batch_size 16 --image_size 256 --fold 4 --logdir resnetrs50_split3d5x7csn_mixup --num_workers 16


# python train.py --seed 2022 --model_name resnetrs50 --drop_path 0.4 --drop_path_rate 0.2 --num_3d_layers 3 --in_chans 5 \
#     --mixup_p 0.5 --mixup_alpha 0.2  --lr 3e-4  --batch_size 16 --image_size 256 --start_z 15 --end_z -15 --shift_z 4 \
#     --fold 0 --gpus 4 --epochs 25 --logdir resnetrs50_split3d5x7csn_s4_mixup --num_workers 6
# python eval.py --seed 2022 --batch_size 16 --image_size 256 --fold 0 --logdir resnetrs50_split3d5x7csn_s4_mixup --num_workers 16
# python eval_tta.py --seed 2022 --batch_size 16 --image_size 256 --fold 0 --logdir resnetrs50_split3d5x7csn_s4_mixup --num_workers 16

# python train.py --seed 2023 --model_name resnetrs50 --drop_path 0.4 --drop_path_rate 0.2 --num_3d_layers 3 --in_chans 5 \
#     --mixup_p 0.5 --mixup_alpha 0.2  --lr 3e-4  --batch_size 16 --image_size 256 --start_z 15 --end_z -15 --shift_z 4 \
#     --fold 1 --gpus 4 --epochs 25 --logdir resnetrs50_split3d5x7csn_s4_mixup --num_workers 6
# python eval.py --seed 2023 --batch_size 16 --image_size 256 --fold 1 --logdir resnetrs50_split3d5x7csn_s4_mixup --num_workers 16
# python eval_tta.py --seed 2023 --batch_size 16 --image_size 256 --fold 1 --logdir resnetrs50_split3d5x7csn_s4_mixup --num_workers 16

# python train.py --seed 2024 --model_name resnetrs50 --drop_path 0.4 --drop_path_rate 0.2 --num_3d_layers 3 --in_chans 5 \
#     --mixup_p 0.5 --mixup_alpha 0.2  --lr 3e-4  --batch_size 16 --image_size 256 --start_z 15 --end_z -15 --shift_z 4 \
#     --fold 2 --gpus 4 --epochs 25 --logdir resnetrs50_split3d5x7csn_s4_mixup --num_workers 6
# python eval.py --seed 2024 --batch_size 16 --image_size 256 --fold 2 --logdir resnetrs50_split3d5x7csn_s4_mixup --num_workers 16
# python eval_tta.py --seed 2024 --batch_size 16 --image_size 256 --fold 2 --logdir resnetrs50_split3d5x7csn_s4_mixup --num_workers 16

# python train.py --seed 2025 --model_name resnetrs50 --drop_path 0.4 --drop_path_rate 0.2 --num_3d_layers 3 --in_chans 5 \
#     --mixup_p 0.5 --mixup_alpha 0.2  --lr 3e-4  --batch_size 16 --image_size 256 --start_z 15 --end_z -15 --shift_z 4 \
#     --fold 3 --gpus 4 --epochs 25 --logdir resnetrs50_split3d5x7csn_s4_mixup --num_workers 6
# python eval.py --seed 2025 --batch_size 16 --image_size 256 --fold 3 --logdir resnetrs50_split3d5x7csn_s4_mixup --num_workers 16
# python eval_tta.py --seed 2025 --batch_size 16 --image_size 256 --fold 3 --logdir resnetrs50_split3d5x7csn_s4_mixup --num_workers 16

# python train.py --seed 2026 --model_name resnetrs50 --drop_path 0.4 --drop_path_rate 0.2 --num_3d_layers 3 --in_chans 5 \
#     --mixup_p 0.5 --mixup_alpha 0.2  --lr 3e-4  --batch_size 16 --image_size 256 --start_z 15 --end_z -15 --shift_z 4 \
#     --fold 4 --gpus 4 --epochs 25 --logdir resnetrs50_split3d5x7csn_s4_mixup --num_workers 6
# python eval.py --seed 2026 --batch_size 16 --image_size 256 --fold 4 --logdir resnetrs50_split3d5x7csn_s4_mixup --num_workers 16
# python eval_tta.py --seed 2026 --batch_size 16 --image_size 256 --fold 4 --logdir resnetrs50_split3d5x7csn_s4_mixup --num_workers 16


# python train.py --seed 2022 --model_name resnetrs50 --drop_path 0.4 --drop_path_rate 0.2 --num_3d_layers 3 --in_chans 5 \
#     --mixup_p 0.5 --mixup_alpha 0.2  --lr 3e-4  --batch_size 16 --image_size 256 --start_z 10 --end_z -10 --shift_z 2 \
#     --fold 0 --gpus 4 --epochs 25 --logdir resnetrs50_split3d5x9csn_mixup --num_workers 6
# python eval.py --seed 2022 --batch_size 16 --image_size 256 --fold 0 --logdir resnetrs50_split3d5x9csn_mixup --num_workers 16
# python eval_tta.py --seed 2022 --batch_size 16 --image_size 256 --fold 0 --logdir resnetrs50_split3d5x9csn_mixup --num_workers 16

# python train.py --seed 2023 --model_name resnetrs50 --drop_path 0.4 --drop_path_rate 0.2 --num_3d_layers 3 --in_chans 5 \
#     --mixup_p 0.5 --mixup_alpha 0.2  --lr 3e-4  --batch_size 16 --image_size 256 --start_z 10 --end_z -10 --shift_z 2 \
#     --fold 1 --gpus 4 --epochs 25 --logdir resnetrs50_split3d5x9csn_mixup --num_workers 6
# python eval.py --seed 2023 --batch_size 16 --image_size 256 --fold 1 --logdir resnetrs50_split3d5x9csn_mixup --num_workers 16
# python eval_tta.py --seed 2023 --batch_size 16 --image_size 256 --fold 1 --logdir resnetrs50_split3d5x9csn_mixup --num_workers 16

# python train.py --seed 2024 --model_name resnetrs50 --drop_path 0.4 --drop_path_rate 0.2 --num_3d_layers 3 --in_chans 5 \
#     --mixup_p 0.5 --mixup_alpha 0.2  --lr 3e-4  --batch_size 16 --image_size 256 --start_z 10 --end_z -10 --shift_z 2 \
#     --fold 2 --gpus 4 --epochs 25 --logdir resnetrs50_split3d5x9csn_mixup --num_workers 6
# python eval.py --seed 2024 --batch_size 16 --image_size 256 --fold 2 --logdir resnetrs50_split3d5x9csn_mixup --num_workers 16
# python eval_tta.py --seed 2024 --batch_size 16 --image_size 256 --fold 2 --logdir resnetrs50_split3d5x9csn_mixup --num_workers 16

# python train.py --seed 2025 --model_name resnetrs50 --drop_path 0.4 --drop_path_rate 0.2 --num_3d_layers 3 --in_chans 5 \
#     --mixup_p 0.5 --mixup_alpha 0.2  --lr 3e-4  --batch_size 16 --image_size 256 --start_z 10 --end_z -10 --shift_z 2 \
#     --fold 3 --gpus 4 --epochs 25 --logdir resnetrs50_split3d5x9csn_mixup --num_workers 6
# python eval.py --seed 2025 --batch_size 16 --image_size 256 --fold 3 --logdir resnetrs50_split3d5x9csn_mixup --num_workers 16
# python eval_tta.py --seed 2025 --batch_size 16 --image_size 256 --fold 3 --logdir resnetrs50_split3d5x9csn_mixup --num_workers 16

# python train.py --seed 2026 --model_name resnetrs50 --drop_path 0.4 --drop_path_rate 0.2 --num_3d_layers 3 --in_chans 5 \
#     --mixup_p 0.5 --mixup_alpha 0.2  --lr 3e-4  --batch_size 16 --image_size 256 --start_z 10 --end_z -10 --shift_z 2 \
#     --fold 4 --gpus 4 --epochs 25 --logdir resnetrs50_split3d5x9csn_mixup --num_workers 6
# python eval.py --seed 2026 --batch_size 16 --image_size 256 --fold 4 --logdir resnetrs50_split3d5x9csn_mixup --num_workers 16
# python eval_tta.py --seed 2026 --batch_size 16 --image_size 256 --fold 4 --logdir resnetrs50_split3d5x9csn_mixup --num_workers 16


# python train.py --seed 2022 --model_name resnetrs50 --drop_path 0.4 --drop_path_rate 0.2 --num_3d_layers 3 --in_chans 4 \
#     --mixup_p 0.5 --mixup_alpha 0.2  --lr 3e-4  --batch_size 16 --image_size 256 --start_z 14 --end_z -15 --shift_z 2 \
#     --fold 0 --gpus 4 --epochs 25 --logdir resnetrs50_split3d4x9csn_mixup --num_workers 6
# python eval.py --seed 2022 --batch_size 16 --image_size 256 --fold 0 --logdir resnetrs50_split3d4x9csn_mixup --num_workers 16
# python eval_tta.py --seed 2022 --batch_size 16 --image_size 256 --fold 0 --logdir resnetrs50_split3d4x9csn_mixup --num_workers 16

# python train.py --seed 2023 --model_name resnetrs50 --drop_path 0.4 --drop_path_rate 0.2 --num_3d_layers 3 --in_chans 4 \
#     --mixup_p 0.5 --mixup_alpha 0.2  --lr 3e-4  --batch_size 16 --image_size 256 --start_z 14 --end_z -15 --shift_z 2 \
#     --fold 1 --gpus 4 --epochs 25 --logdir resnetrs50_split3d4x9csn_mixup --num_workers 6
# python eval.py --seed 2023 --batch_size 16 --image_size 256 --fold 1 --logdir resnetrs50_split3d4x9csn_mixup --num_workers 16
# python eval_tta.py --seed 2023 --batch_size 16 --image_size 256 --fold 1 --logdir resnetrs50_split3d4x9csn_mixup --num_workers 16

# python train.py --seed 2024 --model_name resnetrs50 --drop_path 0.4 --drop_path_rate 0.2 --num_3d_layers 3 --in_chans 4 \
#     --mixup_p 0.5 --mixup_alpha 0.2  --lr 3e-4  --batch_size 16 --image_size 256 --start_z 14 --end_z -15 --shift_z 2 \
#     --fold 2 --gpus 4 --epochs 25 --logdir resnetrs50_split3d4x9csn_mixup --num_workers 6
# python eval.py --seed 2024 --batch_size 16 --image_size 256 --fold 2 --logdir resnetrs50_split3d4x9csn_mixup --num_workers 16
# python eval_tta.py --seed 2024 --batch_size 16 --image_size 256 --fold 2 --logdir resnetrs50_split3d4x9csn_mixup --num_workers 16

# python train.py --seed 2025 --model_name resnetrs50 --drop_path 0.4 --drop_path_rate 0.2 --num_3d_layers 3 --in_chans 4 \
#     --mixup_p 0.5 --mixup_alpha 0.2  --lr 3e-4  --batch_size 16 --image_size 256 --start_z 14 --end_z -15 --shift_z 2 \
#     --fold 3 --gpus 4 --epochs 25 --logdir resnetrs50_split3d4x9csn_mixup --num_workers 6
# python eval.py --seed 2025 --batch_size 16 --image_size 256 --fold 3 --logdir resnetrs50_split3d4x9csn_mixup --num_workers 16
# python eval_tta.py --seed 2025 --batch_size 16 --image_size 256 --fold 3 --logdir resnetrs50_split3d4x9csn_mixup --num_workers 16

# python train.py --seed 2026 --model_name resnetrs50 --drop_path 0.4 --drop_path_rate 0.2 --num_3d_layers 3 --in_chans 4 \
#     --mixup_p 0.5 --mixup_alpha 0.2  --lr 3e-4  --batch_size 16 --image_size 256 --start_z 14 --end_z -15 --shift_z 2 \
#     --fold 4 --gpus 4 --epochs 25 --logdir resnetrs50_split3d4x9csn_mixup --num_workers 6
# python eval.py --seed 2026 --batch_size 16 --image_size 256 --fold 4 --logdir resnetrs50_split3d4x9csn_mixup --num_workers 16
# python eval_tta.py --seed 2026 --batch_size 16 --image_size 256 --fold 4 --logdir resnetrs50_split3d4x9csn_mixup --num_workers 16


python train.py --seed 2022 --model_name resnetrs50 --drop_path 0.4 --drop_path_rate 0.2 --num_3d_layers 3 --in_chans 3 \
    --mixup_p 0.5 --mixup_alpha 0.2  --lr 3e-4  --batch_size 16 --image_size 256 --start_z 19 --end_z -19 --shift_z 2 \
    --fold 0 --gpus 4 --epochs 25 --logdir resnetrs50_split3d3x9csn_mixup --num_workers 6
python eval.py --seed 2022 --batch_size 16 --image_size 256 --fold 0 --logdir resnetrs50_split3d3x9csn_mixup --num_workers 16
python eval_tta.py --seed 2022 --batch_size 16 --image_size 256 --fold 0 --logdir resnetrs50_split3d3x9csn_mixup --num_workers 16

python train.py --seed 2023 --model_name resnetrs50 --drop_path 0.4 --drop_path_rate 0.2 --num_3d_layers 3 --in_chans 3 \
    --mixup_p 0.5 --mixup_alpha 0.2  --lr 3e-4  --batch_size 16 --image_size 256 --start_z 19 --end_z -19 --shift_z 2 \
    --fold 1 --gpus 4 --epochs 25 --logdir resnetrs50_split3d3x9csn_mixup --num_workers 6
python eval.py --seed 2023 --batch_size 16 --image_size 256 --fold 1 --logdir resnetrs50_split3d3x9csn_mixup --num_workers 16
python eval_tta.py --seed 2023 --batch_size 16 --image_size 256 --fold 1 --logdir resnetrs50_split3d3x9csn_mixup --num_workers 16

python train.py --seed 2024 --model_name resnetrs50 --drop_path 0.4 --drop_path_rate 0.2 --num_3d_layers 3 --in_chans 3 \
    --mixup_p 0.5 --mixup_alpha 0.2  --lr 3e-4  --batch_size 16 --image_size 256 --start_z 19 --end_z -19 --shift_z 2 \
    --fold 2 --gpus 4 --epochs 25 --logdir resnetrs50_split3d3x9csn_mixup --num_workers 6
python eval.py --seed 2024 --batch_size 16 --image_size 256 --fold 2 --logdir resnetrs50_split3d3x9csn_mixup --num_workers 16
python eval_tta.py --seed 2024 --batch_size 16 --image_size 256 --fold 2 --logdir resnetrs50_split3d3x9csn_mixup --num_workers 16

python train.py --seed 2025 --model_name resnetrs50 --drop_path 0.4 --drop_path_rate 0.2 --num_3d_layers 3 --in_chans 3 \
    --mixup_p 0.5 --mixup_alpha 0.2  --lr 3e-4  --batch_size 16 --image_size 256 --start_z 19 --end_z -19 --shift_z 2 \
    --fold 3 --gpus 4 --epochs 25 --logdir resnetrs50_split3d3x9csn_mixup --num_workers 6
python eval.py --seed 2025 --batch_size 16 --image_size 256 --fold 3 --logdir resnetrs50_split3d3x9csn_mixup --num_workers 16
python eval_tta.py --seed 2025 --batch_size 16 --image_size 256 --fold 3 --logdir resnetrs50_split3d3x9csn_mixup --num_workers 16

python train.py --seed 2026 --model_name resnetrs50 --drop_path 0.4 --drop_path_rate 0.2 --num_3d_layers 3 --in_chans 3 \
    --mixup_p 0.5 --mixup_alpha 0.2  --lr 3e-4  --batch_size 16 --image_size 256 --start_z 19 --end_z -19 --shift_z 2 \
    --fold 4 --gpus 4 --epochs 25 --logdir resnetrs50_split3d3x9csn_mixup --num_workers 6
python eval.py --seed 2026 --batch_size 16 --image_size 256 --fold 4 --logdir resnetrs50_split3d3x9csn_mixup --num_workers 16
python eval_tta.py --seed 2026 --batch_size 16 --image_size 256 --fold 4 --logdir resnetrs50_split3d3x9csn_mixup --num_workers 16


# on train.py --seed 2022 --model_name resnetrs50 --drop_path 0.4 --drop_path_rate 0.2 --num_3d_layers 3 --in_chans 7 \
#     --mixup_p 0.5 --mixup_alpha 0.2  --lr 3e-4  --batch_size 16 --image_size 256 --start_z 15 --end_z -15 --shift_z 2 \
#     --fold 0 --gpus 4 --epochs 25 --logdir resnetrs50_split3d7x5csn_mixup --num_workers 6
# python eval.py --seed 2022 --batch_size 16 --image_size 256 --fold 0 --logdir resnetrs50_split3d7x5csn_mixup --num_workers 16
# python eval_tta.py --seed 2022 --batch_size 16 --image_size 256 --fold 0 --logdir resnetrs50_split3d7x5csn_mixup --num_workers 16

# python train.py --seed 2023 --model_name resnetrs50 --drop_path 0.4 --drop_path_rate 0.2 --num_3d_layers 3 --in_chans 7 \
#     --mixup_p 0.5 --mixup_alpha 0.2  --lr 3e-4  --batch_size 16 --image_size 256 --start_z 15 --end_z -15 --shift_z 2 \
#     --fold 1 --gpus 4 --epochs 25 --logdir resnetrs50_split3d7x5csn_mixup --num_workers 6
# python eval.py --seed 2023 --batch_size 16 --image_size 256 --fold 1 --logdir resnetrs50_split3d7x5csn_mixup --num_workers 16
# python eval_tta.py --seed 2023 --batch_size 16 --image_size 256 --fold 1 --logdir resnetrs50_split3d7x5csn_mixup --num_workers 16

# python train.py --seed 2024 --model_name resnetrs50 --drop_path 0.4 --drop_path_rate 0.2 --num_3d_layers 3 --in_chans 7 \
#     --mixup_p 0.5 --mixup_alpha 0.2  --lr 3e-4  --batch_size 16 --image_size 256 --start_z 15 --end_z -15 --shift_z 2 \
#     --fold 2 --gpus 4 --epochs 25 --logdir resnetrs50_split3d7x5csn_mixup --num_workers 6
# python eval.py --seed 2024 --batch_size 16 --image_size 256 --fold 2 --logdir resnetrs50_split3d7x5csn_mixup --num_workers 16
# python eval_tta.py --seed 2024 --batch_size 16 --image_size 256 --fold 2 --logdir resnetrs50_split3d7x5csn_mixup --num_workers 16

# python train.py --seed 2025 --model_name resnetrs50 --drop_path 0.4 --drop_path_rate 0.2 --num_3d_layers 3 --in_chans 7 \
#     --mixup_p 0.5 --mixup_alpha 0.2  --lr 3e-4  --batch_size 16 --image_size 256 --start_z 15 --end_z -15 --shift_z 2 \
#     --fold 3 --gpus 4 --epochs 25 --logdir resnetrs50_split3d7x5csn_mixup --num_workers 6
# python eval.py --seed 2025 --batch_size 16 --image_size 256 --fold 3 --logdir resnetrs50_split3d7x5csn_mixup --num_workers 16
# python eval_tta.py --seed 2025 --batch_size 16 --image_size 256 --fold 3 --logdir resnetrs50_split3d7x5csn_mixup --num_workers 16

# python train.py --seed 2026 --model_name resnetrs50 --drop_path 0.4 --drop_path_rate 0.2 --num_3d_layers 3 --in_chans 7 \
#     --mixup_p 0.5 --mixup_alpha 0.2  --lr 3e-4  --batch_size 16 --image_size 256 --start_z 15 --end_z -15 --shift_z 2 \
#     --fold 4 --gpus 4 --epochs 25 --logdir resnetrs50_split3d7x5csn_mixup --num_workers 6
# python eval.py --seed 2026 --batch_size 16 --image_size 256 --fold 4 --logdir resnetrs50_split3d7x5csn_mixup --num_workers 16
# python eval_tta.py --seed 2026 --batch_size 16 --image_size 256 --fold 4 --logdir resnetrs50_split3d7x5csn_mixup --num_workers 16


