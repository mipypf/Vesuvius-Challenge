python train.py --seed 2022 --model_name resnetrs50 --drop_rate 0.4 --drop_path_rate 0.2 --num_3d_layer 3 --in_chans 5 \
    --mixup_p 0.5 --mixup_alpha 0.2 --lr 1e-3  --batch_size 16 --image_size 256 --start_z 15 --end_z -15 --shift_z 2 \
    --fold 0 --gpus 4 --epochs 30 --logdir resnetrs50_split3d5x7csn_mixup_ep30 --num_workers 6
python eval_tta_fp16.py --seed 2022 --batch_size 16 --image_size 256 --fold 0 --logdir resnetrs50_split3d5x7csn_mixup_ep30 --num_workers 16

python train.py --seed 2023 --model_name resnetrs50 --drop_rate 0.4 --drop_path_rate 0.2 --num_3d_layer 3 --in_chans 5 \
    --mixup_p 0.5 --mixup_alpha 0.2 --lr 1e-3  --batch_size 16 --image_size 256 --start_z 15 --end_z -15 --shift_z 2 \
    --fold 1 --gpus 4 --epochs 30 --logdir resnetrs50_split3d5x7csn_mixup_ep30 --num_workers 6
python eval_tta_fp16.py --seed 2022 --batch_size 16 --image_size 256 --fold 1 --logdir resnetrs50_split3d5x7csn_mixup_ep30 --num_workers 16

python train.py --seed 2024 --model_name resnetrs50 --drop_rate 0.4 --drop_path_rate 0.2 --num_3d_layer 3 --in_chans 5 \
    --mixup_p 0.5 --mixup_alpha 0.2 --lr 1e-3   --batch_size 16 --image_size 256 --start_z 15 --end_z -15 --shift_z 2 \
    --fold 2 --gpus 4 --epochs 30 --logdir resnetrs50_split3d5x7csn_mixup_ep30 --num_workers 6
python eval_tta_fp16.py --seed 2024 --batch_size 16 --image_size 256 --fold 2 --logdir resnetrs50_split3d5x7csn_mixup_ep30 --num_workers 16

python train.py --seed 2025 --model_name resnetrs50 --drop_rate 0.4 --drop_path_rate 0.2 --num_3d_layer 3 --in_chans 5 \
    --mixup_p 0.5 --mixup_alpha 0.2 --lr 1e-3  --batch_size 16 --image_size 256 --start_z 15 --end_z -15 --shift_z 2 \
    --fold 3 --gpus 4 --epochs 30 --logdir resnetrs50_split3d5x7csn_mixup_ep30 --num_workers 6
python eval_tta_fp16.py --seed 2025 --batch_size 16 --image_size 256 --fold 3 --logdir resnetrs50_split3d5x7csn_mixup_ep30 --num_workers 16

python train.py --seed 2026 --model_name resnetrs50 --drop_rate 0.4 --drop_path_rate 0.2 --num_3d_layer 3 --in_chans 5 \
    --mixup_p 0.5 --mixup_alpha 0.2 --lr 1e-3  --batch_size 16 --image_size 256 --start_z 15 --end_z -15 --shift_z 2 \
    --fold 4 --gpus 4 --epochs 30 --logdir resnetrs50_split3d5x7csn_mixup_ep30 --num_workers 6
python eval_tta_fp16.py --seed 2026 --batch_size 16 --image_size 256 --fold 4 --logdir resnetrs50_split3d5x7csn_mixup_ep30 --num_workers 16



python train.py --seed 3022 --model_name resnetrs50 --drop_rate 0.4 --drop_path_rate 0.2 --num_3d_layer 6 --in_chans 3 \
    --mixup_p 0.5 --mixup_alpha 0.2 --lr 1e-3   --batch_size 16 --image_size 256 --start_z 19 --end_z -19 --shift_z 2 \
    --fold 0 --gpus 4 --epochs 30 --logdir resnetrs50_split3d3x9csn_l6_mixup_ep30 --num_workers 6
python eval_tta_fp16.py --seed 2022 --batch_size 16 --image_size 256 --fold 0 --logdir resnetrs50_split3d3x9csn_l6_mixup_ep30 --num_workers 16

python train.py --seed 3023 --model_name resnetrs50 --drop_rate 0.4 --drop_path_rate 0.2 --num_3d_layer 6 --in_chans 3 \
    --mixup_p 0.5 --mixup_alpha 0.2 --lr 1e-3   --batch_size 16 --image_size 256 --start_z 19 --end_z -19 --shift_z 2 \
    --fold 1 --gpus 4 --epochs 30 --logdir resnetrs50_split3d3x9csn_l6_mixup_ep30 --num_workers 6
python eval_tta_fp16.py --seed 2023 --batch_size 16 --image_size 256 --fold 1 --logdir resnetrs50_split3d3x9csn_l6_mixup_ep30 --num_workers 16

python train.py --seed 3024 --model_name resnetrs50 --drop_rate 0.4 --drop_path_rate 0.2 --num_3d_layer 6 --in_chans 3 \
    --mixup_p 0.5 --mixup_alpha 0.2 --lr 1e-3   --batch_size 16 --image_size 256 --start_z 19 --end_z -19 --shift_z 2 \
    --fold 2 --gpus 4 --epochs 30 --logdir resnetrs50_split3d3x9csn_l6_mixup_ep30 --num_workers 6
python eval_tta_fp16.py --seed 2024 --batch_size 16 --image_size 256 --fold 2 --logdir resnetrs50_split3d3x9csn_l6_mixup_ep30 --num_workers 16

python train.py --seed 3025 --model_name resnetrs50 --drop_rate 0.4 --drop_path_rate 0.2 --num_3d_layer 6 --in_chans 3 \
    --mixup_p 0.5 --mixup_alpha 0.2 --lr 1e-3   --batch_size 16 --image_size 256 --start_z 19 --end_z -19 --shift_z 2 \
    --fold 3 --gpus 4 --epochs 30 --logdir resnetrs50_split3d3x9csn_l6_mixup_ep30 --num_workers 6
python eval_tta_fp16.py --seed 2025 --batch_size 16 --image_size 256 --fold 3 --logdir resnetrs50_split3d3x9csn_l6_mixup_ep30 --num_workers 16

python train.py --seed 3026 --model_name resnetrs50 --drop_rate 0.4 --drop_path_rate 0.2 --num_3d_layer 6 --in_chans 3 \
    --mixup_p 0.5 --mixup_alpha 0.2 --lr 1e-3   --batch_size 16 --image_size 256 --start_z 19 --end_z -19 --shift_z 2 \
    --fold 4 --gpus 4 --epochs 30 --logdir resnetrs50_split3d3x9csn_l6_mixup_ep30 --num_workers 6
python eval_tta_fp16.py --seed 3026 --batch_size 16 --image_size 256 --fold 4 --logdir resnetrs50_split3d3x9csn_l6_mixup_ep30 --num_workers 16



python train.py --seed 4022 --model_name convnext_tiny.fb_in1k --drop_rate 0.4 --drop_path_rate 0.2 --num_3d_layer 3 --in_chans 5 \
    --mixup_p 0.5 --mixup_alpha 0.2  --lr 1e-3  --batch_size 16 --image_size 256 --start_z 15 --end_z -15 --shift_z 2 \
    --fold 0 --gpus 4 --epochs 30 --logdir convnext_tiny_split3d5x7csn_mixup_ep30 --num_workers 6
python eval_tta_fp16.py --seed 2022 --batch_size 16 --image_size 256 --fold 0 --logdir convnext_tiny_split3d5x7csn_mixup_ep30 --num_workers 16

python train.py --seed 4023 --model_name convnext_tiny.fb_in1k --drop_rate 0.4 --drop_path_rate 0.2 --num_3d_layer 3 --in_chans 5 \
    --mixup_p 0.5 --mixup_alpha 0.2  --lr 1e-3  --batch_size 16 --image_size 256 --start_z 15 --end_z -15 --shift_z 2 \
    --fold 1 --gpus 4 --epochs 30 --logdir convnext_tiny_split3d5x7csn_mixup_ep30 --num_workers 6
python eval_tta_fp16.py --seed 2023 --batch_size 16 --image_size 256 --fold 1 --logdir convnext_tiny_split3d5x7csn_mixup_ep30 --num_workers 16

python train.py --seed 4024 --model_name convnext_tiny.fb_in1k --drop_rate 0.4 --drop_path_rate 0.2 --num_3d_layer 3 --in_chans 5 \
    --mixup_p 0.5 --mixup_alpha 0.2  --lr 1e-3  --batch_size 16 --image_size 256 --start_z 15 --end_z -15 --shift_z 2 \
    --fold 2 --gpus 4 --epochs 30 --logdir convnext_tiny_split3d5x7csn_mixup_ep30 --num_workers 6
python eval_tta_fp16.py --seed 2024 --batch_size 16 --image_size 256 --fold 2 --logdir convnext_tiny_split3d5x7csn_mixup_ep30 --num_workers 16

python train.py --seed 4025 --model_name convnext_tiny.fb_in1k --drop_rate 0.4 --drop_path_rate 0.2 --num_3d_layer 3 --in_chans 5 \
    --mixup_p 0.5 --mixup_alpha 0.2  --lr 1e-3  --batch_size 16 --image_size 256 --start_z 15 --end_z -15 --shift_z 2 \
    --fold 3 --gpus 4 --epochs 30 --logdir convnext_tiny_split3d5x7csn_mixup_ep30 --num_workers 6
python eval_tta_fp16.py --seed 2025 --batch_size 16 --image_size 256 --fold 3 --logdir convnext_tiny_split3d5x7csn_mixup_ep30 --num_workers 16

python train.py --seed 4026 --model_name convnext_tiny.fb_in1k --drop_rate 0.4 --drop_path_rate 0.2 --num_3d_layer 3 --in_chans 5 \
    --mixup_p 0.5 --mixup_alpha 0.2  --lr 1e-3  --batch_size 16 --image_size 256 --start_z 15 --end_z -15 --shift_z 2 \
    --fold 4 --gpus 4 --epochs 30 --logdir convnext_tiny_split3d5x7csn_mixup_ep30 --num_workers 6
python eval_tta_fp16.py --seed 2026 --batch_size 16 --image_size 256 --fold 4 --logdir convnext_tiny_split3d5x7csn_mixup_ep30 --num_workers 16



python train.py --seed 5022 --model_name convnext_tiny.fb_in1k --drop_rate 0.4 --drop_path_rate 0.2 --num_3d_layer 6 --in_chans 3 \
    --mixup_p 0.5 --mixup_alpha 0.2  --lr 1e-3  --batch_size 16 --image_size 256 --start_z 19 --end_z -19 --shift_z 2 \
    --fold 0 --gpus 4 --epochs 30 --logdir   --num_workers 6
python eval_tta_fp16.py --seed 2022 --batch_size 16 --image_size 256 --fold 0 --logdir convnext_tiny_split3d3x9csn_l6_mixup_ep30 --num_workers 16

python train.py --seed 5023 --model_name convnext_tiny.fb_in1k --drop_rate 0.4 --drop_path_rate 0.2 --num_3d_layer 6 --in_chans 3 \
    --mixup_p 0.5 --mixup_alpha 0.2  --lr 1e-3  --batch_size 16 --image_size 256 --start_z 19 --end_z -19 --shift_z 2 \
    --fold 1 --gpus 4 --epochs 30 --logdir convnext_tiny_split3d3x9csn_l6_mixup_ep30 --num_workers 6
python eval_tta_fp16.py --seed 2024 --batch_size 16 --image_size 256 --fold 1 --logdir convnext_tiny_split3d3x9csn_l6_mixup_ep30 --num_workers 16

python train.py --seed 5024 --model_name convnext_tiny.fb_in1k --drop_rate 0.4 --drop_path_rate 0.2 --num_3d_layer 6 --in_chans 3 \
    --mixup_p 0.5 --mixup_alpha 0.2  --lr 1e-3  --batch_size 16 --image_size 256 --start_z 19 --end_z -19 --shift_z 2 \
    --fold 2 --gpus 4 --epochs 30 --logdir convnext_tiny_split3d3x9csn_l6_mixup_ep30 --num_workers 6
python eval_tta_fp16.py --seed 2024 --batch_size 16 --image_size 256 --fold 2 --logdir convnext_tiny_split3d3x9csn_l6_mixup_ep30 --num_workers 16

python train.py --seed 5025 --model_name convnext_tiny.fb_in1k --drop_rate 0.4 --drop_path_rate 0.2 --num_3d_layer 6 --in_chans 3 \
    --mixup_p 0.5 --mixup_alpha 0.2  --lr 1e-3  --batch_size 16 --image_size 256 --start_z 19 --end_z -19 --shift_z 2 \
    --fold 3 --gpus 4 --epochs 30 --logdir convnext_tiny_split3d3x9csn_l6_mixup_ep30 --num_workers 6
python eval_tta_fp16.py --seed 2024 --batch_size 16 --image_size 256 --fold 3 --logdir convnext_tiny_split3d3x9csn_l6_mixup_ep30 --num_workers 16

python train.py --seed 5026 --model_name convnext_tiny.fb_in1k --drop_rate 0.4 --drop_path_rate 0.2 --num_3d_layer 6 --in_chans 3 \
    --mixup_p 0.5 --mixup_alpha 0.2  --lr 1e-3  --batch_size 16 --image_size 256 --start_z 19 --end_z -19 --shift_z 2 \
    --fold 4 --gpus 4 --epochs 30 --logdir convnext_tiny_split3d3x9csn_l6_mixup_ep30 --num_workers 6
python eval_tta_fp16.py --seed 2024 --batch_size 16 --image_size 256 --fold 4 --logdir convnext_tiny_split3d3x9csn_l6_mixup_ep30 --num_workers 16


python train.py --seed 6022 --model_name swinv2_tiny_window8_256.ms_in1k --drop_rate 0.4 --drop_path_rate 0.2 --num_3d_layer 3 --in_chans 5 \
    --mixup_p 0.5 --mixup_alpha 0.2  --lr 2e-4  --batch_size 12 --image_size 256 --start_z 15 --end_z -15 --shift_z 2 \
    --fold 0 --gpus 4 --epochs 30 --logdir swinv2_tiny_window8_256_split3d5x7csn_mixup_ep30 --num_workers 6
python eval_tta_fp16.py --seed 2022 --batch_size 16 --image_size 256 --fold 0 --logdir swinv2_tiny_window8_256_split3d5x7csn_mixup_ep30 --num_workers 16

python train.py --seed 6023 --model_name swinv2_tiny_window8_256.ms_in1k --drop_rate 0.4 --drop_path_rate 0.2 --num_3d_layer 3 --in_chans 5 \
    --mixup_p 0.5 --mixup_alpha 0.2  --lr 2e-4  --batch_size 12 --image_size 256 --start_z 15 --end_z -15 --shift_z 2 \
    --fold 1 --gpus 4 --epochs 30 --logdir swinv2_tiny_window8_256_split3d5x7csn_mixup_ep30 --num_workers 6
python eval_tta_fp16.py --seed 2023 --batch_size 16 --image_size 256 --fold 1 --logdir swinv2_tiny_window8_256_split3d5x7csn_mixup_ep30 --num_workers 16

python train.py --seed 6024 --model_name swinv2_tiny_window8_256.ms_in1k --drop_rate 0.4 --drop_path_rate 0.2 --num_3d_layer 3 --in_chans 5 \
    --mixup_p 0.5 --mixup_alpha 0.2  --lr 2e-4  --batch_size 12 --image_size 256 --start_z 15 --end_z -15 --shift_z 2 \
    --fold 2 --gpus 4 --epochs 30 --logdir swinv2_tiny_window8_256_split3d5x7csn_mixup_ep30 --num_workers 6
python eval_tta_fp16.py --seed 2024 --batch_size 16 --image_size 256 --fold 2 --logdir swinv2_tiny_window8_256_split3d5x7csn_mixup_ep30 --num_workers 16

python train.py --seed 6025 --model_name swinv2_tiny_window8_256.ms_in1k --drop_rate 0.4 --drop_path_rate 0.2 --num_3d_layer 3 --in_chans 5 \
    --mixup_p 0.5 --mixup_alpha 0.2  --lr 2e-4  --batch_size 12 --image_size 256 --start_z 15 --end_z -15 --shift_z 2 \
    --fold 3 --gpus 4 --epochs 30 --logdir swinv2_tiny_window8_256_split3d5x7csn_mixup_ep30 --num_workers 6
python eval_tta_fp16.py --seed 2025 --batch_size 16 --image_size 256 --fold 3 --logdir swinv2_tiny_window8_256_split3d5x7csn_mixup_ep30 --num_workers 16

python train.py --seed 6026 --model_name swinv2_tiny_window8_256.ms_in1k --drop_rate 0.4 --drop_path_rate 0.2 --num_3d_layer 3 --in_chans 5 \
    --mixup_p 0.5 --mixup_alpha 0.2  --lr 2e-4  --batch_size 12 --image_size 256 --start_z 15 --end_z -15 --shift_z 2 \
    --fold 4 --gpus 4 --epochs 30 --logdir swinv2_tiny_window8_256_split3d5x7csn_mixup_ep30 --num_workers 6
python eval_tta_fp16.py --seed 2026 --batch_size 16 --image_size 256 --fold 4 --logdir swinv2_tiny_window8_256_split3d5x7csn_mixup_ep30 --num_workers 16


python train.py --seed 7022 --model_name swinv2_tiny_window8_256.ms_in1k --drop_rate 0.4 --drop_path_rate 0.2 --num_3d_layer 6 --in_chans 3 \
    --mixup_p 0.5 --mixup_alpha 0.2  --lr 2e-4  --batch_size 12 --image_size 256 --start_z 19 --end_z -19 --shift_z 2 \
    --fold 0 --gpus 4 --epochs 30 --logdir swinv2_tiny_window8_256_split3d3x9csn_l6_mixup_ep30 --num_workers 6
python eval_tta_fp16.py --seed 2022 --batch_size 16 --image_size 256 --fold 0 --logdir swinv2_tiny_window8_256_split3d3x9csn_l6_mixup_ep30 --num_workers 16

python train.py --seed 7023 --model_name swinv2_tiny_window8_256.ms_in1k --drop_rate 0.4 --drop_path_rate 0.2 --num_3d_layer 6 --in_chans 3 \
    --mixup_p 0.5 --mixup_alpha 0.2  --lr 2e-4  --batch_size 12 --image_size 256 --start_z 19 --end_z -19 --shift_z 2 \
    --fold 1 --gpus 4 --epochs 30 --logdir swinv2_tiny_window8_256_split3d3x9csn_l6_mixup_ep30 --num_workers 6
python eval_tta_fp16.py --seed 2024 --batch_size 16 --image_size 256 --fold 1 --logdir swinv2_tiny_window8_256_split3d3x9csn_l6_mixup_ep30 --num_workers 16

python train.py --seed 7024 --model_name swinv2_tiny_window8_256.ms_in1k --drop_rate 0.4 --drop_path_rate 0.2 --num_3d_layer 6 --in_chans 3 \
    --mixup_p 0.5 --mixup_alpha 0.2  --lr 2e-4  --batch_size 12 --image_size 256 --start_z 19 --end_z -19 --shift_z 2 \
    --fold 2 --gpus 4 --epochs 30 --logdir swinv2_tiny_window8_256_split3d3x9csn_l6_mixup_ep30 --num_workers 6
python eval_tta_fp16.py --seed 2024 --batch_size 16 --image_size 256 --fold 2 --logdir swinv2_tiny_window8_256_split3d3x9csn_l6_mixup_ep30 --num_workers 16

python train.py --seed 7025 --model_name swinv2_tiny_window8_256.ms_in1k --drop_rate 0.4 --drop_path_rate 0.2 --num_3d_layer 6 --in_chans 3 \
    --mixup_p 0.5 --mixup_alpha 0.2  --lr 2e-4  --batch_size 12 --image_size 256 --start_z 19 --end_z -19 --shift_z 2 \
    --fold 3 --gpus 4 --epochs 30 --logdir swinv2_tiny_window8_256_split3d3x9csn_l6_mixup_ep30 --num_workers 6
python eval_tta_fp16.py --seed 2024 --batch_size 16 --image_size 256 --fold 3 --logdir swinv2_tiny_window8_256_split3d3x9csn_l6_mixup_ep30 --num_workers 16

python train.py --seed 7026 --model_name swinv2_tiny_window8_256.ms_in1k --drop_rate 0.4 --drop_path_rate 0.2 --num_3d_layer 6 --in_chans 3 \
    --mixup_p 0.5 --mixup_alpha 0.2  --lr 2e-4  --batch_size 12 --image_size 256 --start_z 19 --end_z -19 --shift_z 2 \
    --fold 4 --gpus 4 --epochs 30 --logdir swinv2_tiny_window8_256_split3d3x9csn_l6_mixup_ep30 --num_workers 6
python eval_tta_fp16.py --seed 2024 --batch_size 16 --image_size 256 --fold 4 --logdir swinv2_tiny_window8_256_split3d3x9csn_l6_mixup_ep30 --num_workers 16



python train.py --seed 9022 --model_name swin_small_patch4_window7_224.ms_in22k_ft_in1k --drop_rate 0.4 --drop_path_rate 0.2 --num_3d_layer 3 --in_chans 5 \
    --mixup_p 0.5 --mixup_alpha 0.2  --lr 2e-4 --batch_size 12 --image_size 224 --start_z 15 --end_z -15 --shift_z 2 \
    --fold 0 --gpus 4 --epochs 30 --logdir swin_small_patch4_window7_224_split3d5x7csn_mixup_ep30 --num_workers 6
python eval_tta_fp16.py --seed 2022 --batch_size 16 --image_size 224 --fold 0 --logdir swin_small_patch4_window7_224_split3d5x7csn_mixup_ep30 --num_workers 16

python train.py --seed 9023 --model_name swin_small_patch4_window7_224.ms_in22k_ft_in1k --drop_rate 0.4 --drop_path_rate 0.2 --num_3d_layer 3 --in_chans 5 \
    --mixup_p 0.5 --mixup_alpha 0.2  --lr 2e-4 --batch_size 12 --image_size 224 --start_z 15 --end_z -15 --shift_z 2 \
    --fold 1 --gpus 4 --epochs 30 --logdir swin_small_patch4_window7_224_split3d5x7csn_mixup_ep30 --num_workers 6
python eval_tta_fp16.py --seed 2023 --batch_size 16 --image_size 224 --fold 1 --logdir swin_small_patch4_window7_224_split3d5x7csn_mixup_ep30 --num_workers 16

python train.py --seed 9024 --model_name swin_small_patch4_window7_224.ms_in22k_ft_in1k --drop_rate 0.4 --drop_path_rate 0.2 --num_3d_layer 3 --in_chans 5 \
    --mixup_p 0.5 --mixup_alpha 0.2  --lr 2e-4 --batch_size 12 --image_size 224 --start_z 15 --end_z -15 --shift_z 2 \
    --fold 2 --gpus 4 --epochs 30 --logdir swin_small_patch4_window7_224_split3d5x7csn_mixup_ep30 --num_workers 6
python eval_tta_fp16.py --seed 2024 --batch_size 16 --image_size 224 --fold 2 --logdir swin_small_patch4_window7_224_split3d5x7csn_mixup_ep30 --num_workers 16

python train.py --seed 9025 --model_name swin_small_patch4_window7_224.ms_in22k_ft_in1k --drop_rate 0.4 --drop_path_rate 0.2 --num_3d_layer 3 --in_chans 5 \
    --mixup_p 0.5 --mixup_alpha 0.2  --lr 2e-4 --batch_size 12 --image_size 224 --start_z 15 --end_z -15 --shift_z 2 \
    --fold 3 --gpus 4 --epochs 30 --logdir swin_small_patch4_window7_224_split3d5x7csn_mixup_ep30 --num_workers 6
python eval_tta_fp16.py --seed 2025 --batch_size 16 --image_size 224 --fold 3 --logdir swin_small_patch4_window7_224_split3d5x7csn_mixup_ep30 --num_workers 16

python train.py --seed 9026 --model_name swin_small_patch4_window7_224.ms_in22k_ft_in1k --drop_rate 0.4 --drop_path_rate 0.2 --num_3d_layer 3 --in_chans 5 \
    --mixup_p 0.5 --mixup_alpha 0.2  --lr 2e-4 --batch_size 12 --image_size 224 --start_z 15 --end_z -15 --shift_z 2 \
    --fold 4 --gpus 4 --epochs 30 --logdir swin_small_patch4_window7_224_split3d5x7csn_mixup_ep30 --num_workers 6
python eval_tta_fp16.py --seed 2026 --batch_size 16 --image_size 224 --fold 4 --logdir swin_small_patch4_window7_224_split3d5x7csn_mixup_ep30 --num_workers 16


python train.py --seed 10022 --model_name swin_small_patch4_window7_224.ms_in22k_ft_in1k --drop_rate 0.4 --drop_path_rate 0.2 --num_3d_layer 6 --in_chans 3 \
    --mixup_p 0.5 --mixup_alpha 0.2  --lr 2e-4 --batch_size 12 --image_size 224 --start_z 19 --end_z -19 --shift_z 2 \
    --fold 0 --gpus 4 --epochs 30 --logdir swin_small_patch4_window7_224_split3d3x9csn_l6_mixup_ep30 --num_workers 6
python eval_tta_fp16.py --seed 2022 --batch_size 16 --image_size 224 --fold 0 --logdir swin_small_patch4_window7_224_split3d3x9csn_l6_mixup_ep30 --num_workers 16

python train.py --seed 10023 --model_name swin_small_patch4_window7_224.ms_in22k_ft_in1k --drop_rate 0.4 --drop_path_rate 0.2 --num_3d_layer 6 --in_chans 3 \
    --mixup_p 0.5 --mixup_alpha 0.2  --lr 2e-4 --batch_size 12 --image_size 224 --start_z 19 --end_z -19 --shift_z 2 \
    --fold 1 --gpus 4 --epochs 30 --logdir swin_small_patch4_window7_224_split3d3x9csn_l6_mixup_ep30 --num_workers 6
python eval_tta_fp16.py --seed 2024 --batch_size 16 --image_size 224 --fold 1 --logdir swin_small_patch4_window7_224_split3d3x9csn_l6_mixup_ep30 --num_workers 16

python train.py --seed 10024 --model_name swin_small_patch4_window7_224.ms_in22k_ft_in1k --drop_rate 0.4 --drop_path_rate 0.2 --num_3d_layer 6 --in_chans 3 \
    --mixup_p 0.5 --mixup_alpha 0.2  --lr 2e-4 --batch_size 12 --image_size 224 --start_z 19 --end_z -19 --shift_z 2 \
    --fold 2 --gpus 4 --epochs 30 --logdir swin_small_patch4_window7_224_split3d3x9csn_l6_mixup_ep30 --num_workers 6
python eval_tta_fp16.py --seed 2024 --batch_size 16 --image_size 224 --fold 2 --logdir swin_small_patch4_window7_224_split3d3x9csn_l6_mixup_ep30 --num_workers 16

python train.py --seed 10025 --model_name swin_small_patch4_window7_224.ms_in22k_ft_in1k --drop_rate 0.4 --drop_path_rate 0.2 --num_3d_layer 6 --in_chans 3 \
    --mixup_p 0.5 --mixup_alpha 0.2  --lr 2e-4 --batch_size 12 --image_size 224 --start_z 19 --end_z -19 --shift_z 2 \
    --fold 3 --gpus 4 --epochs 30 --logdir swin_small_patch4_window7_224_split3d3x9csn_l6_mixup_ep30 --num_workers 6
python eval_tta_fp16.py --seed 2024 --batch_size 16 --image_size 224 --fold 3 --logdir swin_small_patch4_window7_224_split3d3x9csn_l6_mixup_ep30 --num_workers 16

python train.py --seed 10026 --model_name swin_small_patch4_window7_224.ms_in22k_ft_in1k --drop_rate 0.4 --drop_path_rate 0.2 --num_3d_layer 6 --in_chans 3 \
    --mixup_p 0.5 --mixup_alpha 0.2  --lr 2e-4 --batch_size 12 --image_size 224 --start_z 19 --end_z -19 --shift_z 2 \
    --fold 4 --gpus 4 --epochs 30 --logdir swin_small_patch4_window7_224_split3d3x9csn_l6_mixup_ep30 --num_workers 6
python eval_tta_fp16.py --seed 2024 --batch_size 16 --image_size 224 --fold 4 --logdir swin_small_patch4_window7_224_split3d3x9csn_l6_mixup_ep30 --num_workers 16



python train.py --seed 11022 --model_name ecaresnet26t.ra2_in1k --drop_rate 0.4 --drop_path_rate 0.2 --num_3d_layer 6 --in_chans 3 \
    --mixup_p 0.5 --mixup_alpha 0.2 --lr 1e-3  --batch_size 16 --image_size 256 --start_z 14 --end_z -15 --shift_z 2 \
    --fold 0 --gpus 4 --epochs 30 --logdir ecaresnet26t_split3d3x12csn_l6_mixup_ep30 --num_workers 6
python eval_tta_fp16.py --seed 11022 --batch_size 16 --image_size 256 --fold 0 --logdir ecaresnet26t_split3d3x12csn_l6_mixup_ep30  --num_workers 16

python train.py --seed 11027 --model_name ecaresnet26t.ra2_in1k --drop_rate 0.4 --drop_path_rate 0.2 --num_3d_layer 6 --in_chans 3 \
    --mixup_p 0.5 --mixup_alpha 0.2 --lr 1e-3  --batch_size 16 --image_size 256 --start_z 14 --end_z -15 --shift_z 2 \
    --fold 1 --gpus 4 --epochs 30 --logdir ecaresnet26t_split3d3x12csn_l6_mixup_ep30  --num_workers 6
python eval_tta_fp16.py --seed 11022 --batch_size 16 --image_size 256 --fold 1 --logdir ecaresnet26t_split3d3x12csn_l6_mixup_ep30  --num_workers 16

python train.py --seed 11024 --model_name ecaresnet26t.ra2_in1k --drop_rate 0.4 --drop_path_rate 0.2 --num_3d_layer 6 --in_chans 3 \
    --mixup_p 0.5 --mixup_alpha 0.2 --lr 1e-3   --batch_size 16 --image_size 256 --start_z 14 --end_z -15 --shift_z 2 \
    --fold 2 --gpus 4 --epochs 30 --logdir ecaresnet26t_split3d3x12csn_l6_mixup_ep30  --num_workers 6
python eval_tta_fp16.py --seed 11024 --batch_size 16 --image_size 256 --fold 2 --logdir ecaresnet26t_split3d3x12csn_l6_mixup_ep30  --num_workers 16

python train.py --seed 11025 --model_name ecaresnet26t.ra2_in1k --drop_rate 0.4 --drop_path_rate 0.2 --num_3d_layer 6 --in_chans 3 \
    --mixup_p 0.5 --mixup_alpha 0.2 --lr 1e-3  --batch_size 16 --image_size 256 --start_z 14 --end_z -15 --shift_z 2 \
    --fold 3 --gpus 4 --epochs 30 --logdir ecaresnet26t_split3d3x12csn_l6_mixup_ep30  --num_workers 6
python eval_tta_fp16.py --seed 11025 --batch_size 16 --image_size 256 --fold 3 --logdir ecaresnet26t_split3d3x12csn_l6_mixup_ep30  --num_workers 16

python train.py --seed 11026 --model_name ecaresnet26t.ra2_in1k --drop_rate 0.4 --drop_path_rate 0.2 --num_3d_layer 6 --in_chans 3 \
    --mixup_p 0.5 --mixup_alpha 0.2 --lr 1e-3  --batch_size 16 --image_size 256 --start_z 14 --end_z -15 --shift_z 2 \
    --fold 4 --gpus 4 --epochs 30 --logdir ecaresnet26t_split3d3x12csn_l6_mixup_ep30  --num_workers 6
python eval_tta_fp16.py --seed 11026 --batch_size 16 --image_size 256 --fold 4 --logdir ecaresnet26t_split3d3x12csn_l6_mixup_ep30  --num_workers 16



python train.py --seed 12022 --model_name ecaresnet26t.ra2_in1k --drop_rate 0.4 --drop_path_rate 0.2 --num_3d_layer 6 --in_chans 2 \
    --mixup_p 0.5 --mixup_alpha 0.2 --lr 1e-3  --batch_size 16 --image_size 224 --start_z 17 --end_z -18 --shift_z 2 \
    --fold 0 --gpus 4 --epochs 30 --logdir ecaresnet26t_split3d2x15csn_l6_mixup_ep30 --num_workers 6
python eval_tta_fp16.py --seed 2022 --batch_size 16 --image_size 224 --fold 0 --logdir ecaresnet26t_split3d2x15csn_l6_mixup_ep30 --num_workers 16

python train.py --seed 12023 --model_name ecaresnet26t.ra2_in1k --drop_rate 0.4 --drop_path_rate 0.2 --num_3d_layer 6 --in_chans 2 \
    --mixup_p 0.5 --mixup_alpha 0.2 --lr 1e-3  --batch_size 16 --image_size 224 --start_z 17 --end_z -18 --shift_z 2 \
    --fold 1 --gpus 4 --epochs 30 --logdir ecaresnet26t_split3d2x15csn_l6_mixup_ep30 --num_workers 6
python eval_tta_fp16.py --seed 2023 --batch_size 16 --image_size 224 --fold 1 --logdir ecaresnet26t_split3d2x15csn_l6_mixup_ep30 --num_workers 16

python train.py --seed 12024 --model_name ecaresnet26t.ra2_in1k --drop_rate 0.4 --drop_path_rate 0.2 --num_3d_layer 6 --in_chans 2 \
    --mixup_p 0.5 --mixup_alpha 0.2 --lr 1e-3  --batch_size 16 --image_size 224 --start_z 17 --end_z -18 --shift_z 2 \
    --fold 2 --gpus 4 --epochs 30 --logdir ecaresnet26t_split3d2x15csn_l6_mixup_ep30 --num_workers 6
python eval_tta_fp16.py --seed 2024 --batch_size 16 --image_size 224 --fold 2 --logdir ecaresnet26t_split3d2x15csn_l6_mixup_ep30 --num_workers 16

python train.py --seed 12025 --model_name ecaresnet26t.ra2_in1k --drop_rate 0.4 --drop_path_rate 0.2 --num_3d_layer 6 --in_chans 2 \
    --mixup_p 0.5 --mixup_alpha 0.2 --lr 1e-3  --batch_size 16 --image_size 224 --start_z 17 --end_z -18 --shift_z 2 \
    --fold 3 --gpus 4 --epochs 30 --logdir ecaresnet26t_split3d2x15csn_l6_mixup_ep30 --num_workers 6
python eval_tta_fp16.py --seed 2025 --batch_size 16 --image_size 224 --fold 3 --logdir ecaresnet26t_split3d2x15csn_l6_mixup_ep30 --num_workers 16

python train.py --seed 12026 --model_name ecaresnet26t.ra2_in1k --drop_rate 0.4 --drop_path_rate 0.2 --num_3d_layer 6 --in_chans 2 \
    --mixup_p 0.5 --mixup_alpha 0.2 --lr 1e-3  --batch_size 16 --image_size 224 --start_z 17 --end_z -18 --shift_z 2 \
    --fold 4 --gpus 4 --epochs 30 --logdir ecaresnet26t_split3d2x15csn_l6_mixup_ep30 --num_workers 6
python eval_tta_fp16.py --seed 3026 --batch_size 16 --image_size 224 --fold 4 --logdir ecaresnet26t_split3d2x15csn_l6_mixup_ep30 --num_workers 16



python train.py --seed 2022 --model_name resnext50d_32x4d --drop_rate 0.4 --drop_path_rate 0.2 --num_3d_layer 6 --in_chans 3 \
    --mixup_p 0.5 --mixup_alpha 0.2 --lr 0.25e-3   --batch_size 16 --image_size 256 --start_z 19 --end_z -19 --shift_z 2 \
    --fold 0 --gpus 1 --epochs 15 --logdir resnext50d_32x4d_split3d3x9csn_l6_mixup_ep15 --num_workers 16
python eval_tta.py --seed 2022 --batch_size 16 --image_size 256 --fold 0 --logdir resnext50d_32x4d_split3d3x9csn_l6_mixup_ep15 --num_workers 16

python train.py --seed 2023 --model_name resnext50d_32x4d --drop_rate 0.4 --drop_path_rate 0.2 --num_3d_layer 6 --in_chans 3 \
    --mixup_p 0.5 --mixup_alpha 0.2 --lr 0.25e-3   --batch_size 16 --image_size 256 --start_z 19 --end_z -19 --shift_z 2 \
    --fold 1 --gpus 1 --epochs 15 --logdir resnext50d_32x4d_split3d3x9csn_l6_mixup_ep15 --num_workers 16
python eval_tta.py --seed 2023 --batch_size 16 --image_size 256 --fold 1 --logdir resnext50d_32x4d_split3d3x9csn_l6_mixup_ep15 --num_workers 16