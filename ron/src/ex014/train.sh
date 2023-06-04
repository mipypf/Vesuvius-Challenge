# python train.py --seed 2032 --model_name resnetrs50 --drop_path_rate 0.5 --num_3d_layer 3 --in_chans 5 \
#     --mixup_p 0.5 --mixup_alpha 0.2 --lr 1e-3  --batch_size 16 --image_size 256 --start_z 15 --end_z -15 --shift_z 2 \
#     --fold 0 --gpus 4 --epochs 30 --logdir resnetrs50_split3d5x7csn_mixup_ep30 --num_workers 6
# python eval_tta_fp16.py --seed 2032 --batch_size 16 --image_size 256 --fold 0 --logdir resnetrs50_split3d5x7csn_mixup_ep30 --num_workers 16

# python train.py --seed 2033 --model_name resnetrs50 --drop_path_rate 0.5 --num_3d_layer 3 --in_chans 5 \
#     --mixup_p 0.5 --mixup_alpha 0.2 --lr 1e-3  --batch_size 16 --image_size 256 --start_z 15 --end_z -15 --shift_z 2 \
#     --fold 1 --gpus 4 --epochs 30 --logdir resnetrs50_split3d5x7csn_mixup_ep30 --num_workers 6
# python eval_tta_fp16.py --seed 2032 --batch_size 16 --image_size 256 --fold 1 --logdir resnetrs50_split3d5x7csn_mixup_ep30 --num_workers 16

# python train.py --seed 2034 --model_name resnetrs50 --drop_path_rate 0.5 --num_3d_layer 3 --in_chans 5 \
#     --mixup_p 0.5 --mixup_alpha 0.2 --lr 1e-3   --batch_size 16 --image_size 256 --start_z 15 --end_z -15 --shift_z 2 \
#     --fold 2 --gpus 4 --epochs 30 --logdir resnetrs50_split3d5x7csn_mixup_ep30 --num_workers 6
# python eval_tta_fp16.py --seed 2034 --batch_size 16 --image_size 256 --fold 2 --logdir resnetrs50_split3d5x7csn_mixup_ep30 --num_workers 16


# python train.py --seed 5032 --model_name convnext_tiny.fb_in1k --drop_path_rate 0.5 --num_3d_layer 6 --in_chans 3 \
#     --mixup_p 0.5 --mixup_alpha 0.2  --lr 1e-3  --batch_size 16 --image_size 256 --start_z 19 --end_z -19 --shift_z 2 \
#     --fold 0 --gpus 4 --epochs 30 --logdir convnext_tiny_split3d3x9csn_l6_mixup_ep30  --num_workers 6
# python eval_tta_fp16.py --seed 2032 --batch_size 16 --image_size 256 --fold 0 --logdir convnext_tiny_split3d3x9csn_l6_mixup_ep30 --num_workers 16

# python train.py --seed 5033 --model_name convnext_tiny.fb_in1k --drop_path_rate 0.5 --num_3d_layer 6 --in_chans 3 \
#     --mixup_p 0.5 --mixup_alpha 0.2  --lr 1e-3  --batch_size 16 --image_size 256 --start_z 19 --end_z -19 --shift_z 2 \
#     --fold 1 --gpus 4 --epochs 30 --logdir convnext_tiny_split3d3x9csn_l6_mixup_ep30 --num_workers 6
# python eval_tta_fp16.py --seed 2034 --batch_size 16 --image_size 256 --fold 1 --logdir convnext_tiny_split3d3x9csn_l6_mixup_ep30 --num_workers 16

# python train.py --seed 5034 --model_name convnext_tiny.fb_in1k --drop_path_rate 0.5 --num_3d_layer 6 --in_chans 3 \
#     --mixup_p 0.5 --mixup_alpha 0.2  --lr 1e-3  --batch_size 16 --image_size 256 --start_z 19 --end_z -19 --shift_z 2 \
#     --fold 2 --gpus 4 --epochs 30 --logdir convnext_tiny_split3d3x9csn_l6_mixup_ep30 --num_workers 6
# python eval_tta_fp16.py --seed 2034 --batch_size 16 --image_size 256 --fold 2 --logdir convnext_tiny_split3d3x9csn_l6_mixup_ep30 --num_workers 16


# python train.py --seed 6032 --model_name swinv2_tiny_window8_256.ms_in1k --drop_path_rate 0.5 --num_3d_layer 3 --in_chans 5 \
#     --mixup_p 0.5 --mixup_alpha 0.2  --lr 2e-4  --batch_size 12 --image_size 256 --start_z 15 --end_z -15 --shift_z 2 \
#     --fold 0 --gpus 4 --epochs 30 --logdir swinv2_tiny_window8_256_split3d5x7csn_mixup_ep30 --num_workers 6
# python eval_tta_fp16.py --seed 2032 --batch_size 16 --image_size 256 --fold 0 --logdir swinv2_tiny_window8_256_split3d5x7csn_mixup_ep30 --num_workers 16

# python train.py --seed 6033 --model_name swinv2_tiny_window8_256.ms_in1k --drop_path_rate 0.5 --num_3d_layer 3 --in_chans 5 \
#     --mixup_p 0.5 --mixup_alpha 0.2  --lr 2e-4  --batch_size 12 --image_size 256 --start_z 15 --end_z -15 --shift_z 2 \
#     --fold 1 --gpus 4 --epochs 30 --logdir swinv2_tiny_window8_256_split3d5x7csn_mixup_ep30 --num_workers 6
# python eval_tta_fp16.py --seed 2033 --batch_size 16 --image_size 256 --fold 1 --logdir swinv2_tiny_window8_256_split3d5x7csn_mixup_ep30 --num_workers 16

# python train.py --seed 6034 --model_name swinv2_tiny_window8_256.ms_in1k --drop_path_rate 0.5 --num_3d_layer 3 --in_chans 5 \
#     --mixup_p 0.5 --mixup_alpha 0.2  --lr 2e-4  --batch_size 12 --image_size 256 --start_z 15 --end_z -15 --shift_z 2 \
#     --fold 2 --gpus 4 --epochs 30 --logdir swinv2_tiny_window8_256_split3d5x7csn_mixup_ep30 --num_workers 6
# python eval_tta_fp16.py --seed 2034 --batch_size 16 --image_size 256 --fold 2 --logdir swinv2_tiny_window8_256_split3d5x7csn_mixup_ep30 --num_workers 16


# python train.py --seed 10032 --model_name swin_small_patch4_window7_224.ms_in22k_ft_in1k --drop_path_rate 0.5 --num_3d_layer 6 --in_chans 3 \
#     --mixup_p 0.5 --mixup_alpha 0.2  --lr 2e-4 --batch_size 12 --image_size 224 --start_z 19 --end_z -19 --shift_z 2 \
#     --fold 0 --gpus 4 --epochs 30 --logdir swin_small_patch4_window7_224_split3d3x9csn_l6_mixup_ep30 --num_workers 6
# python eval_tta_fp16.py --seed 2032 --batch_size 16 --image_size 224 --fold 0 --logdir swin_small_patch4_window7_224_split3d3x9csn_l6_mixup_ep30 --num_workers 16

# python train.py --seed 10033 --model_name swin_small_patch4_window7_224.ms_in22k_ft_in1k --drop_path_rate 0.5 --num_3d_layer 6 --in_chans 3 \
#     --mixup_p 0.5 --mixup_alpha 0.2  --lr 2e-4 --batch_size 12 --image_size 224 --start_z 19 --end_z -19 --shift_z 2 \
#     --fold 1 --gpus 4 --epochs 30 --logdir swin_small_patch4_window7_224_split3d3x9csn_l6_mixup_ep30 --num_workers 6
# python eval_tta_fp16.py --seed 2034 --batch_size 16 --image_size 224 --fold 1 --logdir swin_small_patch4_window7_224_split3d3x9csn_l6_mixup_ep30 --num_workers 16

# python train.py --seed 10034 --model_name swin_small_patch4_window7_224.ms_in22k_ft_in1k --drop_path_rate 0.5 --num_3d_layer 6 --in_chans 3 \
#     --mixup_p 0.5 --mixup_alpha 0.2  --lr 2e-4 --batch_size 12 --image_size 224 --start_z 19 --end_z -19 --shift_z 2 \
#     --fold 2 --gpus 4 --epochs 30 --logdir swin_small_patch4_window7_224_split3d3x9csn_l6_mixup_ep30 --num_workers 6
# python eval_tta_fp16.py --seed 2034 --batch_size 16 --image_size 224 --fold 2 --logdir swin_small_patch4_window7_224_split3d3x9csn_l6_mixup_ep30 --num_workers 16


# python train.py --seed 11032 --model_name ecaresnet26t.ra2_in1k --drop_path_rate 0.5 --num_3d_layer 6 --in_chans 3 \
#     --mixup_p 0.5 --mixup_alpha 0.2 --lr 1e-3  --batch_size 16 --image_size 256 --start_z 14 --end_z -15 --shift_z 2 \
#     --fold 0 --gpus 4 --epochs 30 --logdir ecaresnet26t_split3d3x12csn_l6_mixup_ep30 --num_workers 6
# python eval_tta_fp16.py --seed 11032 --batch_size 16 --image_size 256 --fold 0 --logdir ecaresnet26t_split3d3x12csn_l6_mixup_ep30  --num_workers 16

# python train.py --seed 11037 --model_name ecaresnet26t.ra2_in1k --drop_path_rate 0.5 --num_3d_layer 6 --in_chans 3 \
#     --mixup_p 0.5 --mixup_alpha 0.2 --lr 1e-3  --batch_size 16 --image_size 256 --start_z 14 --end_z -15 --shift_z 2 \
#     --fold 1 --gpus 4 --epochs 30 --logdir ecaresnet26t_split3d3x12csn_l6_mixup_ep30  --num_workers 6
# python eval_tta_fp16.py --seed 11032 --batch_size 16 --image_size 256 --fold 1 --logdir ecaresnet26t_split3d3x12csn_l6_mixup_ep30  --num_workers 16

# python train.py --seed 11034 --model_name ecaresnet26t.ra2_in1k --drop_path_rate 0.5 --num_3d_layer 6 --in_chans 3 \
#     --mixup_p 0.5 --mixup_alpha 0.2 --lr 1e-3   --batch_size 16 --image_size 256 --start_z 14 --end_z -15 --shift_z 2 \
#     --fold 2 --gpus 4 --epochs 30 --logdir ecaresnet26t_split3d3x12csn_l6_mixup_ep30  --num_workers 6
# python eval_tta_fp16.py --seed 11034 --batch_size 16 --image_size 256 --fold 2 --logdir ecaresnet26t_split3d3x12csn_l6_mixup_ep30  --num_workers 16


# python train.py --seed 20032 --model_name ecaresnet50t.ra2_in1k --drop_path_rate 0.5 --num_3d_layer 6 --in_chans  \
#     --mixup_p 0.5 --mixup_alpha 0.2 --lr 1e-3  --batch_size 16 --image_size 256 --start_z 19 --end_z -19 --shift_z 2 \
#     --fold 0 --gpus 4 --epochs 30 --logdir ecaresnet50t_split3d3x9csn_l6_mixup_ep30 --num_workers 6
# python eval_tta_fp16.py --seed 2032 --batch_size 16 --image_size 256 --fold 0 --logdir resnetrs50_split3d3x9csn_l6_mixup_ep30 --num_workers 16

# python train.py --seed 20033 --model_name ecaresnet50t.ra2_in1k --drop_path_rate 0.5 --num_3d_layer 6 --in_chans  \
#     --mixup_p 0.5 --mixup_alpha 0.2 --lr 1e-3  --batch_size 16 --image_size 256 --start_z 19 --end_z -19 --shift_z 2 \
#     --fold 1 --gpus 4 --epochs 30 --logdir ecaresnet50t_split3d3x9csn_l6_mixup_ep30 --num_workers 6
# python eval_tta_fp16.py --seed 2032 --batch_size 16 --image_size 256 --fold 1 --logdir resnetrs50_split3d3x9csn_l6_mixup_ep30 --num_workers 16

# python train.py --seed 20034 --model_name ecaresnet50t.ra2_in1k --drop_path_rate 0.5 --num_3d_layer 6 --in_chans  \
#     --mixup_p 0.5 --mixup_alpha 0.2 --lr 1e-3   --batch_size 16 --image_size 256 --start_z 15 --end_z -15 --shift_z 2 \
#     --fold 2 --gpus 4 --epochs 30 --logdir ecaresnet50t_split3d3x9csn_l6_mixup_ep30 --num_workers 6
# python eval_tta_fp16.py --seed 2034 --batch_size 16 --image_size 256 --fold 2 --logdir resnetrs50_split3d3x9csn_l6_mixup_ep30 --num_workers 16

# python train.py --seed 2032 --model_name resnext50d_32x4d --drop_rate 0.4 --drop_path_rate 0.2 --num_3d_layer 6 --in_chans 3 --mixup_p 0.5 --mixup_alpha 0.2 --lr 0.25e-3   --batch_size 16 --image_size 256 --start_z 19 --end_z -19 --shift_z 2 --fold 0 --gpus 1 --epochs 15 --logdir resnext50d_32x4d_split3d3x9csn_l6_mixup_ep15 --num_workers 16
python eval_tta_fp16.py --seed 2032 --batch_size 16 --image_size 256 --fold 0 --logdir resnext50d_32x4d_split3d3x9csn_l6_mixup_ep15 --num_workers 16