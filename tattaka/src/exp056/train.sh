# python train.py --seed 2022 --drop_path_rate 0.2 --mixup_p 0.5 --mixup_alpha 0.2 --lr 1e-3 --backbone_lr 2e-4 --in_chans 1 --batch_size 16 --image_size 192 --start_z 16 --end_z -17 --shift_z 2 \
#     --fold 0 --gpus 4 --epochs 30 --logdir resnet3d50csnir1x32_mixup_ep30 --num_workers 6
# python eval_tta_fp16.py --seed 2022 --batch_size 16 --image_size 192 --fold 0 --logdir resnet3d50csnir1x32_mixup_ep30 --num_workers 16

# python train.py --seed 2023 --drop_path_rate 0.2 --mixup_p 0.5 --mixup_alpha 0.2 --lr 1e-3 --backbone_lr 2e-4 --in_chans 1 --batch_size 16 --image_size 192 --start_z 16 --end_z -17 --shift_z 2 \
#     --fold 1 --gpus 4 --epochs 30 --logdir resnet3d50csnir1x32_mixup_ep30 --num_workers 6
# python eval_tta_fp16.py --seed 2022 --batch_size 16 --image_size 192 --fold 1 --logdir resnet3d50csnir1x32_mixup_ep30 --num_workers 16

# python train.py --seed 2024 --drop_path_rate 0.2 --mixup_p 0.5 --mixup_alpha 0.2 --lr 1e-3 --backbone_lr 2e-4 --in_chans 1 --batch_size 16 --image_size 192 --start_z 16 --end_z -17 --shift_z 2 \
#     --fold 2 --gpus 4 --epochs 30 --logdir resnet3d50csnir1x32_mixup_ep30 --num_workers 6
# python eval_tta_fp16.py --seed 2024 --batch_size 16 --image_size 192 --fold 2 --logdir resnet3d50csnir1x32_mixup_ep30 --num_workers 16

# python train.py --seed 2025 --drop_path_rate 0.2 --mixup_p 0.5 --mixup_alpha 0.2 --lr 1e-3 --backbone_lr 2e-4 --in_chans 1 --batch_size 16 --image_size 192 --start_z 16 --end_z -17 --shift_z 2 \
#     --fold 3 --gpus 4 --epochs 30 --logdir resnet3d50csnir1x32_mixup_ep30 --num_workers 6
# python eval_tta_fp16.py --seed 2025 --batch_size 16 --image_size 192 --fold 3 --logdir resnet3d50csnir1x32_mixup_ep30 --num_workers 16

# python train.py --seed 2026 --drop_path_rate 0.2 --mixup_p 0.5 --mixup_alpha 0.2 --lr 1e-3 --backbone_lr 2e-4 --in_chans 1 --batch_size 16 --image_size 192 --start_z 16 --end_z -17 --shift_z 2 \
#     --fold 4 --gpus 4 --epochs 30 --logdir resnet3d50csnir1x32_mixup_ep30 --num_workers 6
# python eval_tta_fp16.py --seed 2026 --batch_size 16 --image_size 192 --fold 4 --logdir resnet3d50csnir1x32_mixup_ep30 --num_workers 16


# python train.py --seed 8022 --depth 152 --drop_path_rate 0.2 --mixup_p 0.5 --mixup_alpha 0.2 --lr 1e-3 --backbone_lr 2e-4 --in_chans 1 --batch_size 16 --image_size 192 --start_z 20 --end_z -21 --shift_z 2 \
#     --fold 0 --gpus 4 --epochs 30 --logdir resnet3d152csnir1x24_mixup_ep30 --num_workers 6
# python eval_tta_fp16.py --seed 2022 --batch_size 16 --image_size 192 --fold 0 --logdir resnet3d152csnir1x24_mixup_ep30 --num_workers 16

# python train.py --seed 8023 --depth 152 --drop_path_rate 0.2 --mixup_p 0.5 --mixup_alpha 0.2 --lr 1e-3 --backbone_lr 2e-4 --in_chans 1 --batch_size 16 --image_size 192 --start_z 20 --end_z -21 --shift_z 2 \
#     --fold 1 --gpus 4 --epochs 30 --logdir resnet3d152csnir1x24_mixup_ep30 --num_workers 6
# python eval_tta_fp16.py --seed 2022 --batch_size 16 --image_size 192 --fold 1 --logdir resnet3d152csnir1x24_mixup_ep30 --num_workers 16

# python train.py --seed 8024 --depth 152 --drop_path_rate 0.2 --mixup_p 0.5 --mixup_alpha 0.2 --lr 1e-3 --backbone_lr 2e-4 --in_chans 1 --batch_size 16 --image_size 192 --start_z 20 --end_z -21 --shift_z 2 \
#     --fold 2 --gpus 4 --epochs 30 --logdir resnet3d152csnir1x24_mixup_ep30 --num_workers 6
# python eval_tta_fp16.py --seed 2024 --batch_size 16 --image_size 192 --fold 2 --logdir resnet3d152csnir1x24_mixup_ep30 --num_workers 16

# python train.py --seed 8025 --depth 152 --drop_path_rate 0.2 --mixup_p 0.5 --mixup_alpha 0.2 --lr 1e-3 --backbone_lr 2e-4 --in_chans 1 --batch_size 16 --image_size 192 --start_z 20 --end_z -21 --shift_z 2 \
#     --fold 3 --gpus 4 --epochs 30 --logdir resnet3d152csnir1x24_mixup_ep30 --num_workers 6
# python eval_tta_fp16.py --seed 2025 --batch_size 16 --image_size 192 --fold 3 --logdir resnet3d152csnir1x24_mixup_ep30 --num_workers 16

# python train.py --seed 8026 --depth 152 --drop_path_rate 0.2 --mixup_p 0.5 --mixup_alpha 0.2 --lr 1e-3 --backbone_lr 2e-4 --in_chans 1 --batch_size 16 --image_size 192 --start_z 20 --end_z -21 --shift_z 2 \
#     --fold 4 --gpus 4 --epochs 30 --logdir resnet3d152csnir1x24_mixup_ep30 --num_workers 6
# python eval_tta_fp16.py --seed 2026 --batch_size 16 --image_size 192 --fold 4 --logdir resnet3d152csnir1x24_mixup_ep30 --num_workers 16



# python train.py --seed 14022 --drop_path_rate 0.2 --mixup_p 0.5 --mixup_alpha 0.2 --lr 1e-3 --backbone_lr 2e-4 --in_chans 1 --batch_size 16 --image_size 224 --start_z 8 --end_z -9 --shift_z 2 --sampling_z 2\
#     --fold 0 --gpus 4 --epochs 30 --logdir resnet3d50csnir1x24s2_mixup_ep30_224 --num_workers 6
# python eval_tta_fp16.py --seed 2022 --batch_size 16 --image_size 224 --fold 0 --logdir resnet3d50csnir1x24s2_mixup_ep30_224 --num_workers 16

# python train.py --seed 14023 --drop_path_rate 0.2 --mixup_p 0.5 --mixup_alpha 0.2 --lr 1e-3 --backbone_lr 2e-4 --in_chans 1 --batch_size 16 --image_size 224 --start_z 8 --end_z -9 --shift_z 2 --sampling_z 2 \
#     --fold 1 --gpus 4 --epochs 30 --logdir resnet3d50csnir1x24s2_mixup_ep30_224 --num_workers 6
# python eval_tta_fp16.py --seed 12022 --batch_size 16 --image_size 224 --fold 1 --logdir resnet3d50csnir1x24s2_mixup_ep30_224 --num_workers 16

# python train.py --seed 14024 --drop_path_rate 0.2 --mixup_p 0.5 --mixup_alpha 0.2 --lr 1e-3 --backbone_lr 2e-4 --in_chans 1 --batch_size 16 --image_size 224 --start_z 8 --end_z -9 --shift_z 2 --sampling_z 2 \
#     --fold 2 --gpus 4 --epochs 30 --logdir resnet3d50csnir1x24s2_mixup_ep30_224 --num_workers 6
# python eval_tta_fp16.py --seed 12024 --batch_size 16 --image_size 224 --fold 2 --logdir resnet3d50csnir1x24s2_mixup_ep30_224 --num_workers 16

# python train.py --seed 14025 --drop_path_rate 0.2 --mixup_p 0.5 --mixup_alpha 0.2 --lr 1e-3 --backbone_lr 2e-4 --in_chans 1 --batch_size 16 --image_size 224 --start_z 8 --end_z -9 --shift_z 2 --sampling_z 2 \
#     --fold 3 --gpus 4 --epochs 30 --logdir resnet3d50csnir1x24s2_mixup_ep30_224 --num_workers 6
# python eval_tta_fp16.py --seed 12025 --batch_size 16 --image_size 224 --fold 3 --logdir resnet3d50csnir1x24s2_mixup_ep30_224 --num_workers 16

# python train.py --seed 14026 --drop_path_rate 0.2 --mixup_p 0.5 --mixup_alpha 0.2 --lr 1e-3 --backbone_lr 2e-4 --in_chans 1 --batch_size 16 --image_size 224 --start_z 8 --end_z -9 --shift_z 2 --sampling_z 2 \
#     --fold 4 --gpus 4 --epochs 30 --logdir resnet3d50csnir1x24s2_mixup_ep30_224 --num_workers 6
# python eval_tta_fp16.py --seed 12026 --batch_size 16 --image_size 224 --fold 4 --logdir resnet3d50csnir1x24s2_mixup_ep30_224 --num_workers 16



python train.py --seed 12022 --drop_path_rate 0.2 --mixup_p 0.5 --mixup_alpha 0.2 --lr 5e-4 --backbone_lr 1e-4 --in_chans 1 --batch_size 8 --image_size 224 --start_z 16 --end_z -17 --shift_z 2 \
    --fold 0 --gpus 4 --epochs 30 --logdir resnet3d50csnir1x32_mixup_ep30_224 --num_workers 6
python eval_tta_fp16.py --seed 2022 --batch_size 16 --image_size 224 --fold 0 --logdir resnet3d50csnir1x32_mixup_ep30_224 --num_workers 16

python train.py --seed 12023 --drop_path_rate 0.2 --mixup_p 0.5 --mixup_alpha 0.2 --lr 5e-4 --backbone_lr 1e-4 --in_chans 1 --batch_size 8 --image_size 224 --start_z 16 --end_z -17 --shift_z 2 \
    --fold 1 --gpus 4 --epochs 30 --logdir resnet3d50csnir1x32_mixup_ep30_224 --num_workers 6
python eval_tta_fp16.py --seed 12022 --batch_size 16 --image_size 224 --fold 1 --logdir resnet3d50csnir1x32_mixup_ep30_224 --num_workers 16

python train.py --seed 12024 --drop_path_rate 0.2 --mixup_p 0.5 --mixup_alpha 0.2 --lr 5e-4 --backbone_lr 1e-4 --in_chans 1 --batch_size 8 --image_size 224 --start_z 16 --end_z -17 --shift_z 2 \
    --fold 2 --gpus 4 --epochs 30 --logdir resnet3d50csnir1x32_mixup_ep30_224 --num_workers 6
python eval_tta_fp16.py --seed 12024 --batch_size 16 --image_size 224 --fold 2 --logdir resnet3d50csnir1x32_mixup_ep30_224 --num_workers 16

python train.py --seed 12025 --drop_path_rate 0.2 --mixup_p 0.5 --mixup_alpha 0.2 --lr 5e-4 --backbone_lr 1e-4 --in_chans 1 --batch_size 8 --image_size 224 --start_z 16 --end_z -17 --shift_z 2 \
    --fold 3 --gpus 4 --epochs 30 --logdir resnet3d50csnir1x32_mixup_ep30_224 --num_workers 6
python eval_tta_fp16.py --seed 12025 --batch_size 16 --image_size 224 --fold 3 --logdir resnet3d50csnir1x32_mixup_ep30_224 --num_workers 16

python train.py --seed 12026 --drop_path_rate 0.2 --mixup_p 0.5 --mixup_alpha 0.2 --lr 5e-4 --backbone_lr 1e-4 --in_chans 1 --batch_size 8 --image_size 224 --start_z 16 --end_z -17 --shift_z 2 \
    --fold 4 --gpus 4 --epochs 30 --logdir resnet3d50csnir1x32_mixup_ep30_224 --num_workers 6
python eval_tta_fp16.py --seed 12026 --batch_size 16 --image_size 224 --fold 4 --logdir resnet3d50csnir1x32_mixup_ep30_224 --num_workers 16


python train.py --seed 13022 --depth 152 --drop_path_rate 0.2 --mixup_p 0.5 --mixup_alpha 0.2 --lr 5e-4 --backbone_lr 1e-4 --in_chans 1 --batch_size 8 --image_size 224 --start_z 20 --end_z -21 --shift_z 2 \
    --fold 0 --gpus 4 --epochs 30 --logdir resnet3d152csnir1x24_mixup_ep30_224 --num_workers 6
python eval_tta_fp16.py --seed 2022 --batch_size 16 --image_size 224 --fold 0 --logdir resnet3d152csnir1x24_mixup_ep30_224 --num_workers 16

python train.py --seed 13023 --depth 152 --drop_path_rate 0.2 --mixup_p 0.5 --mixup_alpha 0.2 --lr 5e-4 --backbone_lr 1e-4 --in_chans 1 --batch_size 8 --image_size 224 --start_z 20 --end_z -21 --shift_z 2 \
    --fold 1 --gpus 4 --epochs 30 --logdir resnet3d152csnir1x24_mixup_ep30_224 --num_workers 6
python eval_tta_fp16.py --seed 2022 --batch_size 16 --image_size 224 --fold 1 --logdir resnet3d152csnir1x24_mixup_ep30_224 --num_workers 16

python train.py --seed 13024 --depth 152 --drop_path_rate 0.2 --mixup_p 0.5 --mixup_alpha 0.2 --lr 5e-4 --backbone_lr 1e-4 --in_chans 1 --batch_size 8 --image_size 224 --start_z 20 --end_z -21 --shift_z 2 \
    --fold 2 --gpus 4 --epochs 30 --logdir resnet3d152csnir1x24_mixup_ep30_224 --num_workers 6
python eval_tta_fp16.py --seed 2024 --batch_size 16 --image_size 224 --fold 2 --logdir resnet3d152csnir1x24_mixup_ep30_224 --num_workers 16

python train.py --seed 13025 --depth 152 --drop_path_rate 0.2 --mixup_p 0.5 --mixup_alpha 0.2 --lr 5e-4 --backbone_lr 1e-4 --in_chans 1 --batch_size 8 --image_size 224 --start_z 20 --end_z -21 --shift_z 2 \
    --fold 3 --gpus 4 --epochs 30 --logdir resnet3d152csnir1x24_mixup_ep30_224 --num_workers 6
python eval_tta_fp16.py --seed 2025 --batch_size 16 --image_size 224 --fold 3 --logdir resnet3d152csnir1x24_mixup_ep30_224 --num_workers 16

python train.py --seed 13026 --depth 152 --drop_path_rate 0.2 --mixup_p 0.5 --mixup_alpha 0.2 --lr 5e-4 --backbone_lr 1e-4 --in_chans 1 --batch_size 8 --image_size 224 --start_z 20 --end_z -21 --shift_z 2 \
    --fold 4 --gpus 4 --epochs 30 --logdir resnet3d152csnir1x24_mixup_ep30_224 --num_workers 6
python eval_tta_fp16.py --seed 2026 --batch_size 16 --image_size 224 --fold 4 --logdir resnet3d152csnir1x24_mixup_ep30_224 --num_workers 16
