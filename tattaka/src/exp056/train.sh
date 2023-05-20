python train.py --seed 2022 --drop_path_rate 0.4 --mixup_p 0.5 --mixup_alpha 0.2 --lr 1e-3 --in_chans 2 --batch_size 16 --image_size 256 --start_z 16 --end_z -17 --shift_z 2 \
    --fold 0 --gpus 4 --epochs 30 --logdir resnet3d50csnir2x16_mixup_ep30 --num_workers 6
python eval_tta.py --seed 2022 --batch_size 16 --image_size 256 --fold 0 --logdir resnet3d50csnir2x16_mixup_ep30 --num_workers 16

python train.py --seed 2023 --drop_path_rate 0.4 --mixup_p 0.5 --mixup_alpha 0.2 --lr 1e-3  --in_chans 2 --batch_size 16 --image_size 256 --start_z 16 --end_z -17 --shift_z 2 \
    --fold 1 --gpus 4 --epochs 30 --logdir resnet3d50csnir2x16_mixup_ep30 --num_workers 6
python eval_tta.py --seed 2022 --batch_size 16 --image_size 256 --fold 1 --logdir resnet3d50csnir2x16_mixup_ep30 --num_workers 16

python train.py --seed 2024 --drop_path_rate 0.4 --mixup_p 0.5 --mixup_alpha 0.2 --lr 1e-3   --in_chans 2 --batch_size 16 --image_size 256 --start_z 16 --end_z -17 --shift_z 2 \
    --fold 2 --gpus 4 --epochs 30 --logdir resnet3d50csnir2x16_mixup_ep30 --num_workers 6
python eval_tta.py --seed 2024 --batch_size 16 --image_size 256 --fold 2 --logdir resnet3d50csnir2x16_mixup_ep30 --num_workers 16

python train.py --seed 2025 --drop_path_rate 0.4 --mixup_p 0.5 --mixup_alpha 0.2 --lr 1e-3  --in_chans 2 --batch_size 16 --image_size 256 --start_z 16 --end_z -17 --shift_z 2 \
    --fold 3 --gpus 4 --epochs 30 --logdir resnet3d50csnir2x16_mixup_ep30 --num_workers 6
python eval_tta.py --seed 2025 --batch_size 16 --image_size 256 --fold 3 --logdir resnet3d50csnir2x16_mixup_ep30 --num_workers 16

python train.py --seed 2026 --drop_path_rate 0.4 --mixup_p 0.5 --mixup_alpha 0.2 --lr 1e-3  --in_chans 2 --batch_size 16 --image_size 256 --start_z 16 --end_z -17 --shift_z 2 \
    --fold 4 --gpus 4 --epochs 30 --logdir resnet3d50csnir2x16_mixup_ep30 --num_workers 6
python eval_tta.py --seed 2026 --batch_size 16 --image_size 256 --fold 4 --logdir resnet3d50csnir2x16_mixup_ep30 --num_workers 16

