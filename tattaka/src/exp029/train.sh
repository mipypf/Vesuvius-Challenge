# python train.py --seed 2022 --model_name tf_efficientnet_b2_ns --drop_path 0.4 --drop_path_rate 0.2 --num_3d_layers 3 --in_chans 7 \
#     --mixup_p 0.5 --mixup_alpha 0.2  --lr 5e-3  --batch_size 16 --image_size 256 --start_z 8 --end_z -8 --shift_z 2 \
#     --fold 0 --gpus 4 --epochs 25 --logdir effnet_b2_split3d7x7csn_mixup --num_workers 6
# python eval.py --seed 2022 --batch_size 16 --image_size 256 --fold 0 --logdir effnet_b2_split3d7x7csn_mixup --num_workers 16
# python eval_tta.py --seed 2022 --batch_size 16 --image_size 256 --fold 0 --logdir effnet_b2_split3d7x7csn_mixup --num_workers 16

# python train.py --seed 2024 --model_name tf_efficientnet_b2_ns --drop_path 0.4 --drop_path_rate 0.2 --num_3d_layers 3 --in_chans 7 \
#     --mixup_p 0.5 --mixup_alpha 0.2  --lr 5e-3  --batch_size 16 --image_size 256 --start_z 8 --end_z -8 --shift_z 2 \
#     --fold 2 --gpus 4 --epochs 25 --logdir effnet_b2_split3d7x7csn_mixup --num_workers 6
# python eval.py --seed 2024 --batch_size 16 --image_size 256 --fold 2 --logdir effnet_b2_split3d7x7csn_mixup --num_workers 16
# python eval_tta.py --seed 2024 --batch_size 16 --image_size 256 --fold 2 --logdir effnet_b2_split3d7x7csn_mixup --num_workers 16


python train.py --seed 2022 --model_name tf_efficientnet_b2_ns --drop_path 0.4 --drop_path_rate 0.2 --num_3d_layers 3 --in_chans 5 \
    --mixup_p 0.5 --mixup_alpha 0.2  --lr 5e-3  --batch_size 16 --image_size 256 --start_z 15 --end_z -15 --shift_z 2 \
    --fold 0 --gpus 4 --epochs 25 --logdir effnet_b2_split3d5x7csn_mixup --num_workers 6
python eval.py --seed 2022 --batch_size 16 --image_size 256 --fold 0 --logdir effnet_b2_split3d5x7csn_mixup --num_workers 16
python eval_tta.py --seed 2022 --batch_size 16 --image_size 256 --fold 0 --logdir effnet_b2_split3d5x7csn_mixup --num_workers 16

python train.py --seed 2024 --model_name tf_efficientnet_b2_ns --drop_path 0.4 --drop_path_rate 0.2 --num_3d_layers 3 --in_chans 5 \
    --mixup_p 0.5 --mixup_alpha 0.2  --lr 5e-3  --batch_size 16 --image_size 256 --start_z 15 --end_z -15 --shift_z 2 \
    --fold 2 --gpus 4 --epochs 25 --logdir effnet_b2_split3d5x7csn_mixup --num_workers 6
python eval.py --seed 2024 --batch_size 16 --image_size 256 --fold 2 --logdir effnet_b2_split3d5x7csn_mixup --num_workers 16
python eval_tta.py --seed 2024 --batch_size 16 --image_size 256 --fold 2 --logdir effnet_b2_split3d5x7csn_mixup --num_workers 16


python train.py --seed 2022 --model_name tf_efficientnet_b2_ns --drop_path 0.4 --drop_path_rate 0.2 --num_3d_layers 6 --in_chans 3 \
    --mixup_p 0.5 --mixup_alpha 0.2  --lr 5e-3  --batch_size 16 --image_size 256 --start_z 19 --end_z -19 --shift_z 2 \
    --fold 0 --gpus 4 --epochs 25 --logdir effnet_b2_split3d3x9csn_l6_mixup --num_workers 6
python eval.py --seed 2022 --batch_size 16 --image_size 256 --fold 0 --logdir effnet_b2_split3d3x9csn_l6_mixup --num_workers 16
python eval_tta.py --seed 2022 --batch_size 16 --image_size 256 --fold 0 --logdir effnet_b2_split3d3x9csn_l6_mixup --num_workers 16

python train.py --seed 2024 --model_name tf_efficientnet_b2_ns --drop_path 0.4 --drop_path_rate 0.2 --num_3d_layers 6 --in_chans 3 \
    --mixup_p 0.5 --mixup_alpha 0.2  --lr 5e-3  --batch_size 16 --image_size 256 --start_z 19 --end_z -19 --shift_z 2 \
    --fold 2 --gpus 4 --epochs 25 --logdir effnet_b2_split3d3x9csn_l6_mixup --num_workers 6
python eval.py --seed 2024 --batch_size 16 --image_size 256 --fold 2 --logdir effnet_b2_split3d3x9csn_l6_mixup --num_workers 16
python eval_tta.py --seed 2024 --batch_size 16 --image_size 256 --fold 2 --logdir effnet_b2_split3d3x9csn_l6_mixup --num_workers 16
