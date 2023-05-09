python train.py --seed 2022 --model_name resnetrs50 --drop_path 0.4 --drop_path_rate 0.2 --num_3d_layers 6 --in_chans 3 --subsample 2 \
    --mixup_p 0.5 --mixup_alpha 0.2  --lr 3e-4  --batch_size 16 --image_size 256 --start_z 5 --end_z -6 --shift_z 2 \
    --fold 0 --gpus 4 --epochs 25 --logdir resnetrs50_split3d3x9csn_l6_mixup --num_workers 6
python eval.py --seed 2022 --batch_size 16 --image_size 256 --fold 0 --logdir resnetrs50_split3d3x9csn_l6_mixup --num_workers 16
python eval_tta.py --seed 2022 --batch_size 16 --image_size 256 --fold 0 --logdir resnetrs50_split3d3x9csn_l6_mixup --num_workers 16

python train.py --seed 2024 --model_name resnetrs50 --drop_path 0.4 --drop_path_rate 0.2 --num_3d_layers 6 --in_chans 3  --subsample 2 \
    --mixup_p 0.5 --mixup_alpha 0.2  --lr 3e-4  --batch_size 16 --image_size 256 --start_z 5 --end_z -6 --shift_z 2 \
    --fold 2 --gpus 4 --epochs 25 --logdir resnetrs50_split3d3x9csn_l6_mixup --num_workers 6
python eval.py --seed 2024 --batch_size 16 --image_size 256 --fold 2 --logdir resnetrs50_split3d3x9csn_l6_mixup --num_workers 16
python eval_tta.py --seed 2024 --batch_size 16 --image_size 256 --fold 2 --logdir resnetrs50_split3d3x9csn_l6_mixup --num_workers 16


python train.py --seed 2022 --model_name resnetrs50 --drop_path 0.4 --drop_path_rate 0.2 --num_3d_layers 6 --in_chans 4 --subsample 2 \
    --mixup_p 0.5 --mixup_alpha 0.2  --lr 3e-4  --batch_size 16 --image_size 256 --start_z 4 --end_z -5 --shift_z 2 \
    --fold 0 --gpus 4 --epochs 25 --logdir resnetrs50_split3d4x7csn_l6_mixup --num_workers 6
python eval.py --seed 2022 --batch_size 16 --image_size 256 --fold 0 --logdir resnetrs50_split3d4x7csn_l6_mixup --num_workers 16
python eval_tta.py --seed 2022 --batch_size 16 --image_size 256 --fold 0 --logdir resnetrs50_split3d4x7csn_l6_mixup --num_workers 16

python train.py --seed 2024 --model_name resnetrs50 --drop_path 0.4 --drop_path_rate 0.2 --num_3d_layers 6 --in_chans 4  --subsample 2 \
    --mixup_p 0.5 --mixup_alpha 0.2  --lr 3e-4  --batch_size 16 --image_size 256 --start_z 4 --end_z -5 --shift_z 2 \
    --fold 2 --gpus 4 --epochs 25 --logdir resnetrs50_split3d4x7csn_l6_mixup --num_workers 6
python eval.py --seed 2024 --batch_size 16 --image_size 256 --fold 2 --logdir resnetrs50_split3d4x7csn_l6_mixup --num_workers 16
python eval_tta.py --seed 2024 --batch_size 16 --image_size 256 --fold 2 --logdir resnetrs50_split3d4x7csn_l6_mixup --num_workers 16