# python train.py --seed 2022 --model_name resnetrs50 --lr 3e-4 --batch_size 16 --image_size 256 --num_3d_layers 3 --mixup_p 0.5 --mixup_alpha 0.2 --fold 0 --gpus 4 --epochs 25 --logdir resnetrs50_split3d7x7csn_mixup --num_workers 6
# python eval.py --seed 2022 --batch_size 16 --image_size 256 --fold 0 --logdir resnetrs50_split3d7x7csn_mixup --num_workers 16
# python eval_tta.py --seed 2022 --batch_size 16 --image_size 256 --fold 0 --logdir resnetrs50_split3d7x7csn_mixup --num_workers 16

# python train.py --seed 2023 --model_name resnetrs50 --lr 3e-4 --batch_size 16 --image_size 256  --num_3d_layers 3 --mixup_p 0.5 --mixup_alpha 0.2 --fold 1 --gpus 4 --epochs 25 --logdir resnetrs50_split3d7x7csn_mixup --num_workers 6
# python eval.py --seed 2023 --batch_size 16 --image_size 256 --fold 1 --logdir resnetrs50_split3d7x7csn_mixup --num_workers 16
# python eval_tta.py --seed 2023 --batch_size 16 --image_size 256 --fold 1 --logdir resnetrs50_split3d7x7csn_mixup --num_workers 16

# python train.py --seed 2024 --model_name resnetrs50 --lr 3e-4 --batch_size 16 --image_size 256  --num_3d_layers 3 --mixup_p 0.5 --mixup_alpha 0.2 --fold 2 --gpus 4 --epochs 25 --logdir resnetrs50_split3d7x7csn_mixup --num_workers 6
# python eval.py --seed 2024 --batch_size 16 --image_size 256 --fold 2 --logdir resnetrs50_split3d7x7csn_mixup --num_workers 16
# python eval_tta.py --seed 2024 --batch_size 16 --image_size 256 --fold 2 --logdir resnetrs50_split3d7x7csn_mixup --num_workers 16


# python train.py --seed 2022 --model_name resnetrs50 --lr 2e-4 --batch_size 8 --image_size 384 --num_3d_layers 3 --mixup_p 0.5 --mixup_alpha 0.2 --fold 0 --gpus 4 --epochs 25 --logdir resnetrs50_split3d7x7csn_mixup_384 --num_workers 6
# python eval.py --seed 2022 --batch_size 16 --image_size 384 --fold 0 --logdir resnetrs50_split3d7x7csn_mixup_384 --num_workers 16
# python eval_tta.py --seed 2022 --batch_size 16 --image_size 384 --fold 0 --logdir resnetrs50_split3d7x7csn_mixup_384 --num_workers 16

# python train.py --seed 2023 --model_name resnetrs50 --lr 2e-4 --batch_size 8 --image_size 384  --num_3d_layers 3 --mixup_p 0.5 --mixup_alpha 0.2 --fold 1 --gpus 4 --epochs 25 --logdir resnetrs50_split3d7x7csn_mixup_384 --num_workers 6
# python eval.py --seed 2023 --batch_size 16 --image_size 384 --fold 1 --logdir resnetrs50_split3d7x7csn_mixup_384 --num_workers 16
# python eval_tta.py --seed 2023 --batch_size 16 --image_size 384 --fold 1 --logdir resnetrs50_split3d7x7csn_mixup_384 --num_workers 16

# python train.py --seed 2024 --model_name resnetrs50 --lr 2e-4 --batch_size 8 --image_size 384  --num_3d_layers 3 --mixup_p 0.5 --mixup_alpha 0.2 --fold 2 --gpus 4 --epochs 25 --logdir resnetrs50_split3d7x7csn_mixup_384 --num_workers 6
# python eval.py --seed 2024 --batch_size 16 --image_size 384 --fold 2 --logdir resnetrs50_split3d7x7csn_mixup_384 --num_workers 16
# python eval_tta.py --seed 2024 --batch_size 16 --image_size 384 --fold 2 --logdir resnetrs50_split3d7x7csn_mixup_384 --num_workers 16


python train.py --seed 2022 --model_name resnetrs50 --lr 3e-4 --batch_size 16 --image_size 192 --num_3d_layers 3 --mixup_p 0.5 --mixup_alpha 0.2 --fold 0 --gpus 4 --epochs 25 --logdir resnetrs50_split3d7x7csn_mixup_192 --num_workers 6
python eval.py --seed 2022 --batch_size 16 --image_size 192 --fold 0 --logdir resnetrs50_split3d7x7csn_mixup_192 --num_workers 16
python eval_tta.py --seed 2022 --batch_size 16 --image_size 192 --fold 0 --logdir resnetrs50_split3d7x7csn_mixup_192 --num_workers 16

python train.py --seed 2023 --model_name resnetrs50 --lr 3e-4 --batch_size 16 --image_size 192  --num_3d_layers 3 --mixup_p 0.5 --mixup_alpha 0.2 --fold 1 --gpus 4 --epochs 25 --logdir resnetrs50_split3d7x7csn_mixup_192 --num_workers 6
python eval.py --seed 2023 --batch_size 16 --image_size 192 --fold 1 --logdir resnetrs50_split3d7x7csn_mixup_192 --num_workers 16
python eval_tta.py --seed 2023 --batch_size 16 --image_size 192 --fold 1 --logdir resnetrs50_split3d7x7csn_mixup_192 --num_workers 16

python train.py --seed 2024 --model_name resnetrs50 --lr 3e-4 --batch_size 16 --image_size 192  --num_3d_layers 3 --mixup_p 0.5 --mixup_alpha 0.2 --fold 2 --gpus 4 --epochs 25 --logdir resnetrs50_split3d7x7csn_mixup_192 --num_workers 6
python eval.py --seed 2024 --batch_size 16 --image_size 192 --fold 2 --logdir resnetrs50_split3d7x7csn_mixup_192 --num_workers 16
python eval_tta.py --seed 2024 --batch_size 16 --image_size 192 --fold 2 --logdir resnetrs50_split3d7x7csn_mixup_192 --num_workers 16


# python train.py --seed 2022 --model_name resnetrs50 --lr 3e-4 --batch_size 16 --image_size 256 --num_3d_layers 3 --mixup_p 0.5 --mixup_alpha 0.2 --fold 0 --gpus 4 --epochs 25 --logdir resnetrs50_split3d7x7csn_mixup --num_workers 6
# python eval.py --seed 2022 --batch_size 16 --image_size 256 --fold 0 --logdir resnetrs50_split3d7x7csn_mixup --num_workers 16
# python eval_tta.py --seed 2022 --batch_size 16 --image_size 256 --fold 0 --logdir resnetrs50_split3d7x7csn_mixup --num_workers 16

# python train.py --seed 2023 --model_name resnetrs50 --lr 3e-4 --batch_size 16 --image_size 256  --num_3d_layers 3 --mixup_p 0.5 --mixup_alpha 0.2 --fold 1 --gpus 4 --epochs 25 --logdir resnetrs50_split3d7x7csn_mixup --num_workers 6
# python eval.py --seed 2023 --batch_size 16 --image_size 256 --fold 1 --logdir resnetrs50_split3d7x7csn_mixup --num_workers 16
# python eval_tta.py --seed 2023 --batch_size 16 --image_size 256 --fold 1 --logdir resnetrs50_split3d7x7csn_mixup --num_workers 16

# python train.py --seed 2024 --model_name resnetrs50 --lr 3e-4 --batch_size 16 --image_size 256  --num_3d_layers 3 --mixup_p 0.5 --mixup_alpha 0.2 --fold 2 --gpus 4 --epochs 25 --logdir resnetrs50_split3d7x7csn_mixup --num_workers 6
# python eval.py --seed 2024 --batch_size 16 --image_size 256 --fold 2 --logdir resnetrs50_split3d7x7csn_mixup --num_workers 16
# python eval_tta.py --seed 2024 --batch_size 16 --image_size 256 --fold 2 --logdir resnetrs50_split3d7x7csn_mixup --num_workers 16