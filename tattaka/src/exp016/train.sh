# python train.py --seed 2022 --model_name resnetrs50 --lr 3e-4 --batch_size 16 --image_size 256 --num_3d_layers 3 --mixup_p 0.5 --mixup_alpha 0.2 --fold 0 --gpus 4 --epochs 25 --logdir resnetrs50_split3d7x7csn_mixup --num_workers 6
# python eval.py --seed 2022 --batch_size 16 --image_size 256 --fold 0 --logdir resnetrs50_split3d7x7csn_mixup --num_workers 16
# python eval_tta.py --seed 2022 --batch_size 16 --image_size 256 --fold 0 --logdir resnetrs50_split3d7x7csn_mixup --num_workers 16

# python train.py --seed 2023 --model_name resnetrs50 --lr 3e-4 --batch_size 16 --image_size 256  --num_3d_layers 3 --mixup_p 0.5 --mixup_alpha 0.2 --fold 1 --gpus 4 --epochs 25 --logdir resnetrs50_split3d7x7csn_mixup --num_workers 6
# python eval.py --seed 2023 --batch_size 16 --image_size 256 --fold 1 --logdir resnetrs50_split3d7x7csn_mixup --num_workers 16
# python eval_tta.py --seed 2023 --batch_size 16 --image_size 256 --fold 1 --logdir resnetrs50_split3d7x7csn_mixup --num_workers 16

python train.py --seed 2024 --model_name resnetrs50 --lr 3e-4 --batch_size 16 --image_size 256  --num_3d_layers 3 --mixup_p 0.5 --mixup_alpha 0.2 --fold 2 --gpus 4 --epochs 25 --logdir resnetrs50_split3d7x7csn_mixup --num_workers 6
python eval.py --seed 2024 --batch_size 16 --image_size 256 --fold 2 --logdir resnetrs50_split3d7x7csn_mixup --num_workers 16
python eval_tta.py --seed 2024 --batch_size 16 --image_size 256 --fold 2 --logdir resnetrs50_split3d7x7csn_mixup --num_workers 16

python train.py --seed 2025 --model_name resnetrs50 --lr 3e-4 --batch_size 16 --image_size 256  --num_3d_layers 3 --mixup_p 0.5 --mixup_alpha 0.2 --fold 3 --gpus 4 --epochs 25 --logdir resnetrs50_split3d7x7csn_mixup --num_workers 6
python eval.py --seed 2025 --batch_size 16 --image_size 256 --fold 3 --logdir resnetrs50_split3d7x7csn_mixup --num_workers 16
python eval_tta.py --seed 2025 --batch_size 16 --image_size 256 --fold 3 --logdir resnetrs50_split3d7x7csn_mixup --num_workers 16

python train.py --seed 2026 --model_name resnetrs50 --lr 3e-4 --batch_size 16 --image_size 256  --num_3d_layers 3 --mixup_p 0.5 --mixup_alpha 0.2 --fold 4 --gpus 4 --epochs 25 --logdir resnetrs50_split3d7x7csn_mixup --num_workers 6
python eval.py --seed 2026 --batch_size 16 --image_size 256 --fold 4 --logdir resnetrs50_split3d7x7csn_mixup --num_workers 16
python eval_tta.py --seed 2026 --batch_size 16 --image_size 256 --fold 4 --logdir resnetrs50_split3d7x7csn_mixup --num_workers 16



python train.py --seed 2022 --model_name resnetrs50 --lr 3e-4 --batch_size 16 --image_size 256 --num_3d_layers 3 --mixup_p 1 --mixup_alpha 0.2 --no_mixup_epochs 10 --fold 0 --gpus 4 --epochs 25 --logdir resnetrs50_split3d7x7csn_mixupv2 --num_workers 6
python eval.py --seed 2022 --batch_size 16 --image_size 256 --fold 0 --logdir resnetrs50_split3d7x7csn_mixupv2 --num_workers 16
python eval_tta.py --seed 2022 --batch_size 16 --image_size 256 --fold 0 --logdir resnetrs50_split3d7x7csn_mixupv2 --num_workers 16

python train.py --seed 2023 --model_name resnetrs50 --lr 3e-4 --batch_size 16 --image_size 256  --num_3d_layers 3 --mixup_p 1 --mixup_alpha 0.2 --no_mixup_epochs 10 --fold 1 --gpus 4 --epochs 25 --logdir resnetrs50_split3d7x7csn_mixupv2 --num_workers 6
python eval.py --seed 2023 --batch_size 16 --image_size 256 --fold 1 --logdir resnetrs50_split3d7x7csn_mixupv2 --num_workers 16
python eval_tta.py --seed 2023 --batch_size 16 --image_size 256 --fold 1 --logdir resnetrs50_split3d7x7csn_mixupv2 --num_workers 16

python train.py --seed 2024 --model_name resnetrs50 --lr 3e-4 --batch_size 16 --image_size 256  --num_3d_layers 3 --mixup_p 1 --mixup_alpha 0.2 --no_mixup_epochs 10 --fold 2 --gpus 4 --epochs 25 --logdir resnetrs50_split3d7x7csn_mixupv2 --num_workers 6
python eval.py --seed 2024 --batch_size 16 --image_size 256 --fold 2 --logdir resnetrs50_split3d7x7csn_mixupv2 --num_workers 16
python eval_tta.py --seed 2024 --batch_size 16 --image_size 256 --fold 2 --logdir resnetrs50_split3d7x7csn_mixupv2 --num_workers 16

python train.py --seed 2023 --model_name resnetrs50 --lr 3e-4 --batch_size 16 --image_size 256  --num_3d_layers 3 --mixup_p 1 --mixup_alpha 0.2 --no_mixup_epochs 10 --fold 3 --gpus 4 --epochs 25 --logdir resnetrs50_split3d7x7csn_mixupv2 --num_workers 6
python eval.py --seed 2023 --batch_size 16 --image_size 256 --fold 3 --logdir resnetrs50_split3d7x7csn_mixupv2 --num_workers 16
python eval_tta.py --seed 2023 --batch_size 16 --image_size 256 --fold 3 --logdir resnetrs50_split3d7x7csn_mixupv2 --num_workers 16

python train.py --seed 2024 --model_name resnetrs50 --lr 3e-4 --batch_size 16 --image_size 256  --num_3d_layers 3 --mixup_p 1 --mixup_alpha 0.2 --no_mixup_epochs 10 --fold 4 --gpus 4 --epochs 25 --logdir resnetrs50_split3d7x7csn_mixupv2 --num_workers 6
python eval.py --seed 2024 --batch_size 16 --image_size 256 --fold 4 --logdir resnetrs50_split3d7x7csn_mixupv2 --num_workers 16
python eval_tta.py --seed 2024 --batch_size 16 --image_size 256 --fold 4 --logdir resnetrs50_split3d7x7csn_mixupv2 --num_workers 16
