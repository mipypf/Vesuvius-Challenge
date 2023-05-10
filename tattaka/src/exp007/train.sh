# python train.py --seed 2022 --model_name resnetrs50 --lr 3e-4 --batch_size 16 --image_size 256 --mixup_p 0.5 --mixup_alpha 0.2 --fold 0 --gpus 4 --epochs 25 --logdir resnetrs50_split_mixup --num_workers 6
# python eval.py --seed 2022 --batch_size 16 --image_size 256 --fold 0 --logdir resnetrs50_split_mixup --num_workers 16

# python train.py --seed 2022 --model_name resnetrs50 --lr 3e-4 --batch_size 16 --image_size 128 --mixup_p 0.5 --mixup_alpha 0.2 --fold 0 --gpus 4 --epochs 25 --logdir resnetrs50_split_mixup_128 --num_workers 6
# python eval.py --seed 2022 --batch_size 16 --image_size 128 --fold 0 --logdir resnetrs50_split_mixup_128 --num_workers 16

# python train.py --seed 2022 --model_name resnetrs50 --lr 3e-4 --batch_size 16 --image_size 256 --mixup_p 0.5 --mixup_alpha 0.2 --fold 1 --gpus 4 --epochs 25 --logdir resnetrs50_split_mixup --num_workers 6
# python eval.py --seed 2022 --batch_size 16 --image_size 256 --fold 1 --logdir resnetrs50_split_mixup --num_workers 16

python train.py --seed 2022 --model_name resnetrs50 --lr 3e-4 --batch_size 16 --image_size 256 --mixup_p 0.5 --mixup_alpha 0.2 --fold 2 --gpus 4 --epochs 25 --logdir resnetrs50_split_mixup --num_workers 6
python eval.py --seed 2022 --batch_size 16 --image_size 256 --fold 2 --logdir resnetrs50_split_mixup --num_workers 16