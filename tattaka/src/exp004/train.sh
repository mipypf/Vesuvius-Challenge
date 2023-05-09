# python train.py --seed 2022 --model_name resnetrs50 --lr 3e-4 --batch_size 16 --image_size 256 --fold 0 --gpus 4 --epochs 10 --logdir resnetrs50_baseline --num_workers 4
# python eval.py --seed 2022 --batch_size 16 --image_size 256 --fold 0 --logdir resnetrs50_baseline --num_workers 16

python train.py --seed 2022 --model_name resnetrs50 --lr 3e-4 --batch_size 16 --image_size 256 --mixup_p 0.5 --mixup_alpha 0.1 --fold 0 --gpus 4 --epochs 50 --logdir resnetrs50_short_mixup --num_workers 8
python eval.py --seed 2022 --batch_size 16 --image_size 256 --fold 0 --logdir resnetrs50_short_mixup --num_workers 16