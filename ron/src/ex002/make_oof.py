# import argparse
# import datetime
# import os
# import warnings
# from glob import glob

# import numpy as np
# import PIL.Image as Image
# import pytorch_lightning as pl
# import torch
# from torch.nn import functional as F
# from tqdm import tqdm
# from train import EXP_ID, InkDetDataModule, InkDetLightningModel

# """
# """
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# def get_args() -> argparse.Namespace:
#     parent_parser = argparse.ArgumentParser(add_help=False)
#     parent_parser.add_argument(
#         "--seed",
#         default=2022,
#         type=int,
#         metavar="SE",
#         help="seed number",
#         dest="seed",
#     )
#     dt_now = datetime.datetime.now()
#     parent_parser.add_argument(
#         "--logdir",
#         default=f"{dt_now.strftime('%Y%m%d-%H-%M-%S')}",
#     )
#     parent_parser.add_argument(
#         "--fold",
#         type=int,
#         default=0,
#     )
#     parser = InkDetDataModule.add_model_specific_args(parent_parser)
#     return parser.parse_args()


# def main(args):
#     pl.seed_everything(args.seed)
#     warnings.simplefilter("ignore")
#     fragment_ids = [1, 2, 3, 4, 5]

#     for i, valid_idx in enumerate(fragment_ids):
#         if args.fold != i:
#             continue
#         valid_volume_paths = np.concatenate(
#             [
#                 np.asarray(
#                     sorted(
#                         glob(
#                             f"../../input/vesuvius_patches_32_5fold/train/{fragment_id}/surface_volume/**/*.npy",
#                             recursive=True,
#                         )
#                     )
#                 )
#                 for fragment_id in fragment_ids
#                 if fragment_id == valid_idx
#             ]
#         )

#         dataloader = InkDetDataModule(
#             train_volume_paths=valid_volume_paths,
#             valid_volume_paths=valid_volume_paths,
#             image_size=args.image_size,
#             num_workers=args.num_workers,
#             batch_size=args.batch_size,
#             preprocess_in_model=True,
#         ).test_dataloader()

#         logdir = f"../../logs/exp{EXP_ID}/{args.logdir}/fold{i}"
#         ckpt_names = ["best_fbeta.ckpt", "best_fbeta-v1.ckpt", "best_fbeta-v2.ckpt"]
#         for ci, ckpt_name in enumerate(ckpt_names):
#             ckpt_path = glob(
#                 f"{logdir}/**/{ckpt_name}",
#                 recursive=True,
#             )[0]
#             print(f"ckpt_path = {ckpt_path}")
#             model = InkDetLightningModel.load_from_checkpoint(
#                 ckpt_path,
#                 valid_fragment_id=valid_idx,
#                 pretrained=False,
#                 preprocess_in_model=True,
#             )
#             model.eval()
#             model = model.to(device=device)
#             y_valid = F.interpolate(
#                 torch.tensor(
#                     np.array(
#                         Image.open(
#                             f"../../input/vesuvius-challenge-ink-detection-5fold/train/{valid_idx}/inklabels.png"
#                         ).convert("1")
#                     ).astype(np.float32)
#                 )[None, None],
#                 scale_factor=1 / 32,
#                 mode="bilinear",
#                 align_corners=True,
#             )[0, 0].numpy()
#             p_valid = np.zeros_like(y_valid, dtype=np.float32)
#             count_pix = np.zeros_like(y_valid, dtype=np.float32)
#             tta_set = [
#                 "vanilla",
#                 "flip_v",
#                 "flip_h",
#                 "flip_vh",
#                 "flip_c",
#                 "flip_vc",
#                 "flip_hc",
#                 "flip_vhc",
#             ]
#             for batch in tqdm(dataloader):
#                 volume, _, x, y = batch
#                 for i in range(len(volume)):
#                     if i % len(tta_set) == 1:
#                         volume[i] = volume[i].flip(1)
#                     elif i % len(tta_set) == 2:
#                         volume[i] = volume[i].flip(2)
#                     elif i % len(tta_set) == 3:
#                         volume[i] = volume[i].flip(1).flip(2)
#                     elif i % len(tta_set) == 4:
#                         volume[i] = volume[i].flip(0)
#                     elif i % len(tta_set) == 5:
#                         volume[i] = volume[i].flip(0).flip(1)
#                     elif i % len(tta_set) == 6:
#                         volume[i] = volume[i].flip(0).flip(2)
#                     elif i % len(tta_set) == 7:
#                         volume[i] = volume[i].flip(0).flip(1).flip(2)

#                 volume = volume.to(device)
#                 with torch.no_grad():
#                     pred_batch = torch.sigmoid(model.model_ema.module(volume))

#                 for i in range(len(pred_batch)):
#                     if i % len(tta_set) == 1:
#                         pred_batch[i] = pred_batch[i].flip(1)
#                     elif i % len(tta_set) == 2:
#                         pred_batch[i] = pred_batch[i].flip(2)
#                     elif i % len(tta_set) == 3:
#                         pred_batch[i] = pred_batch[i].flip(1).flip(2)
#                     elif i % len(tta_set) == 4:
#                         pred_batch[i] = pred_batch[i].flip(0)
#                     elif i % len(tta_set) == 5:
#                         pred_batch[i] = pred_batch[i].flip(0).flip(1)
#                     elif i % len(tta_set) == 6:
#                         pred_batch[i] = pred_batch[i].flip(0).flip(2)
#                     elif i % len(tta_set) == 7:
#                         pred_batch[i] = pred_batch[i].flip(0).flip(1).flip(2)

#                 pred_batch = pred_batch.detach().to(torch.float32).cpu().numpy()
#                 for xi, yi, pi in zip(
#                     x,
#                     y,
#                     pred_batch,
#                 ):
#                     y_lim, x_lim = y_valid[
#                         yi : yi + pred_batch.shape[-2],
#                         xi : xi + pred_batch.shape[-1],
#                     ].shape
#                     p_valid[
#                         yi : yi + pred_batch.shape[-2],
#                         xi : xi + pred_batch.shape[-1],
#                     ] += pi[0, :y_lim, :x_lim]
#                     count_pix[
#                         yi : yi + pred_batch.shape[-2],
#                         xi : xi + pred_batch.shape[-1],
#                     ] += np.ones_like(pi[0, :y_lim, :x_lim])
#             fragment_mask = (
#                 F.interpolate(
#                     torch.tensor(
#                         np.array(
#                             Image.open(
#                                 f"../../input/vesuvius-challenge-ink-detection-5fold/train/{valid_idx}/mask.png"
#                             ).convert("1")
#                         ).astype(np.float32)
#                     )[None, None],
#                     scale_factor=1 / 32,
#                     mode="bilinear",
#                     align_corners=True,
#                 )[0, 0].numpy()
#                 > 0.5
#             )
#             count_pix *= fragment_mask
#             p_valid /= count_pix
#             p_valid = np.nan_to_num(p_valid)
#             count_pix = count_pix > 0
#             p_valid *= fragment_mask
#             os.makedirs(f"{logdir}/oof", exist_ok=True)
#             np.save(f"{logdir}/oof/oof_{ci}", p_valid)


# if __name__ == "__main__":
#     main(get_args())
