import argparse
import datetime
import warnings
from glob import glob

import numpy as np
import PIL.Image as Image
import pytorch_lightning as pl
import torch
from torch.nn import functional as F
from tqdm import tqdm
from train import (
    EXP_ID,
    InkDetDataModule,
    InkDetLightningModel,
    fbeta_score,
    find_threshold_percentile,
)

"""
resnetrs50_split3d7x7csn_mixup/fold1: score: 0.6015447165094452, threshold: 0.8991210937500009
resnetrs50_split3d7x7csn_mixup/fold2: score: 0.6583562762700641, threshold: 0.901171875000001
resnetrs50_split3d7x7csn_mixup/fold3: score: 0.5776996422459107, threshold: 0.9287109375000009
resnetrs50_split3d7x7csn_mixup/fold4: score: 0.6937344481603746, threshold: 0.893945312500001
resnetrs50_split3d7x7csn_mixup/fold5: score: 0.7102958801671159, threshold: 0.900878906250001
mean: 0.6483261926705821, threshold: 0.904765625000001
fold1, 2: 0.58955


resnetrs50_split3d6x7csn_mixup/fold1: score: 0.5750404205088424, threshold: 0.895410156250001
resnetrs50_split3d6x7csn_mixup/fold2: score: 0.6676349164456384, threshold: 0.9097656250000009
resnetrs50_split3d6x7csn_mixup/fold3: score: 0.5958787368233046, threshold: 0.9406250000000013
resnetrs50_split3d6x7csn_mixup/fold4: score: 0.7040523972834943, threshold: 0.8927734375000007
resnetrs50_split3d6x7csn_mixup/fold5: score: 0.7183815085056768, threshold: 0.8980468750000009
mean: 0.6521975959133913, threshold: 0.9073242187500009
fold1, 2: 0.58545


resnetrs50_split3d6x6csn_mixup/fold1: score: 0.5600364898141985, threshold: 0.9101562500000009
resnetrs50_split3d6x6csn_mixup/fold2: score: 0.6764470755479666, threshold: 0.9113281250000009
resnetrs50_split3d6x6csn_mixup/fold3: score: 0.6041788384180238, threshold: 0.929101562500001
resnetrs50_split3d6x6csn_mixup/fold4: score: 0.7023241621454789, threshold: 0.8823242187500009
resnetrs50_split3d6x6csn_mixup/fold5: score: 0.7192546992447728, threshold: 0.9019531250000009
mean: 0.6524482530340882, threshold: 0.906972656250001
fold1, 2: 0.5821

resnetrs50_split3d5x7csn_mixup/fold1: score: 0.5920165068557907, threshold: 0.8914062500000008
resnetrs50_split3d5x7csn_mixup/fold2: score: 0.690057853850225, threshold: 0.9047851562500009
resnetrs50_split3d5x7csn_mixup/fold3: score: 0.6026820573696071, threshold: 0.9298828125000009
resnetrs50_split3d5x7csn_mixup/fold4: score: 0.7055626925643979, threshold: 0.8900390625000011
resnetrs50_split3d5x7csn_mixup/fold5: score: 0.72202471046328, threshold: 0.8992187500000008
mean: 0.6624687642206601, threshold: 0.9030664062500009
fold1, 2: 0.5973

resnetrs50_split3d5x9csn_mixup/fold1: score: 0.5783643234411723, threshold: 0.907617187500001
resnetrs50_split3d5x9csn_mixup/fold2: score: 0.6865175341148777, threshold: 0.9042968750000008
resnetrs50_split3d5x9csn_mixup/fold3: score: 0.5944532617021986, threshold: 0.9269531250000009
resnetrs50_split3d5x9csn_mixup/fold4: score: 0.712603417311144, threshold: 0.8800781250000008
resnetrs50_split3d5x9csn_mixup/fold5: score: 0.7276915696785055, threshold: 0.897265625000001
mean: 0.6599260212495796, threshold: 0.903242187500001
fold1, 2: 0.58635

# resnetrs50_split3d4x9csn_mixup/fold1: score: 0.3825028280002426, threshold: 0.858789062500001
# resnetrs50_split3d4x9csn_mixup/fold2: score: 0.6675098429307952, threshold: 0.9112304687500009
# resnetrs50_split3d4x9csn_mixup/fold3: score: 0.5847637458480953, threshold: 0.9289062500000009
# resnetrs50_split3d4x9csn_mixup/fold4: score: 0.7213017571905146, threshold: 0.8943359375000008
# resnetrs50_split3d4x9csn_mixup/fold5: score: 0.7319721758146208, threshold: 0.893359375000001

resnetrs50_split3d3x9csn_mixup/fold1: score: 0.5730478912368729, threshold: 0.8943359375000008
resnetrs50_split3d3x9csn_mixup/fold2: score: 0.6592843091711791, threshold: 0.9064453125000009
resnetrs50_split3d3x9csn_mixup/fold3: score: 0.6451124668703048, threshold: 0.9304687500000011
resnetrs50_split3d3x9csn_mixup/fold4: score: 0.721976460229544, threshold: 0.8833007812500009
resnetrs50_split3d3x9csn_mixup/fold5: score: 0.7355351571374202, threshold: 0.8996093750000009
mean: 0.6669912569290641, threshold: 0.9028320312500007
fold1, 2: 0.60905
"""
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_args() -> argparse.Namespace:
    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument(
        "--seed",
        default=2022,
        type=int,
        metavar="SE",
        help="seed number",
        dest="seed",
    )
    dt_now = datetime.datetime.now()
    parent_parser.add_argument(
        "--logdir",
        default=f"{dt_now.strftime('%Y%m%d-%H-%M-%S')}",
    )
    parent_parser.add_argument(
        "--fold",
        type=int,
        default=0,
    )
    parser = InkDetDataModule.add_model_specific_args(parent_parser)
    return parser.parse_args()


def main(args):
    pl.seed_everything(args.seed)
    warnings.simplefilter("ignore")
    fragment_ids = [1, 2, 3, 4, 5]
    for i, valid_idx in enumerate(fragment_ids):
        if args.fold != i:
            continue
        valid_volume_paths = np.concatenate(
            [
                np.asarray(
                    sorted(
                        glob(
                            f"../../input/vesuvius_patches_32_5fold/train/{fragment_id}/surface_volume/**/*.npy",
                            recursive=True,
                        )
                    )
                )
                for fragment_id in fragment_ids
                if fragment_id == valid_idx
            ]
        )

        dataloader = InkDetDataModule(
            train_volume_paths=valid_volume_paths,
            valid_volume_paths=valid_volume_paths,
            image_size=args.image_size,
            num_workers=args.num_workers,
            batch_size=args.batch_size,
            preprocess_in_model=True,
        ).test_dataloader()

        logdir = f"../../logs/exp{EXP_ID}/{args.logdir}/fold{i}"
        ckpt_path = glob(
            f"{logdir}/**/best_fbeta.ckpt",
            recursive=True,
        )[0]
        print(f"ckpt_path = {ckpt_path}")
        model = InkDetLightningModel.load_from_checkpoint(
            ckpt_path,
            valid_fragment_id=valid_idx,
            pretrained=False,
            preprocess_in_model=True,
        )
        model.eval()
        model = model.to(device=device)
        y_valid = np.array(
            Image.open(
                f"../../input/vesuvius-challenge-ink-detection-5fold/train/{valid_idx}/inklabels.png"
            ).convert("1")
        )
        p_valid = np.zeros_like(y_valid, dtype=np.float32)
        count_pix = np.zeros_like(y_valid, dtype=np.float32)
        tta_set = [
            "vanilla",
            "flip_v",
            "flip_h",
            "flip_vh",
            "flip_c",
            "flip_vc",
            "flip_hc",
            "flip_vhc",
        ]
        for batch in tqdm(dataloader):
            volume, _, x, y = batch
            for i in range(len(volume)):
                if i % len(tta_set) == 1:
                    volume[i] = volume[i].flip(1)
                elif i % len(tta_set) == 2:
                    volume[i] = volume[i].flip(2)
                elif i % len(tta_set) == 3:
                    volume[i] = volume[i].flip(1).flip(2)
                elif i % len(tta_set) == 4:
                    volume[i] = volume[i].flip(0)
                elif i % len(tta_set) == 5:
                    volume[i] = volume[i].flip(0).flip(1)
                elif i % len(tta_set) == 6:
                    volume[i] = volume[i].flip(0).flip(2)
                elif i % len(tta_set) == 7:
                    volume[i] = volume[i].flip(0).flip(1).flip(2)

            volume = volume.to(device)
            with torch.no_grad():
                pred_batch = torch.sigmoid(model.model(volume))

            for i in range(len(pred_batch)):
                if i % len(tta_set) == 1:
                    pred_batch[i] = pred_batch[i].flip(1)
                elif i % len(tta_set) == 2:
                    pred_batch[i] = pred_batch[i].flip(2)
                elif i % len(tta_set) == 3:
                    pred_batch[i] = pred_batch[i].flip(1).flip(2)
                elif i % len(tta_set) == 4:
                    pred_batch[i] = pred_batch[i].flip(0)
                elif i % len(tta_set) == 5:
                    pred_batch[i] = pred_batch[i].flip(0).flip(1)
                elif i % len(tta_set) == 6:
                    pred_batch[i] = pred_batch[i].flip(0).flip(2)
                elif i % len(tta_set) == 7:
                    pred_batch[i] = pred_batch[i].flip(0).flip(1).flip(2)

            pred_batch = F.interpolate(
                pred_batch.detach().to(torch.float32).cpu(),
                scale_factor=32,
                mode="bilinear",
                align_corners=True,
            ).numpy()
            for xi, yi, pi in zip(
                x,
                y,
                pred_batch,
            ):
                y_lim, x_lim = y_valid[
                    yi * 32 : yi * 32 + volume.shape[-2],
                    xi * 32 : xi * 32 + volume.shape[-1],
                ].shape
                p_valid[
                    yi * 32 : yi * 32 + volume.shape[-2],
                    xi * 32 : xi * 32 + volume.shape[-1],
                ] += pi[0, :y_lim, :x_lim]
                count_pix[
                    yi * 32 : yi * 32 + volume.shape[-2],
                    xi * 32 : xi * 32 + volume.shape[-1],
                ] += np.ones_like(pi[0, :y_lim, :x_lim])
        fragment_mask = np.array(
            Image.open(
                f"../../input/vesuvius-challenge-ink-detection-5fold/train/{valid_idx}/mask.png"
            ).convert("1")
        )
        count_pix *= fragment_mask
        p_valid /= count_pix
        p_valid = np.nan_to_num(p_valid)
        count_pix = count_pix > 0
        p_valid *= fragment_mask
        p_valid_tmp = p_valid.reshape(-1)[np.where(count_pix.reshape(-1))]
        y_valid_tmp = y_valid.reshape(-1)[np.where(count_pix.reshape(-1))]
        threshold = find_threshold_percentile(y_valid_tmp, p_valid_tmp)
        np.save(f"{logdir}/logits", p_valid)
        p_valid = p_valid > np.quantile(p_valid_tmp, threshold)
        score = fbeta_score(y_valid, p_valid, beta=0.5)
        print(f"{args.logdir}/fold{valid_idx}: score: {score}, threshold: {threshold}")
        np.save(
            f"{logdir}/preds",
            (p_valid > np.quantile(p_valid_tmp, threshold)).astype(int),
        )
        np.save(f"{logdir}/true", y_valid)


if __name__ == "__main__":
    main(get_args())
