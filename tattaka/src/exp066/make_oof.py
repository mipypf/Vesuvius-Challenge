import argparse
import datetime
import os
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
Save oof:../../input/oof_5fold/fold0/exp055_resnetrs50_split3d5x7csn_mixup_ep30/oof_fbeta0, score: 0.6616110924110317
Save oof:../../input/oof_5fold/fold0/exp055_resnetrs50_split3d5x7csn_mixup_ep30/oof_fbeta1, score: 0.6585194546782994
Save oof:../../input/oof_5fold/fold0/exp055_resnetrs50_split3d5x7csn_mixup_ep30/oof_fbeta2, score: 0.6585671219047672

Save oof:../../input/oof_5fold/fold1/exp055_resnetrs50_split3d5x7csn_mixup_ep30/oof_fbeta0, score: 0.6766739264738048
Save oof:../../input/oof_5fold/fold1/exp055_resnetrs50_split3d5x7csn_mixup_ep30/oof_fbeta1, score: 0.667151853448103
Save oof:../../input/oof_5fold/fold1/exp055_resnetrs50_split3d5x7csn_mixup_ep30/oof_fbeta2, score: 0.6841792727399588

Save oof:../../input/oof_5fold/fold2/exp055_resnetrs50_split3d5x7csn_mixup_ep30/oof_fbeta0, score: 0.6893144750045893
Save oof:../../input/oof_5fold/fold2/exp055_resnetrs50_split3d5x7csn_mixup_ep30/oof_fbeta1, score: 0.6953917398684613
Save oof:../../input/oof_5fold/fold2/exp055_resnetrs50_split3d5x7csn_mixup_ep30/oof_fbeta2, score: 0.6857131076746803

Save oof:../../input/oof_5fold/fold3/exp055_resnetrs50_split3d5x7csn_mixup_ep30/oof_fbeta0, score: 0.7776853733102816
Save oof:../../input/oof_5fold/fold3/exp055_resnetrs50_split3d5x7csn_mixup_ep30/oof_fbeta1, score: 0.7811130843086694
Save oof:../../input/oof_5fold/fold3/exp055_resnetrs50_split3d5x7csn_mixup_ep30/oof_fbeta2, score: 0.7779789716979516

Save oof:../../input/oof_5fold/fold4/exp055_resnetrs50_split3d5x7csn_mixup_ep30/oof_fbeta0, score: 0.7456068233518977
Save oof:../../input/oof_5fold/fold4/exp055_resnetrs50_split3d5x7csn_mixup_ep30/oof_fbeta1, score: 0.7382901670261449
Save oof:../../input/oof_5fold/fold4/exp055_resnetrs50_split3d5x7csn_mixup_ep30/oof_fbeta2, score: 0.7362676560075636

Save oof:../../input/oof_5fold/fold0/exp055_resnetrs50_split3d3x9csn_l6_mixup_ep30/oof_fbeta0, score: 0.6475739401506333
Save oof:../../input/oof_5fold/fold0/exp055_resnetrs50_split3d3x9csn_l6_mixup_ep30/oof_fbeta1, score: 0.6556910947658888
Save oof:../../input/oof_5fold/fold0/exp055_resnetrs50_split3d3x9csn_l6_mixup_ep30/oof_fbeta2, score: 0.6512049794374971

Save oof:../../input/oof_5fold/fold1/exp055_resnetrs50_split3d3x9csn_l6_mixup_ep30/oof_fbeta0, score: 0.7226838982914648
Save oof:../../input/oof_5fold/fold1/exp055_resnetrs50_split3d3x9csn_l6_mixup_ep30/oof_fbeta1, score: 0.7172161555748804
Save oof:../../input/oof_5fold/fold1/exp055_resnetrs50_split3d3x9csn_l6_mixup_ep30/oof_fbeta2, score: 0.7262084781653851

Save oof:../../input/oof_5fold/fold2/exp055_resnetrs50_split3d3x9csn_l6_mixup_ep30/oof_fbeta0, score: 0.7112823609332312
Save oof:../../input/oof_5fold/fold2/exp055_resnetrs50_split3d3x9csn_l6_mixup_ep30/oof_fbeta1, score: 0.7105617289688445
Save oof:../../input/oof_5fold/fold2/exp055_resnetrs50_split3d3x9csn_l6_mixup_ep30/oof_fbeta2, score: 0.705047399721905

Save oof:../../input/oof_5fold/fold3/exp055_resnetrs50_split3d3x9csn_l6_mixup_ep30/oof_fbeta0, score: 0.7807499937074391
Save oof:../../input/oof_5fold/fold3/exp055_resnetrs50_split3d3x9csn_l6_mixup_ep30/oof_fbeta1, score: 0.7788547787792092
Save oof:../../input/oof_5fold/fold3/exp055_resnetrs50_split3d3x9csn_l6_mixup_ep30/oof_fbeta2, score: 0.7810425118646442

Save oof:../../input/oof_5fold/fold4/exp055_resnetrs50_split3d3x9csn_l6_mixup_ep30/oof_fbeta0, score: 0.7287241432138931
Save oof:../../input/oof_5fold/fold4/exp055_resnetrs50_split3d3x9csn_l6_mixup_ep30/oof_fbeta1, score: 0.738563877731826
Save oof:../../input/oof_5fold/fold4/exp055_resnetrs50_split3d3x9csn_l6_mixup_ep30/oof_fbeta2, score: 0.7285363904134047

Save oof:../../input/oof_5fold/fold0/exp055_convnext_tiny_split3d5x7csn_mixup_ep30/oof_fbeta0, score: 0.6480399318093204
Save oof:../../input/oof_5fold/fold0/exp055_convnext_tiny_split3d5x7csn_mixup_ep30/oof_fbeta1, score: 0.6467729138895146
Save oof:../../input/oof_5fold/fold0/exp055_convnext_tiny_split3d5x7csn_mixup_ep30/oof_fbeta2, score: 0.641928565912841

Save oof:../../input/oof_5fold/fold1/exp055_convnext_tiny_split3d5x7csn_mixup_ep30/oof_fbeta0, score: 0.7349395208454771
Save oof:../../input/oof_5fold/fold1/exp055_convnext_tiny_split3d5x7csn_mixup_ep30/oof_fbeta1, score: 0.7244892593795521
Save oof:../../input/oof_5fold/fold1/exp055_convnext_tiny_split3d5x7csn_mixup_ep30/oof_fbeta2, score: 0.7231017482529712

Save oof:../../input/oof_5fold/fold2/exp055_convnext_tiny_split3d5x7csn_mixup_ep30/oof_fbeta0, score: 0.6900721993798925
Save oof:../../input/oof_5fold/fold2/exp055_convnext_tiny_split3d5x7csn_mixup_ep30/oof_fbeta1, score: 0.688034617062955
Save oof:../../input/oof_5fold/fold2/exp055_convnext_tiny_split3d5x7csn_mixup_ep30/oof_fbeta2, score: 0.6964023895503733

Save oof:../../input/oof_5fold/fold3/exp055_convnext_tiny_split3d5x7csn_mixup_ep30/oof_fbeta0, score: 0.7887197902193216
Save oof:../../input/oof_5fold/fold3/exp055_convnext_tiny_split3d5x7csn_mixup_ep30/oof_fbeta1, score: 0.7827502560766313
Save oof:../../input/oof_5fold/fold3/exp055_convnext_tiny_split3d5x7csn_mixup_ep30/oof_fbeta2, score: 0.7868570365845287

Save oof:../../input/oof_5fold/fold4/exp055_convnext_tiny_split3d5x7csn_mixup_ep30/oof_fbeta0, score: 0.7579462384855769
Save oof:../../input/oof_5fold/fold4/exp055_convnext_tiny_split3d5x7csn_mixup_ep30/oof_fbeta1, score: 0.7611716220933374
Save oof:../../input/oof_5fold/fold4/exp055_convnext_tiny_split3d5x7csn_mixup_ep30/oof_fbeta2, score: 0.7580720982921997

Save oof:../../input/oof_5fold/fold0/exp055_convnext_tiny_split3d3x9csn_l6_mixup_ep30/oof_fbeta0, score: 0.6609206181251918
Save oof:../../input/oof_5fold/fold0/exp055_convnext_tiny_split3d3x9csn_l6_mixup_ep30/oof_fbeta1, score: 0.6555688620159943
Save oof:../../input/oof_5fold/fold0/exp055_convnext_tiny_split3d3x9csn_l6_mixup_ep30/oof_fbeta2, score: 0.6530589709144098

Save oof:../../input/oof_5fold/fold1/exp055_convnext_tiny_split3d3x9csn_l6_mixup_ep30/oof_fbeta0, score: 0.7103011197961271
Save oof:../../input/oof_5fold/fold1/exp055_convnext_tiny_split3d3x9csn_l6_mixup_ep30/oof_fbeta1, score: 0.7121981629135082
Save oof:../../input/oof_5fold/fold1/exp055_convnext_tiny_split3d3x9csn_l6_mixup_ep30/oof_fbeta2, score: 0.7207734881575013

Save oof:../../input/oof_5fold/fold2/exp055_convnext_tiny_split3d3x9csn_l6_mixup_ep30/oof_fbeta0, score: 0.7002915696210247
Save oof:../../input/oof_5fold/fold2/exp055_convnext_tiny_split3d3x9csn_l6_mixup_ep30/oof_fbeta1, score: 0.6950416463217677
Save oof:../../input/oof_5fold/fold2/exp055_convnext_tiny_split3d3x9csn_l6_mixup_ep30/oof_fbeta2, score: 0.6887683348534386

Save oof:../../input/oof_5fold/fold3/exp055_convnext_tiny_split3d3x9csn_l6_mixup_ep30/oof_fbeta0, score: 0.7907009230776294
Save oof:../../input/oof_5fold/fold3/exp055_convnext_tiny_split3d3x9csn_l6_mixup_ep30/oof_fbeta1, score: 0.7875351383425719
Save oof:../../input/oof_5fold/fold3/exp055_convnext_tiny_split3d3x9csn_l6_mixup_ep30/oof_fbeta2, score: 0.7901245525792964

Save oof:../../input/oof_5fold/fold4/exp055_convnext_tiny_split3d3x9csn_l6_mixup_ep30/oof_fbeta0, score: 0.754126946565578
Save oof:../../input/oof_5fold/fold4/exp055_convnext_tiny_split3d3x9csn_l6_mixup_ep30/oof_fbeta1, score: 0.7540022807989776
Save oof:../../input/oof_5fold/fold4/exp055_convnext_tiny_split3d3x9csn_l6_mixup_ep30/oof_fbeta2, score: 0.760584247524533

Save oof:../../input/oof_5fold/fold0/exp055_swinv2_tiny_window8_256_split3d5x7csn_mixup_ep30/oof_fbeta0, score: 0.659608109675679
Save oof:../../input/oof_5fold/fold0/exp055_swinv2_tiny_window8_256_split3d5x7csn_mixup_ep30/oof_fbeta1, score: 0.6657015128303987
Save oof:../../input/oof_5fold/fold0/exp055_swinv2_tiny_window8_256_split3d5x7csn_mixup_ep30/oof_fbeta2, score: 0.6591254166859528

Save oof:../../input/oof_5fold/fold1/exp055_swinv2_tiny_window8_256_split3d5x7csn_mixup_ep30/oof_fbeta0, score: 0.7202747225205692
Save oof:../../input/oof_5fold/fold1/exp055_swinv2_tiny_window8_256_split3d5x7csn_mixup_ep30/oof_fbeta1, score: 0.7176350946190864
Save oof:../../input/oof_5fold/fold1/exp055_swinv2_tiny_window8_256_split3d5x7csn_mixup_ep30/oof_fbeta2, score: 0.7164195843410218

Save oof:../../input/oof_5fold/fold2/exp055_swinv2_tiny_window8_256_split3d5x7csn_mixup_ep30/oof_fbeta0, score: 0.7005684116948904
Save oof:../../input/oof_5fold/fold2/exp055_swinv2_tiny_window8_256_split3d5x7csn_mixup_ep30/oof_fbeta1, score: 0.7008682787220188
Save oof:../../input/oof_5fold/fold2/exp055_swinv2_tiny_window8_256_split3d5x7csn_mixup_ep30/oof_fbeta2, score: 0.7022673647780422

Save oof:../../input/oof_5fold/fold3/exp055_swinv2_tiny_window8_256_split3d5x7csn_mixup_ep30/oof_fbeta0, score: 0.7830606810622647
Save oof:../../input/oof_5fold/fold3/exp055_swinv2_tiny_window8_256_split3d5x7csn_mixup_ep30/oof_fbeta1, score: 0.7848431357106206
Save oof:../../input/oof_5fold/fold3/exp055_swinv2_tiny_window8_256_split3d5x7csn_mixup_ep30/oof_fbeta2, score: 0.7828201221633521

Save oof:../../input/oof_5fold/fold4/exp055_swinv2_tiny_window8_256_split3d5x7csn_mixup_ep30/oof_fbeta0, score: 0.7512288835151714
Save oof:../../input/oof_5fold/fold4/exp055_swinv2_tiny_window8_256_split3d5x7csn_mixup_ep30/oof_fbeta1, score: 0.7453624146775143
Save oof:../../input/oof_5fold/fold4/exp055_swinv2_tiny_window8_256_split3d5x7csn_mixup_ep30/oof_fbeta2, score: 0.7524585355508913

Save oof:../../input/oof_5fold/fold0/exp055_swinv2_tiny_window8_256_split3d3x9csn_l6_mixup_ep30/oof_fbeta0, score: 0.6493931321137051
Save oof:../../input/oof_5fold/fold0/exp055_swinv2_tiny_window8_256_split3d3x9csn_l6_mixup_ep30/oof_fbeta1, score: 0.6599491128836409¥
Save oof:../../input/oof_5fold/fold0/exp055_swinv2_tiny_window8_256_split3d3x9csn_l6_mixup_ep30/oof_fbeta2, score: 0.657873640481697

Save oof:../../input/oof_5fold/fold1/exp055_swinv2_tiny_window8_256_split3d3x9csn_l6_mixup_ep30/oof_fbeta0, score: 0.7385469428581766
Save oof:../../input/oof_5fold/fold1/exp055_swinv2_tiny_window8_256_split3d3x9csn_l6_mixup_ep30/oof_fbeta1, score: 0.7327911882998184
Save oof:../../input/oof_5fold/fold1/exp055_swinv2_tiny_window8_256_split3d3x9csn_l6_mixup_ep30/oof_fbeta2, score: 0.7309770404785878

Save oof:../../input/oof_5fold/fold2/exp055_swinv2_tiny_window8_256_split3d3x9csn_l6_mixup_ep30/oof_fbeta0, score: 0.7156795923483393
Save oof:../../input/oof_5fold/fold2/exp055_swinv2_tiny_window8_256_split3d3x9csn_l6_mixup_ep30/oof_fbeta1, score: 0.7016038804620817
Save oof:../../input/oof_5fold/fold2/exp055_swinv2_tiny_window8_256_split3d3x9csn_l6_mixup_ep30/oof_fbeta2, score: 0.6988533879156302

Save oof:../../input/oof_5fold/fold3/exp055_swinv2_tiny_window8_256_split3d3x9csn_l6_mixup_ep30/oof_fbeta0, score: 0.7809338279543435
Save oof:../../input/oof_5fold/fold3/exp055_swinv2_tiny_window8_256_split3d3x9csn_l6_mixup_ep30/oof_fbeta1, score: 0.7825121798341345
Save oof:../../input/oof_5fold/fold3/exp055_swinv2_tiny_window8_256_split3d3x9csn_l6_mixup_ep30/oof_fbeta2, score: 0.7850722284846162

Save oof:../../input/oof_5fold/fold4/exp055_swinv2_tiny_window8_256_split3d3x9csn_l6_mixup_ep30/oof_fbeta0, score: 0.7565529723487409
Save oof:../../input/oof_5fold/fold4/exp055_swinv2_tiny_window8_256_split3d3x9csn_l6_mixup_ep30/oof_fbeta1, score: 0.7599414071343509
Save oof:../../input/oof_5fold/fold4/exp055_swinv2_tiny_window8_256_split3d3x9csn_l6_mixup_ep30/oof_fbeta2, score: 0.7595005007595521

Save oof:../../input/oof_5fold/fold0/exp055_swin_small_patch4_window7_224_split3d5x7csn_mixup_ep30/oof_fbeta0, score: 0.6437059820900941
Save oof:../../input/oof_5fold/fold0/exp055_swin_small_patch4_window7_224_split3d5x7csn_mixup_ep30/oof_fbeta1, score: 0.6498706803774336
Save oof:../../input/oof_5fold/fold0/exp055_swin_small_patch4_window7_224_split3d5x7csn_mixup_ep30/oof_fbeta2, score: 0.651208834563903

Save oof:../../input/oof_5fold/fold1/exp055_swin_small_patch4_window7_224_split3d5x7csn_mixup_ep30/oof_fbeta0, score: 0.71618750126515
Save oof:../../input/oof_5fold/fold1/exp055_swin_small_patch4_window7_224_split3d5x7csn_mixup_ep30/oof_fbeta1, score: 0.731308955984385
Save oof:../../input/oof_5fold/fold1/exp055_swin_small_patch4_window7_224_split3d5x7csn_mixup_ep30/oof_fbeta2, score: 0.7244164898058874

Save oof:../../input/oof_5fold/fold2/exp055_swin_small_patch4_window7_224_split3d5x7csn_mixup_ep30/oof_fbeta0, score: 0.7027046768758832
Save oof:../../input/oof_5fold/fold2/exp055_swin_small_patch4_window7_224_split3d5x7csn_mixup_ep30/oof_fbeta1, score: 0.697866676969987
Save oof:../../input/oof_5fold/fold2/exp055_swin_small_patch4_window7_224_split3d5x7csn_mixup_ep30/oof_fbeta2, score: 0.6900473017733342

Save oof:../../input/oof_5fold/fold3/exp055_swin_small_patch4_window7_224_split3d5x7csn_mixup_ep30/oof_fbeta0, score: 0.7875062126996014
Save oof:../../input/oof_5fold/fold3/exp055_swin_small_patch4_window7_224_split3d5x7csn_mixup_ep30/oof_fbeta1, score: 0.7812735152880702
Save oof:../../input/oof_5fold/fold3/exp055_swin_small_patch4_window7_224_split3d5x7csn_mixup_ep30/oof_fbeta2, score: 0.774539994809877

Save oof:../../input/oof_5fold/fold4/exp055_swin_small_patch4_window7_224_split3d5x7csn_mixup_ep30/oof_fbeta0, score: 0.7503830381292295
Save oof:../../input/oof_5fold/fold4/exp055_swin_small_patch4_window7_224_split3d5x7csn_mixup_ep30/oof_fbeta1, score: 0.7500501473632571
Save oof:../../input/oof_5fold/fold4/exp055_swin_small_patch4_window7_224_split3d5x7csn_mixup_ep30/oof_fbeta2, score: 0.7456378765100435

Save oof:../../input/oof_5fold/fold0/exp055_swin_small_patch4_window7_224_split3d3x9csn_l6_mixup_ep30/oof_fbeta0, score: 0.6496681029059532
Save oof:../../input/oof_5fold/fold0/exp055_swin_small_patch4_window7_224_split3d3x9csn_l6_mixup_ep30/oof_fbeta1, score: 0.6302080170168016
Save oof:../../input/oof_5fold/fold0/exp055_swin_small_patch4_window7_224_split3d3x9csn_l6_mixup_ep30/oof_fbeta2, score: 0.6335892449264857

Save oof:../../input/oof_5fold/fold1/exp055_swin_small_patch4_window7_224_split3d3x9csn_l6_mixup_ep30/oof_fbeta0, score: 0.7291322700543125
Save oof:../../input/oof_5fold/fold1/exp055_swin_small_patch4_window7_224_split3d3x9csn_l6_mixup_ep30/oof_fbeta1, score: 0.732743033665954
Save oof:../../input/oof_5fold/fold1/exp055_swin_small_patch4_window7_224_split3d3x9csn_l6_mixup_ep30/oof_fbeta2, score: 0.7354895303237721

Save oof:../../input/oof_5fold/fold2/exp055_swin_small_patch4_window7_224_split3d3x9csn_l6_mixup_ep30/oof_fbeta0, score: 0.712814220702011
Save oof:../../input/oof_5fold/fold2/exp055_swin_small_patch4_window7_224_split3d3x9csn_l6_mixup_ep30/oof_fbeta1, score: 0.6929803329217874
Save oof:../../input/oof_5fold/fold2/exp055_swin_small_patch4_window7_224_split3d3x9csn_l6_mixup_ep30/oof_fbeta2, score: 0.6980325302555086

Save oof:../../input/oof_5fold/fold3/exp055_swin_small_patch4_window7_224_split3d3x9csn_l6_mixup_ep30/oof_fbeta0, score: 0.7789634391449469
Save oof:../../input/oof_5fold/fold3/exp055_swin_small_patch4_window7_224_split3d3x9csn_l6_mixup_ep30/oof_fbeta1, score: 0.7822920842287041
Save oof:../../input/oof_5fold/fold3/exp055_swin_small_patch4_window7_224_split3d3x9csn_l6_mixup_ep30/oof_fbeta2, score: 0.7806097677543924

Save oof:../../input/oof_5fold/fold4/exp055_swin_small_patch4_window7_224_split3d3x9csn_l6_mixup_ep30/oof_fbeta0, score: 0.7445987861438901
Save oof:../../input/oof_5fold/fold4/exp055_swin_small_patch4_window7_224_split3d3x9csn_l6_mixup_ep30/oof_fbeta1, score: 0.7480714053173845
Save oof:../../input/oof_5fold/fold4/exp055_swin_small_patch4_window7_224_split3d3x9csn_l6_mixup_ep30/oof_fbeta2, score: 0.7560100315346305

Save oof:../../input/oof_5fold/fold0/exp055_ecaresnet26t_split3d3x12csn_l6_mixup_ep30/oof_fbeta0, score: 0.6603261574289282
Save oof:../../input/oof_5fold/fold0/exp055_ecaresnet26t_split3d3x12csn_l6_mixup_ep30/oof_fbeta1, score: 0.6539367620004454
Save oof:../../input/oof_5fold/fold0/exp055_ecaresnet26t_split3d3x12csn_l6_mixup_ep30/oof_fbeta2, score: 0.6566640329530062

Save oof:../../input/oof_5fold/fold1/exp055_ecaresnet26t_split3d3x12csn_l6_mixup_ep30/oof_fbeta0, score: 0.7171293914568536
Save oof:../../input/oof_5fold/fold1/exp055_ecaresnet26t_split3d3x12csn_l6_mixup_ep30/oof_fbeta1, score: 0.716518106091788
Save oof:../../input/oof_5fold/fold1/exp055_ecaresnet26t_split3d3x12csn_l6_mixup_ep30/oof_fbeta2, score: 0.714373356678842

Save oof:../../input/oof_5fold/fold2/exp055_ecaresnet26t_split3d3x12csn_l6_mixup_ep30/oof_fbeta0, score: 0.707478648308418
Save oof:../../input/oof_5fold/fold2/exp055_ecaresnet26t_split3d3x12csn_l6_mixup_ep30/oof_fbeta1, score: 0.6999674578872603
Save oof:../../input/oof_5fold/fold2/exp055_ecaresnet26t_split3d3x12csn_l6_mixup_ep30/oof_fbeta2, score: 0.69518782634459

Save oof:../../input/oof_5fold/fold3/exp055_ecaresnet26t_split3d3x12csn_l6_mixup_ep30/oof_fbeta0, score: 0.7661473342651401
Save oof:../../input/oof_5fold/fold3/exp055_ecaresnet26t_split3d3x12csn_l6_mixup_ep30/oof_fbeta1, score: 0.7660930747065557
Save oof:../../input/oof_5fold/fold3/exp055_ecaresnet26t_split3d3x12csn_l6_mixup_ep30/oof_fbeta2, score: 0.7693648857493685

Save oof:../../input/oof_5fold/fold4/exp055_ecaresnet26t_split3d3x12csn_l6_mixup_ep30/oof_fbeta0, score: 0.742328060126813
Save oof:../../input/oof_5fold/fold4/exp055_ecaresnet26t_split3d3x12csn_l6_mixup_ep30/oof_fbeta1, score: 0.7348173750400517
Save oof:../../input/oof_5fold/fold4/exp055_ecaresnet26t_split3d3x12csn_l6_mixup_ep30/oof_fbeta2, score: 0.7388042521111653

Save oof:../../input/oof_5fold/fold0/exp055_ecaresnet26t_split3d2x15csn_l6_mixup_ep30/oof_fbeta0, score: 0.6380428298638421
Save oof:../../input/oof_5fold/fold0/exp055_ecaresnet26t_split3d2x15csn_l6_mixup_ep30/oof_fbeta1, score: 0.6357406100249648
Save oof:../../input/oof_5fold/fold0/exp055_ecaresnet26t_split3d2x15csn_l6_mixup_ep30/oof_fbeta2, score: 0.6400376482771944

Save oof:../../input/oof_5fold/fold1/exp055_ecaresnet26t_split3d2x15csn_l6_mixup_ep30/oof_fbeta0, score: 0.6968389137485008
Save oof:../../input/oof_5fold/fold1/exp055_ecaresnet26t_split3d2x15csn_l6_mixup_ep30/oof_fbeta1, score: 0.6883259548289397
Save oof:../../input/oof_5fold/fold1/exp055_ecaresnet26t_split3d2x15csn_l6_mixup_ep30/oof_fbeta2, score: 0.696659127327409

Save oof:../../input/oof_5fold/fold2/exp055_ecaresnet26t_split3d2x15csn_l6_mixup_ep30/oof_fbeta0, score: 0.6990990836925484
Save oof:../../input/oof_5fold/fold2/exp055_ecaresnet26t_split3d2x15csn_l6_mixup_ep30/oof_fbeta1, score: 0.7000885112000012
Save oof:../../input/oof_5fold/fold2/exp055_ecaresnet26t_split3d2x15csn_l6_mixup_ep30/oof_fbeta2, score: 0.7005513938680471

Save oof:../../input/oof_5fold/fold3/exp055_ecaresnet26t_split3d2x15csn_l6_mixup_ep30/oof_fbeta0, score: 0.7599457538682646
Save oof:../../input/oof_5fold/fold3/exp055_ecaresnet26t_split3d2x15csn_l6_mixup_ep30/oof_fbeta1, score: 0.7584431303181104
Save oof:../../input/oof_5fold/fold3/exp055_ecaresnet26t_split3d2x15csn_l6_mixup_ep30/oof_fbeta2, score: 0.7652283573815584

Save oof:../../input/oof_5fold/fold4/exp055_ecaresnet26t_split3d2x15csn_l6_mixup_ep30/oof_fbeta0, score: 0.7311242069813252
Save oof:../../input/oof_5fold/fold4/exp055_ecaresnet26t_split3d2x15csn_l6_mixup_ep30/oof_fbeta1, score: 0.7236011867998322
Save oof:../../input/oof_5fold/fold4/exp055_ecaresnet26t_split3d2x15csn_l6_mixup_ep30/oof_fbeta2, score: 0.7282993631844872
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
            image_size=256,
            num_workers=args.num_workers,
            batch_size=args.batch_size,
            preprocess_in_model=True,
        ).test_dataloader()

        logdir = f"../../logs/exp{EXP_ID}/{args.logdir}/fold{i}"
        outdir = f"../../input/oof_5fold/fold{i}/exp{EXP_ID}_{args.logdir}"
        ckpt_names = ["best_fbeta.ckpt", "best_fbeta-v1.ckpt", "best_fbeta-v2.ckpt"]
        for ci, ckpt_name in enumerate(ckpt_names):
            ckpt_path = glob(
                f"{logdir}/**/{ckpt_name}",
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
            model = model.half().to(device=device)
            y_valid = np.array(
                Image.open(
                    f"../../input/vesuvius-challenge-ink-detection-5fold/train/{valid_idx}/inklabels.png"
                ).convert("1")
            )
            p_valid = np.zeros_like(y_valid, dtype=np.float16)
            count_pix = np.zeros_like(y_valid, dtype=np.uint8)
            tta_set = [
                "vanilla",
                "flip_v",
                "flip_h",
                "flip_vh",
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
                pad = (256 - args.image_size) // 2
                if pad > 0:
                    volume_new = volume[:, :, pad:-pad, pad:-pad].to(device)
                else:
                    volume_new = volume.to(device)
                with torch.no_grad():
                    pred_batch = torch.sigmoid(
                        model.model_ema.module(volume_new.half())
                    )

                for i in range(len(pred_batch)):
                    if i % len(tta_set) == 1:
                        pred_batch[i] = pred_batch[i].flip(1)
                    elif i % len(tta_set) == 2:
                        pred_batch[i] = pred_batch[i].flip(2)
                    elif i % len(tta_set) == 3:
                        pred_batch[i] = pred_batch[i].flip(1).flip(2)

                pred_batch = (
                    F.interpolate(
                        pred_batch.detach().to(torch.float32).cpu(),
                        scale_factor=32,
                        mode="bilinear",
                        align_corners=True,
                    )
                    .to(torch.float16)
                    .numpy()
                )
                pred_batch_new = np.zeros(
                    list(pred_batch.shape[:2]) + list(volume.shape[-2:])
                )  # [bs, 1] + [w, h]
                if pad > 0:
                    pred_batch_new[:, :, pad:-pad, pad:-pad] = pred_batch
                else:
                    pred_batch_new = pred_batch
                for xi, yi, pi in zip(
                    x,
                    y,
                    pred_batch_new,
                ):
                    y_lim, x_lim = y_valid[
                        yi * 32 : yi * 32 + volume.shape[-2],
                        xi * 32 : xi * 32 + volume.shape[-1],
                    ].shape
                    p_valid[
                        yi * 32 : yi * 32 + volume.shape[-2],
                        xi * 32 : xi * 32 + volume.shape[-1],
                    ] += pi[0, :y_lim, :x_lim]
                    count_pix_single = np.zeros_like(pi[0], dtype=np.uint8)
                    if pad > 0:
                        count_pix_single[pad:-pad, pad:-pad] = np.ones_like(
                            pred_batch[0][0], dtype=np.uint8
                        )
                    else:
                        count_pix_single = np.ones_like(
                            pred_batch[0][0], dtype=np.uint8
                        )
                    count_pix[
                        yi * 32 : yi * 32 + volume.shape[-2],
                        xi * 32 : xi * 32 + volume.shape[-1],
                    ] += count_pix_single[:y_lim, :x_lim]
            fragment_mask = (
                np.array(
                    Image.open(
                        f"../../input/vesuvius-challenge-ink-detection-5fold/train/{valid_idx}/mask.png"
                    ).convert("1")
                )
                > 0
            )
            count_pix *= fragment_mask
            p_valid /= count_pix
            p_valid = np.nan_to_num(p_valid, posinf=0, neginf=0)
            count_pix = count_pix > 0
            p_valid *= fragment_mask
            p_valid_tmp = p_valid.reshape(-1)[np.where(count_pix.reshape(-1))]
            y_valid_tmp = y_valid.reshape(-1)[np.where(count_pix.reshape(-1))]
            threshold = find_threshold_percentile(y_valid_tmp, p_valid_tmp)
            p_valid = p_valid > np.quantile(p_valid_tmp, threshold)
            score = fbeta_score(y_valid, p_valid, beta=0.5)
            os.makedirs(f"{outdir}", exist_ok=True)
            oof_filename = f"{outdir}/oof_fbeta{ci}"
            np.save(oof_filename, p_valid)
            print(f"Save oof:{oof_filename}, score: {score}")


if __name__ == "__main__":
    main(get_args())
