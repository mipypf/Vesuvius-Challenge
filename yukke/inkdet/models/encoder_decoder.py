import warnings
from typing import List, Optional

import torch
import torch.nn as nn
from loguru import logger

from inkdet.models import resnet3d

from .decoder import DecoderCNN2d, DecoderCNN3d, DecoderMLP, DecoderSimpleMLP, UpSampleType
from .mvit import mvit_v1_b, mvit_v2_s
from .utils import BatchNorm_Momentum, conv1x1


class EncDec(nn.Module):
    def __init__(
        self,
        encoder_name: str,
        classes: int,
        decoder_type: str = "cnn_2d",
        freeze_encoder: bool = False,
        normalize_image: bool = False,
        # encoder: resnet3d
        encoder_depth: int = 18,
        encoder_no_max_pool: bool = False,
        checkpoint_path: Optional[str] = None,
        # encoder: swin3d
        encoder_dropout: float = 0.0,
        # encoder: mvit
        encoder_num_classes: Optional[int] = None,
        # encoder_replace_max_pool3d: bool = False,
        # decoder: cnn_2d
        dim_compress_mode: str = "mean",
        use_lateral_attention: bool = False,
        decoder_upscale_mode: str = "",  # and for cnn_3d
        decoder_upsample_mode: str = "nearest",  # and for cnn_3d
        decoder_hypercolumn_indexes: List[int] = [0],
        decoder_logit_after_upsample: bool = False,
        # decoder: cnn_3d
        decoder_skip_downscale_factor: Optional[int] = None,
        decoder_skip_d_dim: Optional[int] = None,
        # decoder: mlp
        decoder_patch_expand_channel_scale: int = 2,
        decoder_logit_interpolate_upscale_factor: int = 2,
        decoder_downsample_factor: int = 1,
        encoder_use_pre_trained_model: bool = True,
        decoder_num_layers: int = 1,
        # decoder: simple_mlp
        decoder_embedding_channels: int = 32,
    ):
        super().__init__()
        assert classes > 0
        assert decoder_type in ["cnn_2d", "cnn_3d", "mlp", "simple_mlp"]

        if decoder_upscale_mode:
            warnings.warn("decoder_upscale_mode is deprecated. Please use decoder_upsample_type")
            decoder_upsample_mode = decoder_upsample_mode

        self.normalize_image = normalize_image
        self.decoder_type = decoder_type
        self.dim_compress_mode = dim_compress_mode
        self.use_lateral_attention = use_lateral_attention
        scale_factor: int = 2 if dim_compress_mode == "max-mean" else 1

        if encoder_name == "swin3d_t":
            from .swin_transformer import swin3d_t

            self.encoder = swin3d_t(
                in_channels=1,
                channel_first=True if "cnn" in decoder_type else False,
                dropout=encoder_dropout,
            )
            if encoder_use_pre_trained_model:
                self.encoder.load_checkpoint("swin3d_t")
        elif encoder_name == "swin3d_s":
            from .swin_transformer import swin3d_s

            self.encoder = swin3d_s(
                in_channels=1,
                channel_first=True if "cnn" in decoder_type else False,
                dropout=encoder_dropout,
            )
            if encoder_use_pre_trained_model:
                self.encoder.load_checkpoint("swin3d_s")
        elif encoder_name == "mvit_v1_b":
            self.encoder = mvit_v1_b(
                in_channels=1,
                channel_first=True if "cnn" in decoder_type else False,
                num_classes=encoder_num_classes,
            )
            if encoder_use_pre_trained_model:
                self.encoder.load_checkpoint("mvit_v1_b")
        elif encoder_name == "mvit_v2_s":
            self.encoder = mvit_v2_s(
                in_channels=1,
                channel_first=True if "cnn" in decoder_type else False,
                num_classes=encoder_num_classes,
            )
            if encoder_use_pre_trained_model:
                self.encoder.load_checkpoint("mvit_v2_s")
        elif encoder_name == "r3d_18":
            from .resnet import r3d_18

            self.encoder = r3d_18(
                in_channels=1,
            )
            if encoder_use_pre_trained_model:
                self.encoder.load_checkpoint("r3d_18")
        elif encoder_name == "mc3_18":
            from .resnet import mc3_18

            self.encoder = mc3_18(
                in_channels=1,
            )
            if encoder_use_pre_trained_model:
                self.encoder.load_checkpoint("mc3_18")
        elif encoder_name == "r2plus1d_18":
            from .resnet import r2plus1d_18

            self.encoder = r2plus1d_18(
                in_channels=1,
            )
            if encoder_use_pre_trained_model:
                self.encoder.load_checkpoint("r2plus1d_18")
        else:
            self.encoder = resnet3d.generate_model(
                model_name=encoder_name,
                model_depth=encoder_depth,
                no_max_pool=encoder_no_max_pool,
                n_input_channels=1,
                n_classes=encoder_num_classes,
            )

            def load_checkpoint(checkpoint_path: str):
                if encoder_name == "resnet":
                    the_1st_layer_key = "conv1.weight"
                elif encoder_name == "resnet2p1d":
                    the_1st_layer_key = "conv1_s.weight"
                else:
                    raise ValueError(f"Not Available for {encoder_name}")

                # Convert 3 channel weights to single channel
                # ref: https://timm.fast.ai/models#Case-1:-When-the-number-of-input-channels-is-1
                state_dict = torch.load(checkpoint_path)["state_dict"]
                conv1_weight = state_dict[the_1st_layer_key]
                state_dict[the_1st_layer_key] = conv1_weight.sum(dim=1, keepdim=True)
                if encoder_num_classes is not None:
                    state_dict["fc.weight"] = state_dict["fc.weight"][:encoder_num_classes]
                    state_dict["fc.bias"] = state_dict["fc.bias"][:encoder_num_classes]
                logger.info(f"Load checkpoint {self.encoder.load_state_dict(state_dict, strict=False)}")

            if checkpoint_path:
                load_checkpoint(checkpoint_path)

        if encoder_name in ["resnet", "se_resnet"]:
            if encoder_depth <= 34:
                encoder_dims = [
                    self.encoder.layer1[-1].bn2.num_features * scale_factor,
                    self.encoder.layer2[-1].bn2.num_features * scale_factor,
                    self.encoder.layer3[-1].bn2.num_features * scale_factor,
                    self.encoder.layer4[-1].bn2.num_features * scale_factor,
                ]
            else:
                encoder_dims = [
                    self.encoder.layer1[-1].bn3.num_features * scale_factor,
                    self.encoder.layer2[-1].bn3.num_features * scale_factor,
                    self.encoder.layer3[-1].bn3.num_features * scale_factor,
                    self.encoder.layer4[-1].bn3.num_features * scale_factor,
                ]
        elif encoder_name in ["resnet2p1d"]:
            encoder_dims = [
                self.encoder.layer1[-1].bn2_t.num_features * scale_factor,
                self.encoder.layer2[-1].bn2_t.num_features * scale_factor,
                self.encoder.layer3[-1].bn2_t.num_features * scale_factor,
                self.encoder.layer4[-1].bn2_t.num_features * scale_factor,
            ]
        elif encoder_name in ["r3d_18", "mc3_18", "r2plus1d_18"]:
            encoder_dims = [64, 128, 256, 512]
        elif encoder_name in ["mvit_v1_b"]:
            encoder_dims = [192, 384, 768, 768]
        elif encoder_name in ["swin3d_t", "swin3d_s", "mvit_v2_s"]:
            encoder_dims = [96, 192, 384, 768]
        else:
            raise NotImplementedError(f"encoder_name: {encoder_name}")

        if decoder_type == "cnn_2d":
            assert self.dim_compress_mode in ["max", "mean", "max-mean", "cnn"]

            self.init_lateral_paths(encoder_dims=encoder_dims)

            self.decoder = DecoderCNN2d(
                classes=classes,
                encoder_dims=encoder_dims,
                hypercolumn_indexes=decoder_hypercolumn_indexes,
                upscale_mode=decoder_upsample_mode,
                logit_upscale_factor=2 // decoder_downsample_factor
                if encoder_no_max_pool
                else 4 // decoder_downsample_factor,
                logit_after_upsample=decoder_logit_after_upsample,
                downsample_factor=decoder_downsample_factor,
            )
        elif decoder_type == "cnn_3d":
            if "resnet" in encoder_name:
                logit_upscale_factor = 2 if encoder_no_max_pool else 4
                logit_channel_scale_factor = decoder_skip_d_dim
                # logit_channel_scale_factor = 16 if encoder_no_max_pool else 8
                # logit_channel_scale_factor //= decoder_skip_downscale_factor ** 4
                assert logit_channel_scale_factor > 0
                upsample_type = UpSampleType.HW
            elif "swin3d" in encoder_name:
                logit_upscale_factor = 4
                logit_channel_scale_factor = decoder_skip_d_dim
                # logit_channel_scale_factor = 8
                # logit_channel_scale_factor //= decoder_skip_downscale_factor
                assert logit_channel_scale_factor > 0
                upsample_type = UpSampleType.HW
            else:
                raise ValueError(f"Unknown encoder_name: {encoder_name}")

            self.decoder = DecoderCNN3d(
                classes=classes,
                encoder_dims=encoder_dims,
                upsample_mode=decoder_upsample_mode,
                upsample_type=upsample_type,
                skip_downscale_factor=decoder_skip_downscale_factor,
                skip_d_dim=decoder_skip_d_dim,
                logit_upscale_factor=logit_upscale_factor,
                logit_channel_scale_factor=logit_channel_scale_factor,
            )
        elif decoder_type == "mlp":
            self.decoder = DecoderMLP(
                classes=classes,
                encoder_dims=encoder_dims,
                patch_expand_channel_scale=decoder_patch_expand_channel_scale,
                logit_interpolate_upscale_factor=decoder_logit_interpolate_upscale_factor,
                downsample_factor=decoder_downsample_factor,
                num_layers=decoder_num_layers,
            )
        elif decoder_type == "simple_mlp":
            self.decoder = DecoderSimpleMLP(
                classes=classes,
                encoder_dims=encoder_dims,
                embedding_channels=decoder_embedding_channels,
            )

        if freeze_encoder:
            logger.info("Freeze encoder.")
            for p in self.encoder.parameters():
                p.requires_grad = False

    def init_lateral_paths(self, encoder_dims: List[int]):
        if self.dim_compress_mode == "cnn":
            base_channels: int = 1024
            self.lateral_convs = nn.ModuleList(
                [
                    nn.Sequential(
                        # lazy_conv1x1(encoder_dims[i]), # cannot use with compile
                        conv1x1(base_channels, encoder_dims[i]),
                        nn.BatchNorm2d(encoder_dims[i], momentum=BatchNorm_Momentum),
                        nn.ReLU(inplace=True),
                    )
                    for i in range(4)
                ]
            )
            if self.use_lateral_attention:
                base_channels: int = 1024
                squeeze_factor: int = 16

                self.attentions = nn.ModuleList(
                    [
                        nn.Sequential(
                            nn.AdaptiveAvgPool2d(1),
                            conv1x1(encoder_dims[i], encoder_dims[i]),
                            nn.BatchNorm2d(encoder_dims[i], momentum=BatchNorm_Momentum),
                            # conv1x1(encoder_dims[i], encoder_dims[i] // squeeze_factor),
                            # nn.ReLU(inplace=True),
                            # conv1x1(encoder_dims[i] // squeeze_factor, encoder_dims[i]),
                            nn.Sigmoid(),
                        )
                        for i in range(4)
                    ]
                )

        else:
            if self.use_lateral_attention:
                base_channels: int = 1024
                squeeze_factor: int = 16

                self.attentions = nn.ModuleList(
                    [
                        nn.Sequential(
                            nn.AdaptiveAvgPool2d(1),
                            conv1x1(base_channels, base_channels),
                            nn.BatchNorm2d(base_channels, momentum=BatchNorm_Momentum),
                            # conv1x1(base_channels, base_channels // squeeze_factor),
                            # nn.ReLU(inplace=True),
                            # conv1x1(base_channels // squeeze_factor, base_channels),
                            nn.Sigmoid(),
                        )
                        for _ in range(4)
                    ]
                )

    def compress_dim(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        def geometric_dim_compress():
            for i, feature in enumerate(features):
                if self.use_lateral_attention:
                    b, c, d, h, w = feature.shape
                    feature_3d = feature.view(b, c * d, h, w)
                    attention = self.attentions[i](feature_3d)
                    feature_3d = feature_3d * attention
                    feature = feature_3d.view(b, c, d, h, w)
                    # self.attention_outputs.append(attention.cpu().numpy())

                if self.dim_compress_mode == "max":
                    feature_3d = torch.max(feature, dim=2)[0]
                elif self.dim_compress_mode == "mean":
                    feature_3d = torch.mean(feature, dim=2)
                elif self.dim_compress_mode == "max-mean":
                    if feature.shape[2] > 1:
                        feature_3d = torch.cat([feature.max(dim=2)[0], feature.mean(dim=2)], dim=1)
                    else:
                        feature_3d = feature.squeeze(2)
                else:
                    raise ValueError(f"Not available dim_compress_mode: {self.dim_compress_mode}")

                features[i] = feature_3d

        def cnn_dim_compress():
            for i, feature in enumerate(features):
                b, c, d, h, w = feature.shape
                feature_3d = feature.view(b, c * d, h, w)
                feature_3d = self.lateral_convs[i](feature_3d)

                if self.use_lateral_attention:
                    attention = self.attentions[i](feature_3d)
                    feature_3d = feature_3d * attention

                features[i] = feature_3d

        if self.dim_compress_mode == "cnn":
            cnn_dim_compress()
        else:
            geometric_dim_compress()

        return features

    def forward(self, x: torch.Tensor, feature_only: bool = False) -> torch.Tensor:
        if self.normalize_image:
            mean = x.mean(dim=(1, 2, 3), keepdim=True)
            std = x.std(dim=(1, 2, 3), keepdim=True) + 1e-6
            x = (x - mean) / std

        if x.ndim == 4:
            x = x.unsqueeze(1)

        enc_ret = self.encoder(x)
        has_classification: bool = isinstance(enc_ret, tuple)
        if has_classification:
            features: List[torch.Tensor] = enc_ret[0]
        else:
            features = enc_ret

        if self.decoder_type == "cnn_2d":
            features = self.compress_dim(features)

        pred = self.decoder(features, feature_only)

        if has_classification:
            return pred, enc_ret[1]
        else:
            return pred


class Enc3dDec2d(EncDec):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        warnings.warn("Enc3dDec2d is deprecated. Please use EncDec")
