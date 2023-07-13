import dataclasses
from argparse import Action
from pathlib import Path
from typing import List, Optional

import yaml

BASE_KEY = "_base_"
DELETE_KEY = "_delete_"
DEPRECATION_KEY = "_deprecation_"
RESERVED_KEYS = ["filename", "text", "pretty_text"]


@dataclasses.dataclass(frozen=True)
class Config:
    # === general ===
    competition_name: str = "vesuvius"
    data_root: str = "/kaggle/input/vesuvius-challenge-ink-detection"
    root_work_dir: str = "/opt/kaggle-ink-detection/work_dirs"
    work_dir: str = ""
    num_workers: int = 4
    seed: int = 42
    float32_matmul_precision: Optional[str] = None
    log_interval: int = 50
    task: str = "segmentation"

    # === dataset ===
    start_channels: int = dataclasses.field(default_factory=int)
    in_channels: int = dataclasses.field(default_factory=int)
    patch_size: int = dataclasses.field(default_factory=int)
    target_size: int = dataclasses.field(default_factory=int)
    stride: int = dataclasses.field(default_factory=int)
    val_patch_size: int = dataclasses.field(default_factory=int)
    val_stride: int = dataclasses.field(default_factory=int)
    norm_mean: float = 0.0
    norm_std: float = 1.0
    clip: list = dataclasses.field(default_factory=list)
    gamma: Optional[float] = None
    transforms: List[str] = dataclasses.field(default_factory=list)
    # >> deprecated
    use_mixup: bool = False
    mixup_alpha: float = 0.0
    # << deprecated
    disable_mixup_last_epoch: Optional[int] = None
    mix_augmentation_p: float = 0.0
    mix_augmentation_alpha: float = 0.0
    use_manifold_mixup: float = False

    use_amp: bool = True
    use_mask: bool = False
    use_valid_mask: bool = True
    batch_size: int = 16

    train_augmentations: list = dataclasses.field(default_factory=list)
    use_weighted_random_sampler: bool = False

    # === model ===
    num_classes: int = 1
    model_name: str = dataclasses.field(default_factory=str)
    encoder_name: str = dataclasses.field(default_factory=str)
    model_params: dict = dataclasses.field(default_factory=dict)
    head_init_weights: Optional[str] = None
    head_init_bias: Optional[float] = None
    losses: List[str] = dataclasses.field(default_factory=list)
    optional_losses: List[str] = dataclasses.field(default_factory=list)
    checkpoint_path: str = ""
    pretrained_path: str = ""
    logit_clip_epsilon: Optional[float] = None
    use_pseudo_label: bool = False

    # === optimizer ===
    optimizer_name: str = dataclasses.field(default_factory=str)
    optimizer_params: dict = dataclasses.field(default_factory=dict)
    scheduler_name: str = dataclasses.field(default_factory=str)
    scheduler_params: dict = dataclasses.field(default_factory=dict)
    warmup_factor: int = 10
    max_lr: float = 1e-4
    min_lr: float = 1e-6
    # >> deprecated
    weight_decay: float = 1e-6
    # << deprecated
    max_grad_norm: float = 1e3
    epochs: int = 15
    encoder_lr_scale: float = 0

    def dump(self, out_path: Path):
        with open(out_path, "w") as f:
            yaml.dump(dataclasses.asdict(self), f, indent=2, sort_keys=False)

    @classmethod
    def fromfile(cls, cfg_path: Path, options: Optional[dict] = None):
        cfg_dict = cls._load(cfg_path)
        if options is not None:
            cfg_dict = cls.merge_from_dict(cfg_dict, options, True)
        return cls(**cfg_dict)

    @staticmethod
    def _load(cfg_path: Path) -> dict:
        with open(cfg_path) as f:
            _cfg_dict: dict = yaml.safe_load(f)

        if BASE_KEY in _cfg_dict:
            _base_ = _cfg_dict[BASE_KEY]
            if isinstance(_base_, str):
                cfg_dict = Config._load(cfg_path.parent / _base_)
                cfg_dict = _merge_a_into_b(_cfg_dict, cfg_dict)
            else:
                raise TypeError(f"{BASE_KEY} must be str: {type(_base_)}")

            cfg_dict.pop(BASE_KEY)
        else:
            cfg_dict = _cfg_dict.copy()

        return cfg_dict

    @staticmethod
    def merge_from_dict(cfg_dict: dict, options: dict, allow_list_keys: bool = True):
        """Merge list into cfg_dict.

        Merge the dict parsed by MultipleKVAction into this cfg.

        Examples:
            >>> options = {'model.backbone.depth': 50,
            ...            'model.backbone.with_cp':True}
            >>> cfg = Config(dict(model=dict(backbone=dict(type='ResNet'))))
            >>> cfg.merge_from_dict(options)
            >>> cfg_dict = super(Config, self).__getattribute__('_cfg_dict')
            >>> assert cfg_dict == dict(
            ...     model=dict(backbone=dict(depth=50, with_cp=True)))

            >>> # Merge list element
            >>> cfg = Config(dict(pipeline=[
            ...     dict(type='LoadImage'), dict(type='LoadAnnotations')]))
            >>> options = dict(pipeline={'0': dict(type='SelfLoadImage')})
            >>> cfg.merge_from_dict(options, allow_list_keys=True)
            >>> cfg_dict = super(Config, self).__getattribute__('_cfg_dict')
            >>> assert cfg_dict == dict(pipeline=[
            ...     dict(type='SelfLoadImage'), dict(type='LoadAnnotations')])

        Args:
            options (dict): dict of configs to merge from.
            allow_list_keys (bool): If True, int string keys (e.g. '0', '1')
              are allowed in ``options`` and will replace the element of the
              corresponding index in the config if the config is a list.
              Default: True.
        """
        option_cfg_dict = {}
        for full_key, v in options.items():
            d = option_cfg_dict
            key_list = full_key.split(".")
            for subkey in key_list[:-1]:
                d.setdefault(subkey, dict())
                d = d[subkey]
            subkey = key_list[-1]
            d[subkey] = v

        return _merge_a_into_b(option_cfg_dict, cfg_dict, allow_list_keys=allow_list_keys)


# reference: https://github.com/open-mmlab/mmcv/blob/v1.7.1/mmcv/utils/config.py
def _merge_a_into_b(a: dict, b: dict, allow_list_keys: bool = False):
    """merge dict ``a`` into dict ``b`` (non-inplace).

    Values in ``a`` will overwrite ``b``. ``b`` is copied first to avoid
    in-place modifications.

    Args:
        a (dict): The source dict to be merged into ``b``.
        b (dict): The origin dict to be fetch keys from ``a``.
        allow_list_keys (bool): If True, int string keys (e.g. '0', '1')
            are allowed in source ``a`` and will replace the element of the
            corresponding index in b if b is a list. Default: False.

    Returns:
        dict: The modified dict of ``b`` using ``a``.

    Examples:
        # Normally merge a into b.
        >>> Config._merge_a_into_b(
        ...     dict(obj=dict(a=2)), dict(obj=dict(a=1)))
        {'obj': {'a': 2}}

        # Delete b first and merge a into b.
        >>> Config._merge_a_into_b(
        ...     dict(obj=dict(_delete_=True, a=2)), dict(obj=dict(a=1)))
        {'obj': {'a': 2}}

        # b is a list
        >>> Config._merge_a_into_b(
        ...     {'0': dict(a=2)}, [dict(a=1), dict(b=2)], True)
        [{'a': 2}, {'b': 2}]
    """
    b = b.copy()
    for k, v in a.items():
        if allow_list_keys and k.isdigit() and isinstance(b, list):
            k = int(k)
            if len(b) <= k:
                raise KeyError(f"Index {k} exceeds the length of list {b}")
            b[k] = _merge_a_into_b(v, b[k], allow_list_keys)
        elif isinstance(v, dict):
            if k in b and not v.pop(DELETE_KEY, False):
                allowed_types = (dict, list) if allow_list_keys else dict
                if not isinstance(b[k], allowed_types):
                    raise TypeError(
                        f"{k}={v} in child config cannot inherit from "
                        f"base because {k} is a dict in the child config "
                        f"but is of type {type(b[k])} in base config. "
                        f"You may set `{DELETE_KEY}=True` to ignore the "
                        f"base config."
                    )
                b[k] = _merge_a_into_b(v, b[k], allow_list_keys)
            else:
                b[k] = v
        else:
            b[k] = v
    return b


class DictAction(Action):
    """
    argparse action to split an argument into KEY=VALUE form
    on the first = and append to a dictionary. List options can
    be passed as comma separated values, i.e 'KEY=V1,V2,V3', or with explicit
    brackets, i.e. 'KEY=[V1,V2,V3]'. It also support nested brackets to build
    list/tuple values. e.g. 'KEY=[(V1,V2),(V3,V4)]'
    """

    @staticmethod
    def _parse_int_float_bool(val):
        try:
            return int(val)
        except ValueError:
            pass
        try:
            return float(val)
        except ValueError:
            pass
        if val.lower() in ["true", "false"]:
            return True if val.lower() == "true" else False
        if val == "None":
            return None
        return val

    @staticmethod
    def _parse_iterable(val):
        """Parse iterable values in the string.

        All elements inside '()' or '[]' are treated as iterable values.

        Args:
            val (str): Value string.

        Returns:
            list | tuple: The expanded list or tuple from the string.

        Examples:
            >>> DictAction._parse_iterable('1,2,3')
            [1, 2, 3]
            >>> DictAction._parse_iterable('[a, b, c]')
            ['a', 'b', 'c']
            >>> DictAction._parse_iterable('[(1, 2, 3), [a, b], c]')
            [(1, 2, 3), ['a', 'b'], 'c']
        """

        def find_next_comma(string):
            """Find the position of next comma in the string.

            If no ',' is found in the string, return the string length. All
            chars inside '()' and '[]' are treated as one element and thus ','
            inside these brackets are ignored.
            """
            assert (string.count("(") == string.count(")")) and (
                string.count("[") == string.count("]")
            ), f"Imbalanced brackets exist in {string}"
            end = len(string)
            for idx, char in enumerate(string):
                pre = string[:idx]
                # The string before this ',' is balanced
                if (char == ",") and (pre.count("(") == pre.count(")")) and (pre.count("[") == pre.count("]")):
                    end = idx
                    break
            return end

        # Strip ' and " characters and replace whitespace.
        val = val.strip("'\"").replace(" ", "")
        is_tuple = False
        if val.startswith("(") and val.endswith(")"):
            is_tuple = True
            val = val[1:-1]
        elif val.startswith("[") and val.endswith("]"):
            val = val[1:-1]
        elif "," not in val:
            # val is a single value
            return DictAction._parse_int_float_bool(val)

        values = []
        while len(val) > 0:
            comma_idx = find_next_comma(val)
            element = DictAction._parse_iterable(val[:comma_idx])
            values.append(element)
            val = val[comma_idx + 1 :]
        if is_tuple:
            values = tuple(values)
        return values

    def __call__(self, parser, namespace, values, option_string=None):
        options = {}
        for kv in values:
            key, val = kv.split("=", maxsplit=1)
            options[key] = self._parse_iterable(val)
        setattr(namespace, self.dest, options)
