from . import resnet, resnet2p1d, se_resnet


def generate_model(model_name: str, **kwargs):
    assert model_name in ["resnet", "resnet2p1d", "se_resnet"]

    if model_name == "resnet":
        model = resnet.generate_model(**kwargs)
    elif model_name == "resnet2p1d":
        model = resnet2p1d.generate_model(**kwargs)
    elif model_name == "se_resnet":
        model = se_resnet.generate_model(**kwargs)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    return model
