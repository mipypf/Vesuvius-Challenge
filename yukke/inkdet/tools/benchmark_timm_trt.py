import sys

sys.path.append("pytorch-image-models")

import os
import time

import numpy as np
import tensorrt
import timm
import torch
import torch_tensorrt
import tqdm

# from torch2trt import torch2trt

print("torch", torch.__version__)
print("timm", timm.__version__)
print("tensorrt", tensorrt.__version__)
print("torch_tensorrt", torch_tensorrt.__version__)

TENSOR_SHAPE = (8, 3, 256, 256)
WARMUP_ITER = 100
BENCHMARK_ITER = 1000
TRT_TS_WORK_DIR = "trt_ts_work_dir"

MODEL_NAME = "resnetrs50"
# benchmark_classification: 32 sec
# benchmark_features: 31 sec
MODELS = {
    # fp16
    # "resnetrs50": (8, 3, 256, 256),
    # "resnext50d_32x4d": (8, 3, 256, 256),
    # fp32
    # "convnext_tiny": (8, 3, 256, 256),
    "swinv2_tiny_window8_256.ms_in1k": (8, 3, 256, 256),
    # "swin_small_patch4_window7_224.ms_in22k_ft_in1k": (8, 3, 224, 224),
}


def benchmark_features(model_name, input_shape, dtype, skip_trt):
    assert dtype in [torch.float32, torch.float16]
    assert len(input_shape) == 4
    print("=" * 5, model_name, f"({dtype})", "=" * 5)
    dtype_str = str(dtype).split(".")[1]
    model = timm.create_model(model_name, pretrained=True, features_only=True).eval().to(dtype).cuda()

    if not skip_trt:
        trt_path = os.path.join(TRT_TS_WORK_DIR, f"trt_{dtype_str}_{model_name}_features_only.ts")
        if os.path.exists(trt_path):
            print("loading", trt_path)
            trt_model = torch.jit.load(trt_path)

        else:
            print("compiling ...")
            t = time.time()
            try:
                trt_model = torch_tensorrt.compile(
                    model,
                    inputs=[
                        torch_tensorrt.Input(input_shape, dtype=dtype),
                    ],
                    enabled_precisions={torch.float32, torch.float16},
                    workspace_size=1 << 32,
                    # torch_executed_ops=["aten::layer_norm"]
                    # torch_executed_ops=["aten::permute", "aten::layer_norm", "aten::reshape", "aten::transpose"]
                )
            except Exception as e:
                print(model_name, e)
                return

            torch.jit.save(trt_model, trt_path)
            print(f"save ({time.time() - t} sec)", trt_path)

        print("-" * 5, "check_error", "-" * 5)
        errors = []
        for _ in tqdm.tqdm(range(BENCHMARK_ITER)):
            x = torch.randn(*input_shape).to(dtype).cuda()
            with torch.no_grad():
                y1 = model(x)
                y2 = trt_model(x)

            error = torch.mean(torch.abs(y1 - y2)).detach().cpu().numpy()
            errors.append(error.item())

        errors = np.array(errors)
        print(f"l1 error: mean={np.mean(errors)} std={np.std(errors)}")

    print("-" * 5, "benchmark", "-" * 5)
    x = torch.randn(*input_shape).cuda().to(dtype)
    t = time.time()
    for _ in tqdm.tqdm(range(BENCHMARK_ITER)):
        with torch.no_grad():
            y1 = model(x)
    time_pytorch = time.time() - t

    if not skip_trt:
        t = time.time()
        for _ in tqdm.tqdm(range(BENCHMARK_ITER)):
            with torch.no_grad():
                y2 = trt_model(x)
        time_pytorch_compile = time.time() - t
    else:
        time_pytorch_compile = 0

    print(f"Torch: {time_pytorch:.2f} sec, TensorRT: {time_pytorch_compile:.2f}")

    print("DONE!")


def benchmark_features_compile(model_name, input_shape, dtype, skip_trt=False):
    assert dtype in [torch.float32, torch.float16]
    assert len(input_shape) == 4
    print("=" * 5, model_name, "=" * 5)
    dtype_str = str(dtype).split(".")[1]
    model = timm.create_model(model_name, pretrained=True, features_only=True).eval().to("cuda")

    if not skip_trt:
        trt_path = f"trt_{dtype_str}_{model_name}_features_only.pth"
        if os.path.exists(trt_path):
            raise NotImplementedError
            print("loading", trt_path)
            trt_model = torch.jit.load(trt_path)
        else:
            print("compiling ...")
            t = time.time()
            trt_model = torch.compile(model)
            # try:
            #     trt_model = torch.compile(model)
            # except Exception as e:
            #     print(model_name, e)
            #     return

            # torch.save(trt_model.state_dict(), trt_path)
            print(f"save ({time.time() - t} sec)", trt_path)

    print("-" * 5, "check_error", "-" * 5)
    errors = []
    for _ in tqdm.tqdm(range(NUM_ITER)):
        x = torch.randn(*input_shape).cuda().to(dtype)
        with torch.no_grad():
            y1 = model(x)
            if skip_trt:
                continue
            y2 = trt_model(x)

        error = torch.mean(torch.abs(y1 - y2)).detach().cpu().numpy()
        errors.append(error.item())

    errors = np.array(errors)
    print(f"l1 error: mean={np.mean(errors)} std={np.std(errors)}")

    print("-" * 5, "benchmark", "-" * 5)
    x = torch.randn(*input_shape).cuda().to(dtype)
    t = time.time()
    for _ in tqdm.tqdm(range(NUM_ITER * 10)):
        with torch.no_grad():
            y1 = model(x)
    time_pytorch = time.time() - t

    t = time.time()
    for _ in tqdm.tqdm(range(NUM_ITER * 10)):
        with torch.no_grad():
            y2 = trt_model(x)
    time_pytorch_compile = time.time() - t

    print(f"No compile: {time_pytorch:.2f} sec, Compile: {time_pytorch_compile:.2f}")
    print("DONE!")


# benchmark_classification()
# benchmark_features(torch.float32, skip_trt=True)

for model_name, input_shape in MODELS.items():
    # benchmark_features(model_name, input_shape, torch.float32, skip_trt=False)
    # benchmark_features(model_name, input_shape, torch.float16, skip_trt=not model_name.startswith("res"))
    benchmark_features(model_name, input_shape, torch.float16, skip_trt=False)
