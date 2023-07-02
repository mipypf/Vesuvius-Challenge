# tattaka&ron part solution

## 1. INSTALLATION
Use [Kaggle Docker v128](https://console.cloud.google.com/gcr/images/kaggle-gpu-images/GLOBAL/python). Then install the missing packages according to the requiement.txt. The whl file around mmdetection-offline-lib in the requiement.txt can be obtained from the following URL
<https://www.kaggle.com/datasets/nvnnghia/mmdetection-offline-lib>
## 2. DATA

Place competition data as follows:
<pre>
input
└── vesuvius-challenge-ink-detection
</pre>
## 3. PREPROCESSING
Please do the following:
```
cd Vesuvius-Challenge/tattaka_ron/input
python split_fragment2.py
```
Next:
```
cd Vesuvius-Challenge/tattaka_ron/input
python make_patch_32_5fold.py
```
The directory structure will be as follows:
<pre>
input
└── vesuvius-challenge-ink-detection
└── vesuvius-challenge-ink-detection-5fold
    └──train
       └──2
       └──4
       └──5
</pre>
Place directories 1 and 3 of vesuvius-challenge-ink-detection/train as follows:
<pre>
input
└── vesuvius-challenge-ink-detection
└── vesuvius-challenge-ink-detection-5fold
    └──train
       └──1
       └──2
       └──3
       └──4
       └──5
</pre>
## 4. TRAINING
Please do the following:
```
cd Vesuvius-Challenge/tattaka_ron/src/exp055
sh train.sh
```
Next:
```
cd Vesuvius-Challenge/tattaka_ron/src/exp056
sh train.sh
```
## 5. INFERENCE
Actual inference notebook:
https://www.kaggle.com/code/mipypf/ink-segmentation-2-5d-3dcnn-resnet3dcsn-fp16fold01?scriptVersionId=132226669

