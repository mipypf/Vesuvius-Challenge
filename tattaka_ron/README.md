# tattaka&ron part solution

## 1. INSTALLATION
The requirements.txt file should list all Python libraries:
```
pip install -r requirements.txt
```
## 2. DATA

Place competition data as follows:
<pre>
input
└── vesuvius-challenge-ink-detection
</pre>
## 3. PREPROCESSING
Please do the following:
```
python tattaka_ron/input/split_fragment2.py
```
Next:
```
tattaka_ron/input/make_patch_32_5fold.py
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
sh tattaka_ron/src/exp055/train.sh
```
Next:
```
sh tattaka_ron/src/exp056/train.sh
```
## 5. INFERENCE
Actual inference notebook:
https://www.kaggle.com/code/mipypf/ink-segmentation-2-5d-3dcnn-resnet3dcsn-fp16fold01?scriptVersionId=132226669

