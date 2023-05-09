# Idea
* 学習サンプル少ない？
* resumeできる仕組み作ってスポットインスタンスで本気出したい
* layerのどこに文字があるか特定するとか
* 小さいマス目で2.5dか3dで学習して、logitをさらに2d unetで学習する？
  * 一回ir.pngを推論する？同時？
* classificationでやるとして、近隣のパッチがTrueなら閾値を下げる
  * 関連: CRF
* classification + pointrend?
  * 理論値0.97pt出るしいらないかも？
* いつもの3重閾値？
* 後処理で、近傍がthresholdより上ならthreshold引き下げ的な
* 位置エンコーディングを入れる？
* slide window 2.5d
* ドメアダヘッド
* adabn
* backboneをfreezeして大きい解像度でtrain?
* 中間層もっと減らしても？


# WORK
* パッチごとに文字領域かどうかをclassificationする
  * 高速化のためにunpoolする前の出力を用いる？
* 最終出力にmaskを掛ける？
  * そんなに変わらない
* normalizationを色々試す
  * channel wise？
  * fragment wise?
  * maskを使う
    * 普通にやると過学習するっぽい感じある
  * channel wise + fragment wiseが過学習してなさげ
    * だけど学習自体が進んでない
  * patch wise normalization？
    * layer全体で平均をとる？
    * layer間の関わりをとらえるならそっちのが良い
  * Patchwiseでlayer全体でnormalizeを採用
* TTA
  * slideしていくので、順番に適用する
  * [input, input.flip(2), input.flip(3), input.flip(2).flip(3), input.flip(1), input.flip(1).flip(2), input.flip(1).flip(3), input.flip(1).flip(2).flip(3)]
  * 最終的に8TTA
* 3DCNN
  * CSN
    * DFL1st like
* 中間層だけを使う？
  * 65 - (8 * 2)とか
  * https://www.kaggle.com/code/danielhavir/vesuvius-challenge-example-submission/comments
  * https://www.kaggle.com/competitions/vesuvius-challenge-ink-detection/discussion/395330
* channelをaugmentation的にシフトさせる
  * LB上がった
* stochastic depth + dropout
  * 0.2 + 0.4
  * CV改善・LB若干改善
* 3D layerをlayer normで実装し直してみる
  * **推論時はpreprocess_in_model=Trueを忘れずに！**
  * 使うレイヤー数の数探索
    * 7x7
      * 0.6483
    * 6x7
      * 0.6521
    * 6x6
      * 0.6524
    * 5x7
      * 0.6624
    * 3x16
      * 学習遅い
      * 結果も微妙
    * 3x16_192
      * 192だと精度悪い
    * 2x18_192
    * 4x9
    * 3x9
      * わりと良い
    * 2x24_192
    * 5x6
    * 4x7
    * 3x7
* manifold mixup
* mixupを複数種使う(mixup + manifold + cutmix)
  * cutmix
* terekaさんの構造
* モデルサイズ変える
  * 101eにしてbs落とす
    * 速度は約1/2
  * 普通・時間が余ればensembleの種として良いかも
* channel shuffle
* cutmixを半分に増やす
* 40epoch学習させてみる
* Distortion周り足してみる
  * exp033参考
* 小さい領域を削除
  * https://github.com/tattaka/understanding-cloud-organization/blob/master/src/cloud_utils.py
  * 参考
  * 計算時間的にうまいこと小さい画像でできないか
    * というか閾値を切るのを小さい画像でやって最適化を大きい画像でやるといい感じに小領域をフィルタリングできない？
      * いや、残るはず
    * まあ小領域削除を頑張るのはこっちでやったほうが良さげ
    * 閾値計算も同じく
  * 流れとしては、 
    * 元画像を1/32した画像を使う
    * いつも通り閾値処理
      * 画像小さいし、雲コンペの二重閾値をminimize使っても良いかも
    * 最終評価の時だけupsampling
* ema
* label smoothing
* convnext
* 学習率を上げる
  * 正解っぽい
  * 多分、seed設定を直す前は想定よりも大きめlrで学習してたから過学習しなかったっぽい？
  * 今はlr=1e-3だけどもっと上げても良いかも？


# NOT WORK
* Mixupをprob=1で適用して残り4割くらい切ってみる
  * だめ
* 3D layer増やす
  * 微妙
* blur・noise系統試す
* cutout
  * https://albumentations.ai/docs/api_reference/augmentations/dropout/cutout/
* epoch数伸ばしてみる
* gaussian noiseとshapen・embossを合わせると良さげ？
  * CV良かったけどLB下がった？
  * 最適閾値も同じなのでとりあえず保留
* Project & Excite
  * https://arxiv.org/pdf/1906.04649.pdf
* Gaussian noiseをper channelでやってみる
  * TODO: pytorchで実装し直す
* MEMO: augmentation増やす系はCVは増加しているのであとでもう一回試しても良いかも
  * PEも良かった
* fbeta loss検証
  * exp012参照
* 3D SEBlock
  * SEBlock
    * exp014参照
  * EffectiveSE
    * https://github.com/huggingface/pytorch-image-models/blob/main/timm/layers/squeeze_excite.py
  * csse
    * https://github.com/ai-med/squeeze_and_excitation/blob/master/squeeze_and_excitation/squeeze_and_excitation_3D.py
* input batch norm
* 3D layerを増やす
* EfficientNet系列を学習させる
  * mixup切れば学習できるのでは？
  * 厳しい、なんで？
  * DDPが悪さしてたり......？
* SGD
  * 学習できねえ...
* subsampling
  * 単純に飛ばしでとる？
  * 3とか取る時は平均してしまっても？
  * frameの平均をとる
    * 従来->frameを1にしてしまう
    * 最初に平均を作ってから従来
* 3次元にしてから正規化
* 3D branch
  * init_0
* モデルサイズ落としてbsあげる
* 全結合head
* classification head
* aux loss
* patch cutout
  * これ効かないの謎
* IR画像とのマルチタスク
* epochを短くする(overfittingへの対処)
  * だめっぽい
* 内側のsliceを使う(overfittingへの対処)
  * だめっぽい

# TODO
* もう一回試す
  * input_bn
  * augmentation
  * tta検証
  * patch cutout
  * fbeta_loss
* sliceについて考えてみる
  * https://www.kaggle.com/code/ajland/eda-a-slice-by-slice-analysis
  * 内側のsliceを使ったほうが良い？
  * medianを使った外れ値に強い正規化
* mixup=1, 最後にmixupを切る
* shift_zを考える
* median正規化(NaNへの対処)
* random crop & resize
* alldata
  * どれか一個valセット使う
  * thresholdは平均とって考える
* モルフォロジー操作
  * https://docs.opencv.org/4.x/d9/d61/tutorial_py_morphological_ops.html
  * 閾値でやるより賢い気がする
* ink_label更新
  * 今度やる
* segformer encoder
* Transformer Head
* volume wise normalize
  * input_bnで代用できてた？
  * あまり効果なかったしいらないかも？
* TSM
* headにJPUとかASPPとか使う
  * https://github.com/tattaka/ukiyoe
* パッチベースで閾値を決める
  * いくつかの領域に分割して、閾値決め
* 画像サイズ大きくする
* pretrained 3d CNN
  * https://github.com/nvnnghia/nfl3_1st
* Sharp modelの後でUnetなどの大きめ解像度headつけて学習させる？
* 2Stage 
  * stackingも兼ねる？
  * 大きめの画像サイズで学習
    * 1024x1024とか
  * 複数checkpointを使う
    * ttaも含めで
  * 小さめのCNN+UNetとかで良いかも
    * resnet18の上数層とか  


# DONE
* trainとvalidの数を各foldで統一した
* 評価指標更新
* 評価バグなくす
* Diceの計算がバグってるっぽい
  * 直した
* 5fold
* trainをvalidと思い込んでいたので修正
* front・backに振ってどこのlayerが効いてるか見てみる
  * 真ん中っぽい
* 過去実験でバグが影響してそうなやつの洗い出し
  * mixup後にlossをもう一回計算してしまっていた
  * 洗い出し：
    * (備考までにexp042はmixup + ema)
    * 現行デフォ設定(exp043)
      * mixup + ema + label smoothing
    * cls head(exp041)
      * ただしemaとlabel smoothingは実装されていない
    * aux head(exp044)
  * exp045からは修正済み
  * 実験手順
    * exp043として修正版のベースモデルを実験
    * 次にexp045(->exp044)を実験してベースモデルにするかどうか考察する
      * exp043とexp044を長めepoch1実験ずつ流してみて確認する
      * patch_cutoutが効きそうだったら、exp41(->exp045), exp044(->exp046), exp046(->exp047)にも導入
      * それぞれのナンバリングは新しく整理する
        * loss関数の修正
        * loss計算の削除
        * emaの追加
* seed周りの設定見直し
* batchsizeを落としてみる？
  * overfitし出したのはexp049から
  * seed直してから


# ろんさんcode readingメモ
* normalizeでmax_pixel=255なので修正したほうが良いかも？
  * 問題なさそう


