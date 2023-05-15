# ResNetによる画像認識
## 概要
有名モデルResNetをCIAFRデータセットで学習させることができます。
[ResNet](https://www.cv-foundation.org/openaccess/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html)はモデルサイズを18,34,50,101,152から選択可能で、CIFARデータセットはCIAFR10, CIFAR100の何れかを選択できます。

ResNetはResidial Block（残差接続層）と呼ばれる構造を導入することで、よりモデルの層の数を増やして学習できるようにしたモデルです。

残差接続層

![image](https://github.com/sshi2/image_recognition_resnet_cifar/assets/92853315/1faa6694-2966-4b28-86db-6b2195ddf250)


## 学習
学習設定は以下の５つ指定できます。
- データセット（-d）
  - CIFAR10, CIFAR100から選べます。
  - デフォルトはCIFAR10です。
- モデル（-m）
  - ResNet18, 34, 50, 101, 152から選べます。
  - デフォルトはResNet18です。
- バッチサイズ（-bs）
  - int型で指定できます。
  - デフォルトは16です。
- 学習率（-lr）
  - float型で指定できます。
  - デフォルトは0.001です。
- モメンタム（-mt）
  - floatで指定できます。
  - デフォルトは0.9です。

学習は以下のコードで実行できます。
``` Python
python main.py
```
また、以下の設定で実行したい場合は続いて引数を指定します。指定しない場合はデフォルト値になります。
- データセット：CIFAR100
- モデル：ResNet50
- バッチサイズ：32
- 学習率：0.01
- モメンタム：0.8
``` Python
python main.py -d "CIFAR100" -m 50 -bs 32 -lr 0.01 -mt 0.8
```
