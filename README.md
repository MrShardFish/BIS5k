BIS5k: A large-scale dataset for medical segmentation task based on HE-staining images of breast cancer

# BIS5k: A large-scale dataset for medical segmentation task based on HE-staining images of breast cancer

This is a official repo for "BIS5k: A large-scale dataset for medical segmentation task based on HE-staining images of breast cancer". If you use our related data for your researches,  please citing our work!!! Thank you.

## Introduction of our work.
Breast cancer, a high-incidence cancer among female, occupies a large incidence of total female patients with cancer. Pathological examination is the gold standard for breast cancer in clinic diagnosis. However, accuracy and efficient diagnosis is challengeable to pathologists for the complex of breast cancer and laborious work. 
In this work, we release a large-scale and hematoxylin-eosin (HE) staining  dataset of breast cancer for medical image segmentation task, called the breast-cancer image segmentation 5000 (BIS5k). BIS5k contains 5929 images that are divided into training data (5000) and evaluated data (929). All images of BIS5k are collected from clinic cases which include patients with various age and cancer stages. All labels of images are annotated in pixel-level for segmentation task and reviewed by pathological professors carefully. 
![在这里插入图片描述](https://img-blog.csdnimg.cn/direct/981cea92614e48f29f8539919b860d82.png)
![Details of previous datasets on breast cancer](https://img-blog.csdnimg.cn/direct/0bbe23bfefc84bd3bd1fd9a2520c500c.png)

Everyone can download BIS5k according the download links only for researching purposes.
Baidu Netdisk: https://pan.baidu.com/s/1cXQHriWBzPZblWDiGBIJ9w
Code: fx28
 
Evaluated toolkit can be accessed with BIS5k.
This toolkit was improved from Polyp Segmentation Task (UACANet) and could calculate variou metrics (including: dice, iou, and wFm etc). You should put your segmentation results into `./BIS5K_results`, and then change the configs in `./configs/BCSNet.yaml`. Runing the `./Eval.py` to calculate evaluated results.

## How to use BIS5k.
We proposed pathological images with corresponding masks. You can use image-and-mask pairs to develop your supervised, unsupervised, or semi-supervised methods. We also provided evaluated toolkit in this work. You can evaluate your segmentation results with it. The file structure can be found as follows:
```python
// Files
-BIS5k:
    -formal_train: #training data dir
    	-images: #training images dir
    		-bis_he_id000000.png
    		-bis_he_id000001.png
    		...
    	-masks: #trianing masks of corresponding images dir
    	    -bis_he_id000000.png
    		-bis_he_id000001.png
    		...
    -formal_test: #training data dir
    	-images: #training images dir
    		-bis_val_id000000.png
    		-bis_val_id000001.png
    		...
    	-masks: #trianing masks of corresponding images dir
    	    -bis_val_id000000.png
    		-bis_val_id000001.png
    		...
```
You can reorganize directory structure for fiting your project(s).
We encourage you to open your codes and evaluated results, which can effiectively improve development of CAD methods in pathological diagnosis. Meanwhile, due to our restricted developing platform and capacity, we encourage developers to optimize compared methods and open evaluated results. Thank you.
## Evaluated Toolkit
This toolkit was improved from Polyp Segmentation Task (UACANet) and could calculate variou metrics (including: dice, iou, and wFm etc). You should put your segmentation results into `./BIS5K_results`, and then change the configs in `./configs/BCSNet.yaml`. Runing the `./Eval.py` to calculate evaluated results.

## Breast-cancer Segmentation Network (BCSNet)
Furthermore, we also introduce a method (breast-cancer segmentation network, BCSNet) as the benchmark to demonstrate the usage of BIS5k. 

### Training
You should down the pre-trained pth and put them into  `./data/backbone_ckpt`. Then, runing the script `./run/Train.py` to train the model.
Baidu Netdisk: https://pan.baidu.com/s/1o0fKhr8Xg0nDyQ0RCSCU8Q 
Code：cmdv 

### Testing
Runing the script `./run/Test.py` to test the model.

### Testing with our pre-trained weights
Downloading the weight from and putting them into `./run/snapshots/backbone_ckpt/BCSNet`. Then, runing the script `./run/Test.py` to test the model.
Baidu Netdisk: https://pan.baidu.com/s/1WolxnG6Z4TCRkignig9hWQ 
Code：69go
