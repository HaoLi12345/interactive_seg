# 3D interactive segmentation for medical images
A deep learning-based interactive segmentation method is being conducted for [CVPR 2025: Foundation Models for Interactive 3D Biomedical Image Segmentation](https://www.codabench.org/competitions/5263/)


## before you start
1. The model takes prompts and 3D images of size (128x128x128) as input to produce segmentations in each iteration. You may reduce the input size for efficiency.
2. The model training follows a naive approach by processing multi-class images as single-class instance segmentation. You can modify this to accelerate training.
3. A hybrid (parallel) encoder combines a naive CNN and a ViT, making it easy to adapt pretrained weights from foundation models.
4. The code directly crops patches based on given bounding boxes. For details on box generation, refer to this [script](https://github.com/JunMa11/CVPR-MedSegFMCompetition/blob/main/get_boxes.py). Note that the generated boxes do not cover the entire target region.
5. In each iteration, point prompts are randomly sampled from false negative and false positive regions. The sampling process extends from the center point to form a cube-shaped 3x3x3 region. **For multi-class segmentation, you can use positive prompts from other classes as negative prompts, though this is not implemented in the current code.**
6. You may design your own label fusion strategy for evaluation. Please refer to step 4.
7. This code has been trained on the full 16GB training set as well as a subset containing only abdominal and liver tumor datasets. [Dataset information](https://www.codabench.org/competitions/5263/)
8. The data has missing keys, i.e., "boxes" for some images, which are simply skipped. Please make sure to review the dataloader for the non-omitted final evaluation.
9. The evaluation code is not following the official format, and more information can be found [here](https://www.codabench.org/competitions/5263/)

## before training

Please have a look at the readme.txt file under the [useful_scripts](https://github.com/HaoLi12345/interactive_seg/edit/main/src/useful_scripts) folder to generate training data. It will produce a .json type file for the training/validation split.


## training

```
python train.py --split_path your_split_json_file_path --save_name your_save_name
```


## test

```
python test.py --data_dir your_test_image_dir --label_dir your_test_label_dir --save_name your_save_name
```

the console outputs from training and testing are saved in the [screenshot](https://github.com/HaoLi12345/interactive_seg/edit/main/screenshots) folder

## discussion
**Steps 3-6 should improve the results, but these haven't been coded yet.**

**You may focus on some specific organs/structures since some of them already have a high Dice score, such as above 0.9 or 0.95, which doesn't have too much room to be improved.**

Training and validation logs are attached in the [implementation](https://github.com/HaoLi12345/interactive_seg/edit/main/src/implementation) folder. 

Briefly, this model works very well as a baseline method. You can find the screenshots in this [folder](https://github.com/HaoLi12345/interactive_seg/edit/main/screenshots). 

Specifically, I only trained on full (10 percent 16G) dataset for 1 epoch. In contrast, I trained on a selected subset (AbdomenAtlas and LiverTumor) on 5 epochs, where the results can be found in test logs.

Importantly, the training process takes a bit longer than I thought, as the main scope is to build a generic method. In addition, I used generic BCE loss, which is not the best option for medical segmentation.
However, you can train with multiple classes (please refer to #2 in #before you start) at each iteration and reduce the image size. This would lead to better performance.


## results

**AbdomenAtlas: n=100** from independent validation set, mean dice = 0.802 (iteration=6), mean initial dice = 0.776 (iteration=1).


**full: n=2074** from independent validation set, mean dice = 0.802 (iteration=6), mean initial dice = 0.776 (iteration=1).


The above Dice scores are averaged by all subjects. <br />
For each subject, the Dice is derived from the mean of all classes. <br />
For each class, the Dice of the last iteration (iter=6)

The results are computed using cropped ground truths. Please refer to test.py

## execution time

**around 2 seconds** per image with 10 classes, i.e., 128 as input size and a total iteration of 6 for each class.

More details can be found in [log](https://github.com/HaoLi12345/interactive_seg/edit/main/src/implementation) folder.

## contact
This work is built based on our [previous work](https://github.com/MedICL-VU/PRISM), you can find more information there. Please send me an [email](hao.li.1@vanderbilt.edu) for any questions
 




