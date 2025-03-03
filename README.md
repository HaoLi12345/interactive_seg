# interactive_seg
A deep learning-based interactive segmentation method is being conducted for [CVPR 2025: Foundation Models for Interactive 3D Biomedical Image Segmentation](https://www.codabench.org/competitions/5263/)


## before you start
1. The model takes prompts and 3D images of size (128x128x128) as input to produce segmentations in each iteration.
2. The model training follows a naive approach by processing multi-class images as single-class instance segmentation. You can modify this to accelerate training.
3. A hybrid (parallel) encoder combines a naive CNN and a ViT, making it easy to adapt pretrained weights from foundation models.
4. The code directly crops patches based on given bounding boxes. For details on box generation, refer to this [script](https://github.com/JunMa11/CVPR-MedSegFMCompetition/blob/main/get_boxes.py). Note that the generated boxes do not cover the entire target region. Also, you may reduce the input size for efficiency.
5. In each iteration, point prompts are randomly sampled from false negative and false positive regions. The sampling process extends from the center point to form a cube-shaped 3x3x3 region. For multi-class segmentation, you can use positive prompts from other classes as negative prompts, though this is not implemented in the current code.
6. You may design your own label fusion strategy for evaluation. Please refer to step 4.
7. This code has been trained on the full 16GB training set as well as a subset containing only abdominal and liver tumor datasets. [Dataset information](https://www.codabench.org/competitions/5263/)
8. The data has missing keys, i.e., "boxes" for some images, which are simply skipped. Please make sure to review the dataloader for the non-omitted final evaluation.
9. The evaluation code is not following the official format, and more information can be found [here](https://www.codabench.org/competitions/5263/)

## before training

Please have a look at the readme.txt file under the useful_scripts folder to generate training data. It will produce a .json type file for training/validation split.


## training

```
python train.py --split_path your_split_json_file_path --save_name your_save_name
```


## test

```
python test.py --data_dir your_test_image_dir --label_dir your_test_label_dir --save_name your_save_name
```

## discussion

I have attached my training and validation logs. Briefly, this model works very well as a baseline method. You can find the screenshots in the fig folder. Specifically, I only trained on full (10 percent 16G) dataset for 1 epoch. In contrast, I trained on selected subset (AbdomenAtlas and LiverTumor) on 5 epochs, where the results can be found in test logs.

Importantly, the training process takes a bit longer than I thought, as the main scope is to build a generic method. In addition, I used generic BCE loss, which is not the best option for medical segmentation.
However, you can train with multiple classes (please refer #2 in #before you start) at each iteration and reduce the image size. This would lead better performance.


## contact
This work is built based on our [previous work](https://github.com/MedICL-VU/PRISM), you can find more information there. Please send me an [email](hao.li.1@vanderbilt.edu) for any questions
 




