# interactive_seg
deep learning-based interactive segmentation method is being conducted for CVPR 2025: Foundation Models for Interactive 3D Biomedical Image Segmentation


## what does method do
1. The model takes prompts and 3D images of size (128x128x128) as input to produce segmentations in each iteration.
2. The model training follows a naive approach by processing multi-class images as single-class instance segmentation. You can modify this to accelerate training.
3. A hybrid (parallel) encoder combines a naive CNN and a ViT, making it easy to adapt pretrained weights from foundation models.
4. The code directly crops patches based on given bounding boxes. For details on box generation, refer to this [script](https://github.com/JunMa11/CVPR-MedSegFMCompetition/blob/main/get_boxes.py). Note that the generated boxes do not cover the entire target region. Also, you may reduce the input size for efficiency.
5. In each iteration, point prompts are randomly sampled from false negative and false positive regions. The sampling process extends from the center point to form a cube-shaped 3x3x3 region. For multi-class segmentation, you can use positive prompts from other classes as negative prompts, though this is not implemented in the current code.
6. You may design your own label fusion strategy for evaluation. Please refer to step 4.
7. This code has been trained on the full 16GB training set as well as a subset containing only abdominal and liver tumor datasets. [Dataset information](https://www.codabench.org/competitions/5263/)
