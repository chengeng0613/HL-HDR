# HL-HDR: Multi-Exposure High Dynamic Range Reconstruction with High-Low Frequency Decomposition

By Xiang Zhang,Genggeng Chen,Tao Hu,Kangzhen Yang,Fan Zhang,Qingsen Yan

## Abstract 
Generating high-quality High Dynamic Range (HDR) images in dynamic scenes is particularly challenging. 
Recent Transformer have been introduced in HDR imaging, demonstrating promising performance, particularly in scenarios involving large-scale motion compared to previous CNN-based methods. However, Transformer-based methods face hurdles capturing local details and come with high computational complexity, hindering further progress. In this paper, inspired by the distinct characteristics of high and low-frequency in image patterns, we propose a Frequency Decomposition Processing Block (FDPB) for ghost-free HDR imaging.
In the image reconstruction process, FDPB decouples features into resolution-invariant high-frequency features and resolution-reduced low-frequency features to separately address local and global information.
Specifically, considering the characteristics of different frequencies, for the high-frequency components, we design a Local Feature Extractor (LFE) based on CNN to extract local feature maps. Meanwhile, for the low-frequency components, we propose a Global Feature Extractor (GFE) that learns long-range dependencies through carefully designed Transformer modules. Importantly, the downscaled low-frequency features exploit Transformer's remote learning capabilities while substantially reducing self-attention computational costs.
By incorporating the FDPB as basic components, we further build a Low/High-Frequency Aware Network (HL-HDR), a hierarchical network to reconstruct high-quality ghost-free HDR images. Extensive experiments on four public datasets confirm the superior performance of the proposed method, both in terms of quantitative and qualitative evaluations.

## Pipeline
![pipeline](https://github.com/chengeng0613/HL-HDR/blob/main/picture/overview.png)
The architecture of the proposed HL-HDR. HL-HDR consists of two components. The first component is the feature alignment stage, where an Alignment Module is used to align the overexposed and underexposed images with the normally exposed images, which serves as the reference frame. The second component is the feature extraction, where the feature map is divided into high-frequency and low-frequency information, and processed separately based on their characteristics.



## Usage

### Requirements
* Python 3.7.0
* CUDA 10.0 on Ubuntu 18.04

Install the require dependencies:
```bash
conda create -n hlhdr python=3.7
conda activate hlhdr
pip install -r requirements.txt
```

### Dataset
1. Download the dataset (include the training set and test set) from [Kalantari17's dataset](https://cseweb.ucsd.edu/~viscomp/projects/SIG17HDR/)
2. Move the dataset to `./data` and reorganize the directories as follows:
```
./data/Training
|--001
|  |--262A0898.tif
|  |--262A0899.tif
|  |--262A0900.tif
|  |--exposure.txt
|  |--HDRImg.hdr
|--002
...
./data/Test (include 15 scenes from `EXTRA` and `PAPER`)
|--001
|  |--262A2615.tif
|  |--262A2616.tif
|  |--262A2617.tif
|  |--exposure.txt
|  |--HDRImg.hdr
...
|--BarbequeDay
|  |--262A2943.tif
|  |--262A2944.tif
|  |--262A2945.tif
|  |--exposure.txt
|  |--HDRImg.hdr
...
```
3. Prepare the corpped training set by running:
```
cd ./dataset
python gen_crop_data.py
```

### Training & Evaluaton
```
cd HL-HDR
```
To train the model, run:
```
python train.py --model_dir experiments
```
To test, run:
```
python fullimagetest.py
```

## Results
![results](https://github.com/chengeng0613/HL-HDR/blob/main/picture/compare.png)



## Citation
```
@inproceedings{zhang2024hl,
  title={HL-HDR: Multi-Exposure High Dynamic Range Reconstruction with High-Low Frequency Decomposition},
  author={Zhang, Xiang and Chen, Genggeng and Hu, Tao and Yang, Kangzhen and Zhang, Fan and Yan, Qingsen},
  booktitle={2024 International Joint Conference on Neural Networks (IJCNN)},
  pages={1--9},
  year={2024},
  organization={IEEE}
}
```

## Contact
If you have any questions, feel free to contact Genggeng Chen at chengeng0613@gmail.com.

## Checkpoints
The following links are the weights of the Kalantari dataset and the Hu dataset:https://drive.google.com/drive/folders/1XypCej20LUwbba8z-kK7ErMryLoZ4iu5?usp=drive_link
