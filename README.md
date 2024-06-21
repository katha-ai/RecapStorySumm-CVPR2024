<!-- # RecapStorySumm-CVPR2024

## Contents

## About

## Setting up repository

## Download PlotSnap (SumMe & TVSum) Features

## [Optional] Feature Extraction (only if we are providing the raw data)

## Train
### change the values in config for kind of splits and how many splits?

## Inference
### Download weights
### Which checkpoint to test on?

## Cite -->

<div align="center">
    <h1> 🎬⏮️ <a href="https://arxiv.org/abs/2405.11487">"Previously on ..." From Recaps to Story Summarization
</a><br>
    <a href="https://img.shields.io/badge/python-3.8-blue"><img src="https://img.shields.io/badge/python-3.8-blue"></a>
    <a href="https://img.shields.io/badge/made_with-pytorch-red"><img src="https://img.shields.io/badge/made_with-pytorch-red"></a>
    <a href="https://img.shields.io/badge/dataset-PlotSnap-orange"><img src="https://img.shields.io/badge/dataset-Plotsnap-orange"></a>
    <a href="https://katha-ai.github.io/projects/recap-story-summ/"><img src="https://img.shields.io/website?up_message=up&up_color=green&down_message=down&down_color=red&url=https%3A%2F%2Fkatha-ai.github.io%2Fprojects%2Frecap-story-summ&link=https%3A%2F%2Fkatha-ai.github.io%2Fprojects%2Frecap-story-summ"></a>
    <a href="https://arxiv.org/abs/2405.11487"><img src="https://img.shields.io/badge/arXiv-2304.05634-f9f107.svg"></a>
    <a href="https://www.youtube.com/watch?v=VlB9O1vOz5c"><img src="https://badges.aleen42.com/src/youtube.svg"></a>
    <a href="https://i.giphy.com/media/v1.Y2lkPTc5MGI3NjExeDZqdnhyaXgydnltM3JxcmNzazZkanJtanpnYnlqdzZkNzBmc3JuYiZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/3o6ZtgDOEj7K8EYFEc/giphy.gif">
    <a href=""><img src="https://img.shields.io/badge/demo-coming_soon!-blue"></a>
    <!-- ![Static Badge](https://img.shields.io/badge/demo-coming_soon!-blue) -->
    <video width="800px" height="auto" controls>
    <source src="https://katha-ai.github.io/projects/recap-story-summ/static/videos/S08E23_recap.mp4" type="video/mp4">
    </video>
    </a>
</div>

## 📑 Contents
1. [About](#🤖-about)
2. [Setting up the repository](#⚙️-setting-up-the-repository)
    1. [Create a virtual environment](#🐍-create-a-python-virtual-environment)
    2. [Update the config template](#🛠️-configure-the-configsbaseyaml-file)
3. [Feature Extraction](#🔍-feature-extraction)
4. [Downloading and Setting up the data directories](#📥-download)
    1. [PlotSnap features](#🗃️-plotsnap-features)
    2. [TaleSumm pre-trained weights](#🦾-talesumm-pre-trained-weights)
    <!-- 2. [Pre-trained feature backbones](#👐-pre-trained-feature-backbones) -->
5. [Train TaleSumm with different configurations](#🏋️-train)
6. [Inference on TaleSumm to create summaries](#inference)
7. [License](#📜-license)
8. [Bibtex](#📍-cite)

## 🤖 About
This is the official code repository for [**CVPR-2024**](https://cvpr.thecvf.com/Conferences/2024) accepted paper [**"Previously on ..." From Recaps to Story Summarization**](https://arxiv.org/abs/2405.11487). This repository contains the implementation of ***TaleSumm***, a Transformer-based hierarchical model on our proposed dataset ***PlotSnap***. *TaleSumm* processes entire episodes by creating compact shot 🎞️ and dialog 🗣️ representations, and predicts importance scores for each video shot and dialog utterance by enabling interactions between *local story groups*. Our model leverages multiple modalities, including visual and dialog features, to capture a comprehensive understanding of important shots in complex movie environments. Additionally, we provide the pre-trained weights for the *TaleSumm* as well as all the pre-trained feature backbones used in feature extraction. On top of that, we provide pre-extracted features for episodes (per-frame embeddings using `DenseNet169`, `CLIP`, and `MViT`), and dialog features (with finetuned `RoBERTa` backbone).
<br>

## ⚙️ Setting up the repository

### 🐍 Create a `Python`-virtual environment
1. Clone the repository and change the working directory to be project's root.
    ```bash
    $ git clone https://github.com/katha-ai/RecapStorySumm-CVPR2024
    $ cd RecapStorySumm-CVPR2024
    ```
2. This project strictly requires `python==3.8`.

    Create a virtual environment using [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-with-commands).
    ```bash
    $ conda create -n storysumm python=3.8
    $ conda activate storysumm
    (storysumm) $ pip install -r requirements.txt
    ```
    OR

    Create a virtual environment using pip (make sure you have [Python3.8](https://www.python.org/downloads/release/python-380/) installed)
    ```bash
    $ python3.8 -m pip install virtualenv
    $ python3.8 -m virtualenv storysumm
    $ source storysumm/bin/activate
    (storysumm) $ pip install -r requirements.txt
    ```

### 🛠️ Configure the [`configs/base.yaml`](./configs/base.yaml) file

1. Add the absolute paths to the project directory in `configs/base.yaml`

2. E.g., If you have cloned the repository at `/home/user/RecapStorySumm-CVPR2024`, and want to download model checkpoints and the data features, then the path variables in `configs/base.yaml` would be-

    ```yaml
    root: "/home/user/RecapStorySumm-CVPR2024"
    # Save PlotSnap data features here
    data_path: "${root}/data"
    split_dir: "${root}/configs/data_configs/splits"
    # To save dialog (and vision) backbones
    cache_dir: "${root}/cache/"
    # use the following for model checkpoints 
    ckpt_path: "${root}/checkpoints/storysumm"
    ```

Refer to `configs/trainer_config.yaml` and `configs/inference_config.yaml` for the default parameter configuration while training and inferencing, respectively.



## 🔍 Feature Extraction
Follow the instructions in [feature_extractors/README.md](feature_extractors/README.md) **[WIP]** to extract required features from any given video and prepare it summarization. 
> **Note** that we have already provided the pre-extracted features for *PlotSnap* below.

## 📥 Download
### 🗃️ PlotSnap features
You can also use `wget` to download these files-

```bash
# Download the features (as mentioned below into data/ folder)
LINK="https://iiitaphyd-my.sharepoint.com/:f:/g/personal/makarand_tapaswi_iiit_ac_in/EiUo5uxNTklKp0ogbgIewfgBtk4liZul4fDah-9LQLKQDA?e=1mHebV"
wget -O data $LINK
```

|File name | Contents | Comments |
|----------|---------------|----------|
| 24 | <ul><li>Contains total of 8 seasons (S02 to S09).</li><li>Each season then consists of 24 episodes, except S09 that has 12 episodes.</li><li>Each episode consists of: <ol><li>`encodings/`: Consist video and dialog encodings</li><li>`scores/`: Different form of labels for both video and dialog.</li><li>`videvents/`: files that consists of starting and ending timing of constituent shots of recap and episode.</li><li>`SXXEXX.dfd`: Per-frame scores denoting the possibility of shot-boundary</li><li>`SXXEXX.matidx`: per-frame info on shot-index, frame-index, time (seconds:microseconds)</li><li>`SXXEXX.srt`: Dialog File (for visualization)</li><li>`shot_frames/`: 3 frames from each shot.</li></ol> </li></ul> | Contains `S02` to `S09` directories which will occupy 92GB of disk space. |
| Prison Break | <ul><li>Contains total of 2 seasons (`S02` & `S03`).</li><li>They consists of 22 and 13 episodes, respectively.</li><li>The episodes follow the same directory stucture as TV Show `24`.</li></ul> | This occupy 22GB of disk space. |


### 🦾 TaleSumm pre-trained weights
```bash

# Create the checkpoints folder `checkpoints/storysumm` in the project's root folder if not present already and put all checkpoints one-by-one in them.
mkdir -p <absolute_path_to_root>/checkpoints/storysumm

# OR (simply do the following).

# Now download the pre-trained weights (as mentioned below into ckpts/ folder)
LINK="https://iiitaphyd-my.sharepoint.com/:u:/g/personal/makarand_tapaswi_iiit_ac_in/ES91ZF90ArJGiXkEa53-kJABNytKOyOSQlr03dnTf6bKKg?e=shbnsj"
wget -O checkpoints $LINK
```
<!--
1. IntraCVT ka split 0 th ckpt.
2. Multiple labels split ka ckpt
    a. wget obatin the ckpts
    b. How to use these ckpts. (yaml config)
3. To obtain other split ckpts follow the train module.
    a. How to run the model to obatin ckpts  -->
| File name | Comments | Training command |
|-----------|----------|----------|
| `TaleSumm-IntraCVT\|S[1,2,3,4,5]` | IntraCVT split `i=0,1,2,3,4` checkpoint of TaleSumm | `(storysumm) $ python -m trainer split_type='intra-loocv'` |
| `TaleSumm-Final` | Final checkpoint of TaleSumm to be used in production | `(storysumm) $ python -m trainer split_type='final-split.yaml'` |

<!-- These models can be loaded using the following code-
```python
## <TODO> NEED TO ADD CODE ON HOW TO IMPORTANT THE CHECKPOINTS
``` -->

<!-- ### 👐 Pre-trained feature backbones
| File name | Comments |
|-----------|----------|
| [DenseNet169.pt](https://www.google.com) | [DenseNet169](https://openaccess.thecvf.com/content_cvpr_2017/papers/Huang_Densely_Connected_Convolutional_CVPR_2017_paper.pdf) pre-trained on ImageNet, SVHN, and CIFAR|
| [MViTv1_Kinetics400.pt](https://www.google.com) | [MViT_v1](https://openaccess.thecvf.com/content/ICCV2021/html/Fan_Multiscale_Vision_Transformers_ICCV_2021_paper.html) trained on Kinetics400 dataset |
| [CLIP.pt](https://www.google.com) | [OpenAI CLIP](https://openai.com/index/clip/) pre-trained on 4M image-text pairs
| [RoBERTa_finetuned_t10.pt](https://www.google.com) | [RoBERTa-large](https://aclanthology.org/2021.ccl-1.108/) pretrained on the reunion of five datasets : BookCorpus, English Wikipedia, CC-News, OpenWebText, Stories; further finetuned with an objective to predict the important dialogs | -->
<br>


## 🏋️ Train
After completing the above, now you can train *Talesumm* on a ***12GB*** Nvidia-2080 RTX-Ti GPU! You can also use the pre-trained weights provided in the [Download](#🦾-talesumm-pre-trained-weights) section.<br>

> <b>Note</b>: It is recommended to use [wandb](https://wandb.ai) to log & track your experiments

<!-- 
9. Change architecure (no. of layers in video encoder and talesumm decoder)
10. 
-->

Using the default values given in the `config_base.yaml`
1. To train *TaleSumm* for *PlotSnap*, use the default config (no argument required)
    ```bash
    (storysumm) $ python -m trainer
    ```

2. To train Talesumm with a specific modality (valid keywords- `vid`, `dia`, `both`)
    ```bash
    (storysumm) $ python -m trainer modality=both
    ```

3. To train Talesumm on a specific series  (valid keywords- `24`, `prison-break`, `all`)
    ```bash
    (storysumm) $ python -m trainer series='24'
    ```

4. To change the split type to be used for training (valid keywords- `cross-series`, `intra-loocv`, `inter-loocv`, `default-split.yaml`, `fandom-split.yaml`)
    ```bash
    (storysumm) $ python -m trainer split_type=cross-series
    ```

5. To choose which visual features to train on, create a list of the features to be used (valid keywords- `imagenet`, `mvit`, `clip`)
    ```bash
    (storysumm) $ python -m trainer visual_features=['imagenet','mvit','clip']
    ```

6. To choose the fusion style of the visual features (valid keywords- `concat`, `stack`, `mul`)
    ```bash
    (storysumm) $ python -m trainer feat_fusion_style=concat
    ```

7. To choose the type of attention in the model (valid keywords- `sparse`, `full`)
    ```bash
    (storysumm) $ python -m trainer attention_type=sparse
    ```

8. To disable **Group tokens** from the model
    ```bash
    (storysumm) $ python -m trainer withGROUP=False
    ```

    **NOTE** : If `withGROUP` is True then `computeGROUPloss` needs to be True as well


9. To enable wandb logging (recommended)
    ```bash
    (storysumm) $ python -m trainer wandb.logging=True
    ```
<br>

> **NOTE** : We have used 4 GPUs while training that is why the `gpus` parameter in the configuration is set to [0,1,2,3]. If you plan to more or less GPUs, please enter their GPU id's accordingly

## Inference
To summarise a new video using Talesumm, please follow the following commands

```bash
(storysummm) $ python -m inference <overrides for inference_config.yaml>
```

<br>

> **NOTE** : We have used 4 GPUs while training that is why the `gpus` parameter in the configuration is set to [0,1,2,3]. If you plan to more or less GPUs, please enter their GPU id's accordingly


## 📜 License
This code is available for **non-commercial scientific research purposes** as defined in the [LICENSE file](LICENSE). By downloading and using this code you agree to the terms in the [LICENSE](LICENSE). Third-party datasets and software are subject to their respective licenses.


## 📍 Cite
If you find any part of this repository useful, please cite the following paper!
```
@inproceedings{singh2024previously,
title={{"Previously on ..." From Recaps to Story Summarization}}, 
author={Aditya Kumar Singh and Dhruv Srivastava and Makarand Tapaswi},
booktitle = {IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
year={2024},
}
```

