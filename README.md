<div align="center">

# FreeMorph: Tuning-Free Generalized Image Morphing with Diffusion Model
  
<a href="https://yukangcao.github.io/">Yukang Cao</a><sup>\*</sup>,
<a href="https://chenyangsi.top/">Chenyang Si</a><sup>\*</sup>,
<a href="https://personal-page.wjh.app/">Jinghao Wang</a>,
<a href="https://liuziwei7.github.io/">Ziwei Liu</a><sup>†</sup>


[![Paper](http://img.shields.io/badge/Paper-arxiv.2507.01953-B31B1B.svg)](https://arxiv.org/abs/2507.01953)
<a href="https://yukangcao.github.io/FreeMorph/"><img alt="page" src="https://img.shields.io/badge/Webpage-0054a6?logo=Google%20chrome&logoColor=white"></a>

<img src="./docs/static/FreeMorph-teaser.png">
  
Please refer to our webpage for more visualizations.
</div>

## Abstract
We present FreeMorph, the first tuning-free method for image morphing that accommodates inputs with different semantics or layouts. Unlike existing methods that rely on fine-tuning pre-trained diffusion models and are limited by time constraints and semantic/layout discrepancies, FreeMorph delivers high-fidelity image morphing without requiring per-instance training. Despite their efficiency and potential, tuning-free methods face challenges in maintaining high-quality results due to the non-linear nature of the multi-step denoising process and biases inherited from the pre-trained diffusion model. In this paper, we introduce FreeMorph to address these challenges by integrating two key innovations. 1) We first propose a guidance-aware spherical interpolation design that incorporates explicit guidance from the input images by modifying the self-attention modules, thereby addressing identity loss and ensuring directional transitions throughout the generated sequence. 2) We further introduce a step-oriented variation trend that blends self-attention modules derived from each input image to achieve controlled and consistent transitions that respect both inputs. Our extensive evaluations demonstrate that FreeMorph outperforms existing methods, being 10X ~ 50X faster and establishing a new state-of-the-art for image morphing.


## Install
```bash
# python 3.8 cuda 12.1 pytorch 2.1.0
conda create -n freemorph python=3.8 -y && conda activate freemorph
conda install -c "nvidia/label/cuda-12.1.0" cuda-toolkit

pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121

# other dependencies
pip install -r requirements.txt

```

## Image pairs preparation
The folder that contains the image pairs should have the structures like:
```
image_pairs/
    ├── pair1_0.jpg
    ├── pair1_1.jpg
    ├── ...
    ├── pairN_0.jpg
    ├── pairN_1.jpg
```


## Captioning the image pairs

```bash
python caption.py --image_path /PATH/TO/PAIRED_IMAGES --json_path /PATH/TO/DESIRED/CAPTION_PATH
```

## Running FreeMorph
```bash
python freemorph.py --json_path /PATH/TO/DESIRED/CAPTION_PATH
```

## Morph4Data
The 4-class evaluation data, Morph4Data, is now released. You can download the dataset from [Google Drive](https://drive.google.com/file/d/1QLMzGWb-hTLu96JamhDmwfRf7QdIOqX9/view?usp=sharing) or [OneDrive](https://entuedu-my.sharepoint.com/:u:/g/personal/yukang_cao_staff_main_ntu_edu_sg/EVWvUED4QNVHih_H4wOrdKwBotVA27ouUJZXtaDaNPJn-w?e=7Lau1d)


## Misc.
If you want to cite our work, please use the following bib entry:
```
@article{cao2025freemorph,
  title={FreeMorph: Tuning-Free Generalized Image Morphing with Diffusion Model},
  author={Cao, Yukang and Si, Chenyang and Wang, Jinghao and Liu, Ziwei},
  journal={arXiv preprint arXiv:2507.01953},
  year={2025}
}
```
