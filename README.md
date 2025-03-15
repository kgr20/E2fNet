## E2fNet - Official Implementation
**From Brainwaves to Brain Scans: A Robust Neural Network for EEG-to-fMRI Synthesis**

**Authors**: Kristofer Grover Roos, Atsushi Fukuda, Quan Huu Cap<br>
**Paper**: https://arxiv.org/abs/2502.08025<br>

**Abstract**
While functional magnetic resonance imaging (fMRI) offers rich spatial resolution, it is limited by high operational costs and significant infrastructural demands. 
In contrast, electroencephalography (EEG) provides millisecond-level precision in capturing electrical activity but lacks the spatial resolution necessary for precise neural localization. 
To bridge these gaps, we introduce E2fNet, a simple yet effective deep learning model for synthesizing fMRI images from low-cost EEG data. 
E2fNet is specifically designed to capture and translate meaningful features from EEG across electrode channels into accurate fMRI representations. 
Extensive evaluations across three datasets demonstrate that E2fNet consistently outperforms existing methods, achieving state-of-the-art results in terms of the structural similarity index measure (SSIM). 
Our findings suggest that E2fNet is a promising, cost-effective solution for enhancing neuroimaging capabilities.

### Required env
- CUDA >= 11
- Python >= 3.10
- Poetry >= 2.0
  
### Installation
```bash
poetry install --no-root
```

### Datasets pre-processing
See [datasets pre-processing](docs/datasets_howto.md) for download and pre-process all datasets

### Training
Modify [run_bash.sh](run_train.sh) and run
```bash
bash run_train.sh
```
**Note: The current code loads ALL training data to RAM, if you don't have enough RAM, please modify the [eeg2fmri_datasets.py](eeg2fmri_datasets.py) to load smaller chunks of data**

### Pre-trained models
TBD ...

### Inference
See [inference.ipynb](inference.ipynb) for more details. 

### Citation

```
@article{kristofer2025e2fnet,
  title   = {From Brainwaves to Brain Scans: A Robust Neural Network for EEG-to-fMRI Synthesis},
  author  = {Kristofer Grover Roos and Atsushi Fukuda and Quan Huu Cap},
  journal = {arXiv preprint},
  year    = {2025}
}
```