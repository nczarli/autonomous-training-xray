# Autonomous Training in X-Ray Imaging Systems



<p align="center">
  <img src="https://github.com/nczarli/autonomous-training-xray/assets/58233641/7cd06be2-b0a6-4084-b838-05769dc9d582" alt="Sublime's custom image"/>
</p>

### Summary

This project aims to optimize the quality control process for processed cherries by developing a product capable of detecting defective cherries. The current limitations of supervised image classification methods, which require large hand-labelled datasets, are addressed through the implementation of an image processing algorithm. By partnering with Cheyney Design & Development Ltd., a manufacturer of X-Ray machines, the goal is to deliver a module for Delphi, capable of processing 22,000 cherries per minute. The target market for this product is cherry producers in Turkey, a country with a significant annual cherry production. While previous works achieved high accuracy in detecting pits in X-Rays of cherries, they required fine-tuning. This project explores robust training approaches, including semi-supervised object detection and noisy label training methods, to create datasets for machine learning. The effectiveness of these approaches was empirically demonstrated on synthetically generated and real datasets of X-Ray images, with the rule-based algorithm providing sufficient accuracy for machine learning training. Sample Selection and Relabelling, a noisy label method, outperformed the industry standard accuracy, paving the way for a general-purpose training procedure in defect detection tasks.

### Requirements
- 64 Bit Windows OS
- Python 3.9
- Delphi 10.4

### Installation
Delphi modules to install
- [P4D](https://github.com/pyscripter/python4delphi)
- [PythonEnvironments](https://github.com/Embarcadero/PythonEnviroments)
- [Lightweight-Python-Wrappers](https://github.com/Embarcadero/Lightweight-Python-Wrappers)

Install Anaconda (Python environment manager)

- Add Anaconda to Path
- Install [PyTorch](https://pytorch.org/get-started/locally/) on the base environment

Paths
- Change paths to local paths

### References
[SSR: An Efficient and Robust Framework for Learning with Noisy Labels](https://arxiv.org/abs/2111.11288)
