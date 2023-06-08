# Autonomous Training in X-Ray Imaging Systems



<p align="center">
  <img src="https://github.com/nczarli/autonomous-training-xray/assets/58233641/7cd06be2-b0a6-4084-b838-05769dc9d582" alt="Sublime's custom image"/>
</p>

### Summary
Supervised image classification methods have made significant progress but are limited by the requirement for large hand-labelled datasets, which is a time-consuming and labour-intensive process. In the context of processed cherries, this project aims to address these challenges and optimise the existing quality control process. By partnering with [Cheyney Design \& Development Ltd.](https://sapphire-inspection.com/cheyney-design/), a manufacturer of X-Ray machines, the goal is to develop a product capable of detecting defective cherries. The project aims to deliver a module for Delphi, the programming language used by Cheyney's X-Ray machines, capable of processing 22,000 cherries per minute. While previous works on a rule-based algorithm to detect pits (stones) in X-Rays of cherries achieved high accuracies, they required a lot of fine-tuning. Therefore, there is a need for a quick and easy way to create datasets for machine learning. Additionally, deep neural networks have been praised for their performance in image classification tasks. However, they have become so powerful that they can memorize noisy labels which in turn decreases how well they generalise to unseen data. Hence, this project explores robust training approaches, including semi-supervised object detection and noisy label training methods. The effectiveness of these approaches was empirically demonstrated on synthetically generated and real datasets of X-Ray images. Sample Selection and Relabelling, a noisy label method, outperformed the industry standard accuracy, paving the way for a general-purpose training procedure in defect detection tasks.

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
