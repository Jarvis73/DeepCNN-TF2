# Examples for Tensorflow 2.0
Tensorflow 2.0 examples of baseline networks.

## 0. Requirements

* python==3.6
* tensorflow=2.0.0-beta1
* numpy==1.16.4
* cudatoolkit==9.2
* cudnn==7.6.0
* tqdm==4.32.1


## 1. Datasets

- **mnist** (Auto-download): 60000 training images with shape 28x28 and 10000 test images.


## 2. Train Model

### 2.0 Start with Tensorflow 2.0

Training a simple CNN model (1 conv layer and 2 dense layer) for mnist classification task:

```bash
python main_mini.py train
```

Test model with test dataset:

```bash
python main_mini.py test
```

## 3. TODO List

* [ ] Full usage and train/val/test schemes
* [ ] Baseline models
* [ ] Small datasets
* [ ] Data augmentation example
* [ ] Tensorboard example
