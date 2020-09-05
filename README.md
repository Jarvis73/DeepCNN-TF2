# Examples for Tensorflow v2
Tensorflow v2 examples of baseline networks.

## 0. Requirements

* python=3.7
* tensorflow=2.2.0
* numpy=1.18.8
* cudatoolkit=10.1.243
* cudnn=7.6.5
* opencv=3.4.2
* tqdm


## 1. Datasets

- **mnist** (Auto-download): 60000 training images with shape 28x28 and 10000 test images.
- **Cifar** (Auto-download): 60000 training 


## 2. Training/Evaluating Model

### 2.1 Start with Tensorflow v2

Training a simple CNN model with the API `fit()` of the `tf.Keras`:

Usage: `python xxx.py <mode> <gpu> [options]`

```bash
python main_mini_cnn7layers.py train 0
```

Test model with test dataset:

```bash
python main_mini_cnn7layers.py test 0
```

### 2.2 Custom training/testing loop

```bash
python main_custom.py train
```

### 2.3 Start Tensorboard

```bash
tensorboard --logdir ./model_dir --port 16006
# Expore to the network
tensorboard --logdir ./model_dir --port 16006 --host 0.0.0.0
```

## 3. Visualization Data with Tensorboard Projector

```bash
python projector/mnist/main_cnn7layers.py
# With sprite image
python projector/mnist/main_cnn7layers.py -s
```

Open terminal in `projector/mnist/dst_ckpt` and open tensorboard by 

```bash
tensorboard --logdir ./
```

Open browser with address: `http://localhost:6006/#projector` 

Results in 2D (MNIST embedding):

![mnist embedding](https://raw.githubusercontent.com/Jarvis73/DeepCNN-TF2/master/projector/mnist/tSNE-mnist-features.png)


## 4 LDA loss

Reference:

> Dorfer M, Kelz R, Widmer G. Deep linear discriminant analysis[J]. arXiv preprint arXiv:1511.04707, 2015.

```bash
python main_lda.py train
```
