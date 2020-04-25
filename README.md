# AI Programming with Python Project

Project code for Udacity's AI Programming with Python Nanodegree program. In this project, I developed code for an image classifier built with PyTorch, then convert it into a command line application.

Programmer: Leo Tomatsu

Date Created: 4/23/2020

---

## Train.py CLI application
```
usage: train.py [-h] [--arch {densenet,vgg,alexnet}] [--save_dir SAVE_DIR]
                [--lr LR] [--hl HL] [--epochs EPOCHS] [--gpu]
                data_dir

positional arguments:
  data_dir              Path to data files.

optional arguments:
  -h, --help            show this help message and exit
  --arch {densenet,vgg,alexnet}
                        CNN model architecture.
  --save_dir SAVE_DIR   Path to save checkpoint files.
  --lr LR               Learning rate for optimizer.
  --hl HL               Number of hidden layers.
  --epochs EPOCHS       Number of epochs.
  --gpu                 Utilize gpu to train.
```
### Example usage
```
$ python train.py flowers --arch 'densenet' --save_dir checkpoints --lr 0.001 --hl 1024 --epoch 5 --gpu

Training with DenseNet architecture pre-trained model || epochs=5 || device=cuda:0
Epoch: 1/5... Loss: 1.7641 Validation Loss 0.1717 Validation Accuracy: 0.0805
Epoch: 1/5... Loss: 1.6124 Validation Loss 0.1807 Validation Accuracy: 0.1779
Epoch: 2/5... Loss: 0.4765 Validation Loss 0.1210 Validation Accuracy: 0.3821
Epoch: 2/5... Loss: 0.9488 Validation Loss 0.1202 Validation Accuracy: 0.4586
Epoch: 2/5... Loss: 0.7985 Validation Loss 0.0917 Validation Accuracy: 0.5083
Epoch: 3/5... Loss: 0.5728 Validation Loss 0.0511 Validation Accuracy: 0.6263
Epoch: 3/5... Loss: 0.5958 Validation Loss 0.0468 Validation Accuracy: 0.6708
Epoch: 4/5... Loss: 0.1467 Validation Loss 0.0616 Validation Accuracy: 0.6740
Epoch: 4/5... Loss: 0.4848 Validation Loss 0.0415 Validation Accuracy: 0.7378
Epoch: 4/5... Loss: 0.4929 Validation Loss 0.0359 Validation Accuracy: 0.7186
Epoch: 5/5... Loss: 0.3008 Validation Loss 0.0402 Validation Accuracy: 0.7252
Epoch: 5/5... Loss: 0.4257 Validation Loss 0.0379 Validation Accuracy: 0.7549
Total Training Time: 00:13:09
Accuracy of the network on the test images: 74.11477411477412 %%
Model saved successfully as checkpoints/densenet_17_06_19_checkpoint.pth
```

## Predict.py CLI appication
```
usage: predict.py [-h] [--fp_classmap FP_CLASSMAP] [--top_k TOP_K] [--gpu]
                  fp_image fp_checkpoint

positional arguments:
  fp_image              Path to image file.
  fp_checkpoint         Path to checkpoint file.

optional arguments:
  -h, --help            show this help message and exit
  --fp_classmap FP_CLASSMAP
                        Path to classes map file.
  --top_k TOP_K         Number of highest probabilities.
  --gpu                 Use gpu to train.
```

### Example Usage
```
$ python predict.py flowers/test/1/image_06743.jpg checkpoints/densenet_17_06_19_checkpoint.pth --fp_classmap cat_to_name.json --top_k 5 --gpu

Predicting the top 5 classes with DenseNet pre-trained model | device=gpu.
Classes           : ['pink primrose', 'tree poppy', 'hibiscus', 'primula', 'balloon flower']
Probabilities (%): [62.11, 11.24, 8.07, 5.03, 3.51]
Most probable class: pink primrose
```