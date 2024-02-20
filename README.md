# Image Classifier: Transfer Learning

## Flower Classifier - A Deep Learning Project

This repository contains a flower classifier built for Udacity's AI Programming with Python Nanodegree. The classifier is built using PyTorch and can be trained on any set of labelled images. The current version is trained to recognize 102 different species of flowers.

### Project Structure

The project consists of a Jupyter Notebook that contains the entire workflow and two Python executables:

1. `train.py`: Trains the classifier. 
2. `predict.py`: Uses the trained classifier to predict the species of a flower from an image.

### Prerequisites

The code is written in Python 3.6.5. You will need the following packages: Numpy, Pandas, Matplotlib, PyTorch, PIL, and json. To install PyTorch, follow the instructions on the [PyTorch website](https://pytorch.org/).

### Training the Classifier

To train the classifier, run `train.py` with the path to the training data directory as a mandatory argument. For example:

```bash
python train.py data_directory
```

You can customize the training process with the following optional arguments:

- `--save_dir`: The directory to save checkpoints.
- `--arch`: The architecture to use for the neural network. Options are `alexnet` (default) , `vgg19` and `effecientnet-b0`.
- `--learning_rate`: The learning rate for gradient descent. Default is 0.001.
- `--hidden_units`: The number of neurons in an extra hidden layer, if chosen.
- `--epochs`: The number of epochs. Default is 25.
- `--gpu`: Specify `gpu` if a GPU is available. The model will use the CPU otherwise.
- `--data_dir`: Path to dataset.

### Using the Classifier

To use the classifier, run `predict.py` with the path to the input image as a mandatory argument. For example:

```bash
python predict.py /path/to/image
```

You can customize the prediction process with the following optional arguments:

- `--load_dir`: The path to the checkpoint.
- `--top_k`: The number of top K-classes to output. Default is 5.
- `--category_names`: The path to a JSON file mapping categories to names.
- `--gpu`: Specify `gpu` if a GPU is available. The model will use the CPU otherwise.
- `--image_dir`: Input image path. Mandatory.

### Data and JSON File

The data used for this project are not provided in the repository due to their size. However, you can create your own datasets and train the model on them. The data should be organized into three folders: `test`, `train`, and `validate`. Inside these folders, there should be folders bearing a specific number which corresponds to a specific category, clarified in the JSON file.

### Hyperparameters

Choosing the right hyperparameters can be challenging. Here are some tips:

- Increasing the number of epochs improves the accuracy of the network on the training set, but too many epochs can lead to overfitting.
- A large learning rate ensures fast convergence but can lead to overshooting.
- A small learning rate ensures greater accuracy but slows down the learning process.
- `efficientnet-b0` works best for images but takes longer to train than `alexnet` or `vgg19`.

### Pre-Trained Network

The `{arch}_model_checkpoint_{training_num}.pth` file contains a network trained to recognize 102 different species of flowers. To use this pre-trained model for prediction, run:

```bash
python predict.py /path/to/image checkpoint.pth
```

Note: The `{arch}_model_checkpoint_{training_num}.pth` file is not present in the repository.

### GPU Usage

Training deep learning models is computationally intensive and may not be feasible on a common laptop. If you have an NVIDIA GPU, you can install CUDA. Alternatively, you can use cloud services like AWS, Google Cloud, or Google Colab.

### Contributing

Contributions are welcome. Please open an issue to discuss your ideas or initiate a pull request with your changes.

### License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.