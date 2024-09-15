### Project Introduction: Fully Convolutional Siamese Networks for Change Detection

This project focuses on analyzing changes in satellite images using Fully Convolutional Neural Networks. The main goal is to detect changes between pairs of co-registered satellite images.

#### Project Components

1. **File `make_dataset.py`**:
   - This file is responsible for creating the dataset. It uses images and labels from specified folders to generate smaller images and saves them in `.npy` format.
   - The number of images created depends on the variable `IMAGE_NUMBER`, and the image size is related to `IMAGE_SIZE`.

2. **File `model.py`**:
   - This file defines the neural network model. It uses different classes to create convolutional layers and max-pooling layers.
   - The model includes convolutional layers and deconvolution layers to process pairs of input images.

3. **File `train.py`**:
   - This file is used to train the model. It loads data from specified folders, trains the model, and evaluates its performance.
   - In this version of the code, instead of using the SGD optimizer(which is used in the main repository), it uses `torch.optim.Adam` with a specified learning rate (`lr`), which can help improve the model's convergence speed.

#### How to Use
- **Create the Dataset**: To generate the dataset, run the following command:
  ```bash
  python make_dataset.py
  ```
- **Train the Model**: To train the model, use the following command:
  ```bash
  python train.py
  ```

#### Resources
- Related paper: [Fully Convolutional Siamese Networks for Change Detection](https://arxiv.org/abs/1810.08462)
- Dataset used: [Onera Satellite Change Detection Dataset](https://rcdaudt.github.io/oscd/)

This project serves as a useful tool for detecting changes in satellite images and can assist research in environmental monitoring and infrastructure development.
