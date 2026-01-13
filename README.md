Handwritten Digit Generation using C-VAE

Introduction

This project implements a Conditional Variational Autoencoder (C-VAE) to generate realistic handwritten digits. Unlike standard classifiers that simply recognize numbers, this Generative AI model learns the underlying probability distribution of the MNIST dataset to synthesize completely new digit samples.

The model is built from scratch using TensorFlow/Keras and utilizes the "Reparameterization Trick" to sample from a continuous latent space, allowing for smooth interpolation between different digit styles.

Features

Generative Capability: Synthesizes novel 28x28 pixel grayscale images of digits (0-9).

Controlled Generation: Uses a Conditional architecture, allowing users to specify exactly which digit to generate (e.g., forcing the model to generate a "5").

Latent Space Visualization: detailed 2D visualization of the learned manifold, showing how the model groups similar handwriting styles.

Custom Training Loop: Implements a custom train_step in Keras to handle the specific KL-Divergence and Reconstruction losses required for VAEs.

Tech Stack

Language: Python 3

Deep Learning Framework: TensorFlow / Keras

Data Manipulation: NumPy, Pandas

Visualization: Matplotlib, Graphviz (for architecture diagrams)

Environment: Google Colab / Jupyter Notebook

Dataset

The project uses the MNIST Digit Dataset, which consists of:

Training Set: 42,000 labeled images of handwritten digits.

Image Format: 28x28 grayscale pixels (flattened to 784 dimensions).

Source: LeCun et al. (Loaded via Keras or CSV).

Model Architecture

The C-VAE consists of two main parts:

Encoder: Compresses the input image and label into a latent distribution defined by a Mean ($\mu$) and Log-Variance ($\sigma$).

Decoder: Samples a point $z$ from this distribution (conditioned on the label) and reconstructs the original image.

Note: A detailed architecture diagram is generated within the notebook using Graphviz.

Installation & Usage

Option 1: Google Colab (Recommended)

Upload the .ipynb notebook file to your Google Drive.

Open it with Google Colab.

Change the runtime type to GPU for faster training.

Run all cells to train the model and see generated results.

Option 2: Local Execution

Clone this repository:

git clone [https://github.com/your-username/digit-generation-cvae.git](https://github.com/your-username/digit-generation-cvae.git)
cd digit-generation-cvae


Install dependencies:

pip install tensorflow numpy pandas matplotlib graphviz


Run the script or notebook:

jupyter notebook Digit_Generation.ipynb


Results

After training for 30+ epochs, the model is capable of:

Reconstruction: Accurately reproducing input images.

Generation: Creating a 10x10 grid of digits sampled from the latent space.


Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

Fork the project.

Create your feature branch (git checkout -b feature/AmazingFeature).

Commit your changes (git commit -m 'Add some AmazingFeature').

Push to the branch (git push origin feature/AmazingFeature).

Open a Pull Request.

License

This project is licensed under the MIT License - see the LICENSE file for details.

Author: Ayush Pritam
