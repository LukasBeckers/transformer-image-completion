# Transformer Image Completion 
## Introduction
In this repository, I have implemented a transformer network from scratch using TensorFlow. The transformer model is a decoder-only transformer, trained autoregressively to predict the next image segment. The architecture of this model is mainly based on the groundbreaking paper 'Attention is all you need' (https://doi.org/10.48550/arXiv.1706.03762).

I chose the MNIST dataset for this project due to limited computational resources.

## Getting Started

The project was developed using Google Colab notebooks, which were connected to a Google Drive to store the trained model. To recreate the project, you will need to run the "Transformer_Model" notebook followed by the "Image_Completion" notebook. In both notebooks, the paths for saving or loading models will need to be updated to valid paths on your Google Drive. Note that training the model takes multiple hours, and it may be terminated before completion if using a non-pro Colab.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Model Architecture
### Model
I used a decoder-only transformer network with a model-dimension of 400, 4 heads, and 3 layers. To prevent v-ram overflow during training, I used a linear expansion of 2x model-dimension.

### Data Preprocessing
The MNIST images were split into 4x4 image-segments.

![image](https://user-images.githubusercontent.com/77121210/234255571-0be785fa-460a-4636-8e41-c7ce4f592884.png)

 These segments were then flattened to vectors of length 16 and expanded to the model-dimension by a learnable matrix multiplication. I also added positional embeddings to the expanded image-segments using the sin cosine positional embeddings method proposed in the "Attention is all you need" paper.

### Training
The model was trained for 223 epochs using the Adam optimizer, a learning rate of 10^-4, and a dropout probability of 20%. Mean squared error (MSE) was used as the loss function. However, the training was interrupted due to limited computational resources on Google Colab. The model was not trained until the validation loss converged.

## Results
The partially trained model was used to predict the last three rows of image-segments of 50 different MNIST samples. The predicted images were compared to the original images, and the results are shown below:

![image](https://user-images.githubusercontent.com/77121210/234398991-00f1e720-d926-411b-92d5-7fbb501cde1a.png)

Some samples were predicted well, while in others, the predicted image-segments appeared to be shifted by one image-segment. This could be attributed to the incomplete training of the model, which may have led to an incorrect interpretation of the positional embedding..




