# MNIST Digit Classifier

A handwritten digit classifier built with Keras, trained on the MNIST dataset, plus a small pipeline for testing it on photos of my own handwriting. I built it while working through Course 2 of Andrew Ng's Machine Learning Specialization.

## Results

| Set | Accuracy |
|---|---|
| Test | 98.11% |
| Train | 99.86% |

## Model

- Input: 28x28 images, flattened and scaled to 0 to 1
- Layers: Dense(128, ReLU), Dense(64, ReLU), Dense(10, linear)
- Loss: `SparseCategoricalCrossentropy(from_logits=True)`. The output layer is linear and softmax is applied inside the loss, which is more numerically stable.
- Optimizer: Adam, learning rate 0.001
- Training: 20 epochs

## How to run

Install the libraries:

```
pip install tensorflow numpy matplotlib pillow
```

Train the model (this also saves `mnist_digit_model.keras`):

```
python train.py
```

Test it on your own digit: put a photo in the project folder, check the filename inside `predict_custom.py`, then run:

```
python predict_custom.py
```

You can also skip training and use the included saved model with `keras.models.load_model("mnist_digit_model.keras")`.

## Using my own handwriting

MNIST digits are white on black, cropped tight, and centered in a 28x28 frame, so a raw photo does not work. The script:

1. Converts the photo to grayscale and inverts it (dark ink on white paper becomes bright ink on black).
2. Removes the paper background with a threshold.
3. Crops to the digit's bounding box.
4. Scales the longer side to 20 pixels, keeping proportions.
5. Centers it on a black 28x28 canvas and scales values to 0 to 1.

## Limitations

- My first attempts predicted the wrong digit because the whole photo was squashed to 28x28 and the digit became tiny.
- Taking a picture and cropping it to be near to the boundaries.
- Training accuracy is higher than test accuracy, which suggests mild overfitting. Results vary slightly between runs because of random initialization.
- The model is a plain dense network. It has no notion of 2D shape, so it is sensitive to stroke thickness, position, and style.
- Results depend on photo quality: lighting, shadows, and pen thickness all matter.

## Next steps

- Replace the dense network with a CNN.
- Build a version in NumPy from scratch once I've worked through backpropagation.
