import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.models import load_model
from PIL import Image, ImageOps

model = load_model("mnist_digit_model.keras")

img = Image.open("mnist_sample.jpg").convert("L")
img = ImageOps.invert(img)                       # ink bright & paper dark

# 1. zero out faint gray (paper noise); raised to 50 if shadows survive
img = img.point(lambda p: p if p > 50 else 0)

# 2. crop to the digit in pic
box = img.getbbox()
if box is None:
    raise ValueError("No digit found: use a darker pen or lower the 50")
img = img.crop(box)

# 3. scale the longer side to 20 px, keeping the proportions
w, h = img.size
scale = 20 / max(w, h)
new_w, new_h = max(1, round(w * scale)), max(1, round(h * scale))
img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

# 4. center it on a black 28x28 canvas
canvas = Image.new("L", (28, 28), 0)
canvas.paste(img, ((28 - new_w) // 2, (28 - new_h) // 2))

arr = np.array(canvas, dtype=float) / 255.0
arr = arr / arr.max()                           

plt.imshow(arr, cmap="gray")
plt.title("28x28 Input")
plt.show()

logits = model.predict(arr.reshape(1, 28, 28), verbose=0)
print("Predicted digit:", np.argmax(logits, axis=1)[0])
print(np.round(tf.nn.softmax(logits)[0].numpy(), 3))