# keras-segnet
SegNet model for Keras.

### The original articles
- SegNet: A Deep Convolutional Encoder-Decoder Architecture for Image Segmentation(https://arxiv.org/pdf/1511.00561v2.pdf)

## Examples

### Train images

```
import os
import glob
import numpy as np
import cv2
from segnet import SegNet, preprocess_input, to_categorical

input_shape = (360, 480, 3)
nb_classes = 2
nb_epoch = 100
batch_size = 4

X, y = load_train() # need to implement, y shape is (None, 360, 480, nb_classes)
X = preprocess_input(X)
Y = to_categorical(y, nb_classes)
model = SegNet(input_shape=input_shape, classes=nb_classes)
model.compile(loss="categorical_crossentropy", optimizer='adadelta', metrics=["accuracy"])
model.fit(X, Y, batch_size=batch_size, nb_epoch=nb_epoch)
```

