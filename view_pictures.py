# I downloaded data from data library as .gz file. However, when I unzipped it,
# I cant see the pictures, with the help of GPT, I am able to see it.

import numpy as np
import struct
import matplotlib.pyplot as plt


def read_idx(filename):
    with open(filename, 'rb') as f:
        zero, data_type, dims = struct.unpack('>HBB', f.read(4))
        shape = tuple(struct.unpack('>I', f.read(4))[0] for d in range(dims))
        return np.frombuffer(f.read(), dtype=np.uint8).reshape(shape)


# Load the images
images = read_idx('train_images/train-images-idx3-ubyte')

print(images.shape)  # Should output (10000, 28, 28) for the MNIST test set

# Visualize the first image
plt.imshow(images[0], cmap='gray')
plt.show()
