import keras
import numpy as np
from keras.utils import np_utils
from matplotlib import pyplot as plt
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator

img = image.load_img("sample.jpg")
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
print(x.shape)

def show_imgs(imgs, row, col):
    if len(imgs) != (row * col):
        raise ValueError("Invalid imgs len:{} col:{} row{}:".format(len(imgs), row, col))

    for i, img in enumerate(imgs):
        plot_num = i+1
        plt.subplot(row,col,plot_num)
        plt.tick_params(labalbottom="off")
        plt.tick_params(labelleft="off")
        plt.imshow(img)
        plt.show()

    datagen = ImageDataGenerator(
    datagen
    )