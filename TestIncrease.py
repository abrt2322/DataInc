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
    rotation_range=0,
    width_shift_range=0,
    height_shift_range=0,
    shear_range=0,
    zoom_range=0,
    horizontal_flip=False,
    vertical_flip=False
)

max_img_num = 9

imgs =[]
for d in datagen.flow(x, batch_size=1):
    imgs.append(image.array_to_img(d[0],scale=True))
    if (len(imgs)%max_img_num) ==0:
        break
show_imgs(imgs,row=3, col=3)