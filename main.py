import sub
from keras.preprocessing import image
import numpy as np
from keras.preprocessing.image import ImageDataGenerator

img = image.load_img("sample.jpg")
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
print(x.shape)

datagen = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    channel_shift_range=100
)

max_img_num = 16
imgs = []
for d in datagen.flow(x, batch_size=1):
    imgs.append(image.array_to_img(d[0], scale=True))
    if (len(imgs) % max_img_num) == 0:
        break
sub.show_imgs(imgs, row=4, col=4)
