import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing import image

img = image.load_img("sample.jpg")
img = np.array(img)

plt.imshow(img)
plt.show()

datagen = image.ImageDataGenerator(rotation_range=20)
x = img[np.newaxis]
gen = datagen.flow(x, batch_size=1)

plt.figure(figsize=(10, 8))
for i in range(9):
    batches = next(gen)
    gen_img = batches[0].astype(np.unit8)

        plt.subplot(3, 3, i+1)
    plt.imshow(gen_img)
    plt.axis("off")
plt.show()