from keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator, array_to_img
import matplotlib.pyplot as plt

IMAGE_FILE = "./sample.jpg"

# 画像をロード（PIL形式画像）
img = load_img(IMAGE_FILE)

# 貼り付け
plt.imshow(img)

# 表示
plt.show()

# numpyの配列に変換
x = img_to_array(img)

# 4次元配列に変換、以下同じ意味
# x = np.expand_dims(x, axis=0)
x = x.reshape((1,) + x.shape)

# （1,縦サイズ, 横サイズ, チャンネル数)
print(x.shape)

# -90 - 90の範囲でランダムに回転
datagen = ImageDataGenerator(rotation_range=90)

# generatorから9個の画像を生成
# 今回は1枚のみなのでbatch_sizeは1
g = datagen.flow(x, batch_size=1)
for i in range(4):
    batches = g.next()

    # （1,縦サイズ, 横サイズ, チャンネル数)
    print(batches.shape)

    # 画像として表示するため、４次元から3次元データにし、配列から画像にする。
    gen_img = array_to_img(batches[0])

    plt.subplot(2, 2, i + 1)
    plt.imshow(gen_img)
    plt.axis('off')

plt.show()
