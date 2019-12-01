from keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator, array_to_img
import matplotlib.pyplot as plt
import os

DATA_DIR = './'  # データディレクトリ
IMAGE_FILE = 'sample.jpg'  # 対象画像ファイル
SAVE_DIR = os.path.join(DATA_DIR, 'preview3')  # 生成画像の保存先ディレクトリ


if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

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
g = datagen.flow(x, batch_size=1, save_to_dir=SAVE_DIR, save_prefix='sample', save_format='jpg')
for i in range(16):
    batches = g.next()

    # （1,縦サイズ, 横サイズ, チャンネル数)
    print(batches.shape)
    # 画像として表示するため、４次元から3次元データにし、配列から画像にする。
    gen_img = array_to_img(batches[0])

    plt.subplot(4, 4, i + 1)
    plt.imshow(gen_img)
    plt.axis('off')

plt.show()
