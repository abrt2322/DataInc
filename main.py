from keras.preprocessing.image import ImageDataGenerator
import os
import cv2

DATA_DIR = './'  # データディレクトリ
IMAGE_NAME = 'sample.jpg'  # 対象画像ファイル
SAVE_DIR = os.path.join(DATA_DIR, 'preview')  # 生成画像の保存先ディレクトリ

datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

img_array = cv2.imread(os.path.join(DATA_DIR, IMAGE_NAME), )  # 画像読み込み
img_array = img_array.reshape((1,) + img_array.shape)  # 4次元データに変換（flow()に渡すため）

# 保存先ディレクトリが存在しない場合、作成する。
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

# flow()により、ランダム変換したイメージのバッチを作成。
# 指定したディレクトリに生成画像を保存する。
i = 0
for batch in datagen.flow(img_array, batch_size=1,
                          save_to_dir=SAVE_DIR, save_prefix='dog', save_format='jpeg'):
    i += 1
    if i == 10:
        break  # 停止しないと無限ループ
