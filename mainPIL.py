from keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator, array_to_img
import matplotlib.pyplot as plt
from keras_preprocessing.image import list_pictures
import os

#回転：-15~15
#上下平行移動:-0.8~1.2割の移動
#左右平行移動:-0.8~1.2割の移動
#せん断:-5度~5度でせん断
#拡大縮小:0.8~1.2割で拡大縮小
#明度変更:-5.0~5.0の範囲で画素値に値を足す
#各画素値に値を足す:0.3~1.0の範囲で値を変更する

DATA_DIR = './'
SAVE_DIR = os.path.join(DATA_DIR, 'A_D4')  # 生成画像の保存先ディレクトリ
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

# 画像をロード（PIL形式画像）
#img = load_img(IMAGE_FILE)

# 貼り付け
#plt.imshow(img)

# 表示
#plt.show()



datagen = ImageDataGenerator(
    rotation_range=15,
    height_shift_range=0.2,
    width_shift_range=0.2,
    shear_range=5,
    zoom_range=0.2,
    channel_shift_range=5,
    brightness_range=[0.3,1.0]
)

for picture in list_pictures('./D4/'):
    img = img_to_array(load_img(picture, target_size=(200,150)))

    # numpyの配列に変換
    x = img_to_array(img)

    # 4次元配列に変換
    # x = np.expand_dims(x, axis=0)
    x = x.reshape((1,) + x.shape)

    print(x.shape)

    g = datagen.flow(x, batch_size=1, save_to_dir=SAVE_DIR, save_prefix='sample', save_format='jpg')
    for i in range(16):
        batches = g.next()

        # （1,縦サイズ, 横サイズ, チャンネル数)
        #print(batches.shape)
        # 画像として表示するため、４次元から3次元データにし、配列から画像にする。
        gen_img = array_to_img(batches[0])

        #plt.subplot(8, 8, i + 1)
        #plt.imshow(gen_img)
        #plt.axis('off')


