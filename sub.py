from matplotlib import pyplot as plt


def show_imgs(imgs, row, col):
    if len(imgs) != (row * col):
        raise ValueError("Invalid imgs len:{} col:{} row{}:".format(len(imgs), row, col))

    for i, img in enumerate(imgs):
        plot_num = i + 1
        plt.subplot(row, col, plot_num)
        plt.tick_params(labalbottom="off")
        plt.tick_params(labelleft="off")
        plt.imshow(img)
    plt.show()
