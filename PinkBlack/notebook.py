import matplotlib.pyplot as plt


def show_multiple_images(imgs, num=(6,6), save_path=None, img_size=3):
    """
    :param imgs: [img1, img2, img3 ... ] imgs are displayable on plt
    :param num: (height, width)
    :param save_path:
    :return:
    """
    imgsize = img_size
    fig = plt.figure(figsize=(num[1]*imgsize, num[0]*imgsize))
    pos = 1
    for i in range(num[0]):
        for j in range(num[1]):
            if (pos) > len(imgs):
                break
            ax = fig.add_subplot(num[0], num[1], pos)
            ax.imshow(imgs[pos - 1])
            pos = pos + 1
            ax.axis("off")
    if save_path is not None:
        fig.savefig(save_path)
    plt.show()
    plt.close(fig)