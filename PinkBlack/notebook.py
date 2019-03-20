import matplotlib.pyplot as plt
import torch

def show_multiple_images(imgs, labels=[], num=(6,6), save_path=None, img_size=3, display=True):
    """
    :param imgs: [img1, img2, img3 ... ] imgs are displayable on plt
    or tensor batch images
    :param labels: [label1, label2, .. ]
    :param num: (height, width)
    :param save_path:
    :return:
    """
    if len(imgs) == 0:
        return

    if len(labels) > 0 and len(imgs) != len(labels):
        print("PinkBlack: The number of label does not match the number of images")
        labels = []

    fig = plt.figure(figsize=(num[1]*img_size, num[0]*img_size))
    pos = 1
    for i in range(num[0]):
        for j in range(num[1]):
            if (pos) > len(imgs):
                break
            ax = fig.add_subplot(num[0], num[1], pos)
            if len(labels) > 0:
                ax.set_title(labels[pos - 1])
            if torch.is_tensor(imgs[pos - 1]):
                if imgs[pos - 1].shape[0] <= 3:
                    plot_img = imgs[pos-1].cpu().permute(1,2,0).numpy().clip(0, 1)
                else:
                    plot_img = imgs[pos - 1].cpu().numpy()
            else:
                plot_img = imgs[pos - 1]
            ax.imshow(plot_img)
            pos = pos + 1
            ax.axis("off")
    if save_path is not None:
        fig.savefig(save_path)
    if display:
        plt.show()
    plt.close(fig)
