import matplotlib.pyplot as plt
import torch

def show_multiple_images(imgs, num=(6,6), save_path=None, img_size=3):
    """
    :param imgs: [img1, img2, img3 ... ] imgs are displayable on plt
    or tensor batch images
    :param num: (height, width)
    :param save_path:
    :return:
    """
    if len(imgs) == 0:
        return

    fig = plt.figure(figsize=(num[1]*img_size, num[0]*img_size))
    pos = 1
    for i in range(num[0]):
        for j in range(num[1]):
            if (pos) > len(imgs):
                break
            ax = fig.add_subplot(num[0], num[1], pos)
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
    plt.show()
    plt.close(fig)
