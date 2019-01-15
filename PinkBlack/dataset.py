from PIL import Image
import torch.utils.data.Dataset

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


class ImageLabelDataset(torch.utils.data.Dataset):
    """
    class ImageLabelDataset:

    Initialize with list of image paths and its corresponding list of labels

    Example)
    >> image_paths = ['1.jpg', '2.png', '3.jpeg']
    >> image_labels = [1, 2, 3]
    >> dataset = ImageLabelDataset(image_paths, image_labels)
    >> dataset[0]


    """

    def devide(self, ratio=(0.8, 0.2), valid_transform=None, valid_label_transform=None):
        import random
        combined = list(zip(self.paths, self.labels))
        random.shuffle(combined)
        new_paths, new_labels = zip(*combined)

        num_train = int(ratio[0] * len(self.paths))

        train_paths = new_paths[:num_train]
        train_labels = new_labels[:num_train]

        valid_paths = new_paths[num_train:]
        valid_labels = new_labels[num_train:]

        train_set = ImageLabelDataset(train_paths,
                                      train_labels,
                                      transform=self.transform,
                                      label_transform=self.label_transform,
                                      loader=self.loader,
                                      label_loader=self.label_loader)

        valid_transform = self.transform if valid_transform is None else valid_transform
        valid_label_transform = self.label_transform if valid_label_transform is None else valid_label_transform

        valid_set = ImageLabelDataset(valid_paths,
                                      valid_labels,
                                      transform=valid_transform,
                                      label_transform=valid_label_transform,
                                      loader=self.loader,
                                      label_loader=self.label_loader)

        return train_set, valid_set


    def __init__(self, paths, labels, transform=None, label_transform=None, loader=pil_loader, label_loader=None):
        assert len(paths) == len(labels)
        self.paths = paths
        self.labels = labels
        self.transform = transform
        self.label_transform = label_transform
        self.loader = loader
        self.label_loader = label_loader

    def __getitem__(self, index):
        path, label = self.paths[index], self.labels[index]
        img = self.loader(path)
        if self.label_loader is not None:
            label = self.label_loader(label)
        if self.transform is not None:
            img = self.transform(img)
        if self.label_transform is not None:
            label = self.label_transform(label)
        return img, label

    def __len__(self):
        return len(self.paths)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str