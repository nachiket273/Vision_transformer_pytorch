import os
from shutil import copyfile
import torch
from torchvision import datasets
from torchvision import transforms


def create_val_dir_struct(data_dir):
    if not os.path.exists(data_dir):
        raise OSError("Directory doesn't exist")

    val_dir = os.path.join(data_dir, 'val')
    img_dir = os.path.join(val_dir, 'images')
    tgt_dir = os.path.join(val_dir, 'tgt')
    annot_txt = os.path.join(val_dir, 'val_annotations.txt')

    f = open(annot_txt, 'r')
    data = f.readlines()
    img_dict = dict()

    for line in data:
        words = line.split('\t')
        img_dict[words[0]] = words[1]
    f.close()

    if not os.path.exists(tgt_dir):
        os.makedirs(tgt_dir)

    for key, val in img_dict.items():
        new_dir = os.path.join(tgt_dir, val)
        if not os.path.exists(new_dir):
            os.makedirs(new_dir)
        img_path = os.path.join(img_dir, key)
        tgt_path = os.path.join(new_dir, key)
        if os.path.exists(img_path) and not os.path.exists(tgt_path):
            copyfile(img_path, tgt_path)


def id_to_cls(data_dir):
    if not os.path.exists(data_dir):
        raise OSError("Directory doesn't exist")

    wtxt = os.path.join(data_dir, 'words.txt')
    f = open(wtxt, 'r')
    lines = f.readlines()
    id_to_class = dict()
    for line in lines:
        words = line.split('\t')
        name = '/'.join(words[1].split(','))
        id_to_class[words[0]] = name.strip()
    f.close()
    return id_to_class


def get_ds_loader(data_dir, train_bs=16, test_bs=8):
    if not os.path.exists(data_dir):
        raise OSError("Directory doesn't exist")

    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')
    tgt_dir = os.path.join(val_dir, 'tgt')

    norm = transforms.Normalize(mean=[0.4802, 0.4481, 0.3975],
                                std=[0.2302, 0.2265, 0.2262])

    train_tf = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        norm
    ])

    val_tf = transforms.Compose([
        transforms.ToTensor(),
        norm
    ])

    train_ds = datasets.ImageFolder(train_dir, transform=train_tf)
    val_ds = datasets.ImageFolder(tgt_dir, transform=val_tf)

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=train_bs,
                                               shuffle=True, num_workers=1,
                                               pin_memory=True)

    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=test_bs,
                                             shuffle=False, num_workers=1,
                                             pin_memory=True)

    return train_loader, val_loader
