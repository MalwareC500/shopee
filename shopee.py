import torch
import torch.nn.functional as F
from homura import optim, lr_scheduler, callbacks, reporters
from homura.trainers import SupervisedTrainer as Trainer
from homura.vision.data.loaders import cifar10_loaders
from torch.utils.data import DataLoader, random_split, SubsetRandomSampler
from torchvision import transforms
from torchvision.datasets import ImageFolder

from senet.se_resnet import se_resnet101, se_resnet152, se_resnet50
from senet.se_inception import se_inception_v3

import numpy as np


def main():
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])

    valid_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        normalize
    ])

    train_dataset = ImageFolder(args.data, train_transform)
    valid_dataset = ImageFolder(args.data, valid_transform)

    num_samples = int(len(train_dataset) / 10)
    indices = list(range(num_samples))
    split = int(np.floor(0.1 * num_samples))
    np.random.shuffle(indices)
    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)
    train_loader = DataLoader(train_dataset, args.batch_size, sampler=train_sampler, num_workers=4)
    valid_loader = DataLoader(valid_dataset, args.batch_size, sampler=valid_sampler, num_workers=4)
    print("num data:", num_samples)
    print("num train batches:", len(train_loader))
    print("num test batches:", len(valid_loader))
    # return

    # train_loader, test_loader = cifar10_loaders(args.batch_size)

    model = se_resnet50(num_classes=42)
    # model.load_state_dict(torch.load("seresnet50-60a8950a85b2b.pkl"))
    optimizer = optim.SGD(lr=1e-1, momentum=0.9, weight_decay=1e-4)
    scheduler = lr_scheduler.StepLR(80, 0.1)
    tqdm_rep = reporters.TQDMReporter(
        range(args.epochs), callbacks.AccuracyCallback())
    _callbacks = [tqdm_rep, callbacks.AccuracyCallback()]
    with Trainer(model, optimizer, F.cross_entropy, scheduler=scheduler, callbacks=_callbacks) as trainer:
        for _ in tqdm_rep:
            trainer.train(train_loader)
            trainer.test(valid_loader)
            torch.save(trainer.model.state_dict(), "se_resnet50.pkl")


if __name__ == '__main__':
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--reduction", type=int, default=8)
    p.add_argument("--baseline", action="store_true")
    p.add_argument("--data", type=str, default='shopee')

    args = p.parse_args()
    main()
