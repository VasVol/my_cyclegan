import itertools
import os
import random
import warnings
from collections import defaultdict
from dataclasses import dataclass

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as tr
from IPython.display import clear_output
from termcolor import colored
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import trange


@dataclass
class DatasetsClass:
    train_a: Dataset
    train_b: Dataset
    test_a: Dataset
    test_b: Dataset


@dataclass
class DataLoadersClass:
    train_a: DataLoader
    train_b: DataLoader
    test_a: DataLoader
    test_b: DataLoader


class ImageDatasetNoLabel(Dataset):
    def __init__(self, root, transform=None, transforms=None, extensions=(".jpg", ".jpeg", ".png", ".bmp", ".webp")):
        super().__init__()
        self.root = root
        self.transform = transform if transform is not None else transforms
        self.extensions = tuple(ext.lower() for ext in extensions)
        self.img_paths = sorted(
            [
                os.path.join(root, fname)
                for fname in os.listdir(root)
                if os.path.isfile(os.path.join(root, fname))
                and os.path.splitext(fname)[1].lower() in self.extensions
            ]
        )

    def __getitem__(self, index):
        img_path = self.img_paths[index]
        image = cv2.imread(img_path)
        if image is None:
            raise ValueError(f"Failed to read image: {img_path}")
        image = image[:, :, ::-1].copy()

        if self.transform is not None:
            image = self.transform(image)

        return image

    def __len__(self):
        return len(self.img_paths)


def get_channel_statistics(dataset):
    channel_sum = np.zeros(3, dtype=np.float64)
    channel_sum_sq = np.zeros(3, dtype=np.float64)
    total_pixels = 0

    for i in range(len(dataset)):
        image = dataset[i].astype(np.float32) / 255.0
        image = image.reshape(-1, 3)

        channel_sum += image.sum(axis=0)
        channel_sum_sq += (image ** 2).sum(axis=0)
        total_pixels += image.shape[0]

    channel_mean = channel_sum / total_pixels
    channel_std = np.sqrt(channel_sum_sq / total_pixels - channel_mean ** 2)
    return channel_mean, channel_std


def get_transforms(mean=None, std=None, image_size=256, resize_size=286):
    train_transform = tr.Compose([
        tr.ToPILImage(),
        tr.Resize((resize_size, resize_size)),
        tr.RandomCrop((image_size, image_size)),
        tr.RandomHorizontalFlip(p=0.5),
        tr.ToTensor(),
        tr.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    val_transform = tr.Compose([
        tr.ToPILImage(),
        tr.Resize((image_size, image_size)),
        tr.ToTensor(),
        tr.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    def de_normalize(image, normalized=True):
        image = image.detach().cpu().clone()

        if normalized:
            image = image * 0.5 + 0.5

        image = image.clamp(0, 1)
        image = image.permute(1, 2, 0).numpy()
        return image

    return train_transform, val_transform, de_normalize


def show_examples(dataset, transform, de_norm, num_per_image=3, image_index=0, title=""):
    fig, ax = plt.subplots(1, 1 + num_per_image, figsize=(5 * (1 + num_per_image), 5))

    image = dataset[image_index]

    plt.suptitle(title, y=0.95)

    plt.subplot(1, 1 + num_per_image, 1)
    plt.imshow(image)
    plt.title("original")

    for i in range(num_per_image):
        plt.subplot(1, 1 + num_per_image, i + 2)
        plt.title(f"#{i}")
        plt.imshow(de_norm(transform(image)))
    plt.show()


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=0, bias=False),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=0, bias=False),
            nn.InstanceNorm2d(channels),
        )

    def forward(self, x):
        return x + self.block(x)


class ResNetGenerator(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, ngf=64, num_res_blocks=9):
        super().__init__()

        layers = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels, ngf, kernel_size=7, stride=1, padding=0, bias=False),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(inplace=True),
            nn.Conv2d(ngf, ngf * 2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(ngf * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(ngf * 2, ngf * 4, kernel_size=3, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(ngf * 4),
            nn.ReLU(inplace=True),
        ]

        for _ in range(num_res_blocks):
            layers.append(ResidualBlock(ngf * 4))

        layers += [
            nn.ConvTranspose2d(ngf * 4, ngf * 2, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            nn.InstanceNorm2d(ngf * 2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(ngf * 2, ngf, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, out_channels, kernel_size=7, stride=1, padding=0),
            nn.Tanh(),
        ]

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class PatchGANDiscriminator(nn.Module):
    def __init__(self, in_channels=3, ndf=64):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, ndf, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 4, ndf * 8, kernel_size=4, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=1, padding=1),
        )

    def forward(self, x):
        return self.model(x)


class ImagePool:
    def __init__(self, pool_size=50):
        self.pool_size = pool_size
        self.images = []

    def query(self, images):
        if self.pool_size <= 0:
            return images

        returned_images = []
        for image in images.detach():
            image = image.unsqueeze(0)
            if len(self.images) < self.pool_size:
                self.images.append(image.clone())
                returned_images.append(image)
            else:
                if random.random() > 0.5:
                    idx = random.randint(0, self.pool_size - 1)
                    old_image = self.images[idx].clone()
                    self.images[idx] = image.clone()
                    returned_images.append(old_image)
                else:
                    returned_images.append(image)
        return torch.cat(returned_images, dim=0)


class CycleGAN(nn.Module):
    def __init__(
        self,
        img_channels_a=3,
        img_channels_b=3,
        ngf=64,
        ndf=64,
        image_size=256,
        num_res_blocks=None,
        init_gain=0.02,
    ):
        super().__init__()

        if num_res_blocks is None:
            num_res_blocks = 6 if image_size <= 128 else 9

        self.G_AB = ResNetGenerator(
            in_channels=img_channels_a,
            out_channels=img_channels_b,
            ngf=ngf,
            num_res_blocks=num_res_blocks,
        )
        self.G_BA = ResNetGenerator(
            in_channels=img_channels_b,
            out_channels=img_channels_a,
            ngf=ngf,
            num_res_blocks=num_res_blocks,
        )
        self.D_A = PatchGANDiscriminator(in_channels=img_channels_a, ndf=ndf)
        self.D_B = PatchGANDiscriminator(in_channels=img_channels_b, ndf=ndf)

        self.apply(lambda module: init_weights(module, init_gain=init_gain))

    def forward(self, real_a=None, real_b=None):
        outputs = {}

        if real_a is not None:
            fake_b = self.G_AB(real_a)
            rec_a = self.G_BA(fake_b)
            idt_a = self.G_BA(real_a)

            outputs["real_a"] = real_a
            outputs["fake_b"] = fake_b
            outputs["rec_a"] = rec_a
            outputs["idt_a"] = idt_a

        if real_b is not None:
            fake_a = self.G_BA(real_b)
            rec_b = self.G_AB(fake_a)
            idt_b = self.G_AB(real_b)

            outputs["real_b"] = real_b
            outputs["fake_a"] = fake_a
            outputs["rec_b"] = rec_b
            outputs["idt_b"] = idt_b

        if len(outputs) == 0:
            raise ValueError("At least one of real_a or real_b must be provided.")

        return outputs

    def translate_a_to_b(self, x):
        return self.G_AB(x)

    def translate_b_to_a(self, x):
        return self.G_BA(x)

    @staticmethod
    def set_requires_grad(nets, requires_grad):
        if not isinstance(nets, (list, tuple)):
            nets = [nets]

        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad


class CycleConsistencyLoss(nn.Module):
    def __init__(self, reduction="mean"):
        super().__init__()
        self.loss = nn.L1Loss(reduction=reduction)

    def forward(self, x, x_rec):
        return self.loss(x_rec, x)


class AdversarialLossCE(nn.Module):
    def __init__(self, real_label=1.0, fake_label=0.0, reduction="mean"):
        super().__init__()
        self.real_label = real_label
        self.fake_label = fake_label
        self.loss = nn.BCEWithLogitsLoss(reduction=reduction)

    def forward(self, real_pred, fake_pred=None):
        if fake_pred is None:
            real_target = torch.full_like(real_pred, self.real_label)
            return self.loss(real_pred, real_target)

        real_target = torch.full_like(real_pred, self.real_label)
        fake_target = torch.full_like(fake_pred, self.fake_label)

        real_loss = self.loss(real_pred, real_target)
        fake_loss = self.loss(fake_pred, fake_target)
        return 0.5 * (real_loss + fake_loss)


class AdversarialLossMSE(nn.Module):
    def __init__(self, real_label=1.0, fake_label=0.0, reduction="mean"):
        super().__init__()
        self.real_label = real_label
        self.fake_label = fake_label
        self.loss = nn.MSELoss(reduction=reduction)

    def forward(self, real_pred, fake_pred=None):
        if fake_pred is None:
            real_target = torch.full_like(real_pred, self.real_label)
            return self.loss(real_pred, real_target)

        real_target = torch.full_like(real_pred, self.real_label)
        fake_target = torch.full_like(fake_pred, self.fake_label)

        real_loss = self.loss(real_pred, real_target)
        fake_loss = self.loss(fake_pred, fake_target)
        return 0.5 * (real_loss + fake_loss)


class FullDiscriminatorLoss(nn.Module):
    def __init__(self, is_mse=True, real_label=1.0, fake_label=0.0, reduction="mean"):
        super().__init__()
        self.adversarial_loss_func = AdversarialLossMSE(
            real_label=real_label,
            fake_label=fake_label,
            reduction=reduction,
        ) if is_mse else AdversarialLossCE(
            real_label=real_label,
            fake_label=fake_label,
            reduction=reduction,
        )

    def forward(self, real_pred, fake_pred):
        return self.adversarial_loss_func(real_pred, fake_pred)


class FullGeneratorLoss(nn.Module):
    def __init__(self, lambda_value=10.0, is_mse=True, real_label=1.0, fake_label=0.0, reduction="mean"):
        super().__init__()
        self.adversarial_loss_func = AdversarialLossMSE(
            real_label=real_label,
            fake_label=fake_label,
            reduction=reduction,
        ) if is_mse else AdversarialLossCE(
            real_label=real_label,
            fake_label=fake_label,
            reduction=reduction,
        )
        self.cycle_consistency_loss_func = CycleConsistencyLoss(reduction=reduction)
        self.lambda_value = lambda_value

    def forward(self, fake_pred_a, fake_pred_b, real_a, rec_a, real_b, rec_b):
        loss_gan_a = self.adversarial_loss_func(fake_pred_a)
        loss_gan_b = self.adversarial_loss_func(fake_pred_b)
        loss_cycle_a = self.cycle_consistency_loss_func(real_a, rec_a)
        loss_cycle_b = self.cycle_consistency_loss_func(real_b, rec_b)
        return loss_gan_a + loss_gan_b + self.lambda_value * (loss_cycle_a + loss_cycle_b)


def init_weights(module, init_gain=0.02):
    if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(module.weight.data, 0.0, init_gain)
        if module.bias is not None:
            nn.init.zeros_(module.bias.data)
    elif isinstance(module, nn.InstanceNorm2d):
        if module.weight is not None:
            nn.init.normal_(module.weight.data, 1.0, init_gain)
        if module.bias is not None:
            nn.init.zeros_(module.bias.data)


def _resolve_device(device, model=None):
    if device is not None:
        return device
    if model is not None:
        return next(model.parameters()).device
    return torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


def _reduce_patch_predictions(pred):
    if pred.ndim > 2:
        reduce_dims = tuple(range(2, pred.ndim))
        pred = pred.mean(dim=reduce_dims)
    return pred


def train_discriminators(model, opt_d, loader_a, loader_b, criterion_d, device=None):
    model.train()
    losses_tr = []
    device = _resolve_device(device, model)

    model.set_requires_grad([model.D_A, model.D_B], True)
    model.set_requires_grad([model.G_AB, model.G_BA], False)

    iter_a = iter(loader_a)
    iter_b = iter(loader_b)
    batches_per_epoch = min(len(loader_a), len(loader_b))

    for _ in trange(batches_per_epoch):
        imgs_a = next(iter_a).to(device)
        imgs_b = next(iter_b).to(device)

        opt_d.zero_grad()

        fake_b = model.G_AB(imgs_a).detach()
        fake_a = model.G_BA(imgs_b).detach()

        real_pred_a = model.D_A(imgs_a)
        fake_pred_a = model.D_A(fake_a)
        real_pred_b = model.D_B(imgs_b)
        fake_pred_b = model.D_B(fake_b)

        loss_d_a = criterion_d(real_pred_a, fake_pred_a)
        loss_d_b = criterion_d(real_pred_b, fake_pred_b)
        loss = loss_d_a + loss_d_b

        loss.backward()
        opt_d.step()
        losses_tr.append(loss.item())

    return model, opt_d, np.mean(losses_tr)


def train_generators(model, opt_g, loader_a, loader_b, criterion_g, device=None):
    model.train()
    losses_tr = []
    device = _resolve_device(device, model)

    model.set_requires_grad([model.D_A, model.D_B], False)
    model.set_requires_grad([model.G_AB, model.G_BA], True)

    iter_a = iter(loader_a)
    iter_b = iter(loader_b)
    batches_per_epoch = min(len(loader_a), len(loader_b))

    for _ in trange(batches_per_epoch):
        imgs_a = next(iter_a).to(device)
        imgs_b = next(iter_b).to(device)

        opt_g.zero_grad()

        fake_b = model.G_AB(imgs_a)
        rec_a = model.G_BA(fake_b)
        fake_a = model.G_BA(imgs_b)
        rec_b = model.G_AB(fake_a)

        fake_pred_a = model.D_A(fake_a)
        fake_pred_b = model.D_B(fake_b)

        loss = criterion_g(
            fake_pred_a,
            fake_pred_b,
            imgs_a,
            rec_a,
            imgs_b,
            rec_b,
        )

        loss.backward()
        opt_g.step()
        losses_tr.append(loss.item())

    return model, opt_g, np.mean(losses_tr)


def train_cyclegan_epoch(
    model,
    optimizer_g,
    g_iters_per_epoch,
    optimizer_d,
    d_iters_per_epoch,
    loader_a,
    loader_b,
    criterion_d,
    criterion_g,
    fake_a_pool,
    fake_b_pool,
    device=None,
):
    model.train()
    device = _resolve_device(device, model)

    losses_d = []
    losses_g = []

    iter_a = iter(loader_a)
    iter_b = iter(loader_b)
    batches_per_epoch = min(len(loader_a), len(loader_b))

    for _ in trange(batches_per_epoch):
        imgs_a = next(iter_a).to(device)
        imgs_b = next(iter_b).to(device)

        for _ in range(max(1, g_iters_per_epoch)):
            model.set_requires_grad([model.D_A, model.D_B], False)
            model.set_requires_grad([model.G_AB, model.G_BA], True)

            optimizer_g.zero_grad()

            fake_b = model.G_AB(imgs_a)
            rec_a = model.G_BA(fake_b)
            fake_a = model.G_BA(imgs_b)
            rec_b = model.G_AB(fake_a)

            fake_pred_a = model.D_A(fake_a)
            fake_pred_b = model.D_B(fake_b)

            loss_g = criterion_g(
                fake_pred_a,
                fake_pred_b,
                imgs_a,
                rec_a,
                imgs_b,
                rec_b,
            )
            loss_g.backward()
            optimizer_g.step()
            losses_g.append(loss_g.item())

        for _ in range(max(1, d_iters_per_epoch)):
            model.set_requires_grad([model.D_A, model.D_B], True)
            model.set_requires_grad([model.G_AB, model.G_BA], False)

            optimizer_d.zero_grad()

            with torch.no_grad():
                fake_b = model.G_AB(imgs_a)
                fake_a = model.G_BA(imgs_b)

            fake_a_buffered = fake_a_pool.query(fake_a)
            fake_b_buffered = fake_b_pool.query(fake_b)

            real_pred_a = model.D_A(imgs_a)
            fake_pred_a = model.D_A(fake_a_buffered)
            real_pred_b = model.D_B(imgs_b)
            fake_pred_b = model.D_B(fake_b_buffered)

            loss_d_a = criterion_d(real_pred_a, fake_pred_a)
            loss_d_b = criterion_d(real_pred_b, fake_pred_b)
            loss_d = loss_d_a + loss_d_b
            loss_d.backward()
            optimizer_d.step()
            losses_d.append(loss_d.item())

    return model, optimizer_d, optimizer_g, float(np.mean(losses_d)), float(np.mean(losses_g))


def val(model, loader_a, loader_b, criterion_d, criterion_g, device=None):
    model.eval()
    device = _resolve_device(device, model)
    val_data = defaultdict(list)

    with torch.no_grad():
        iter_a = iter(loader_a)
        iter_b = iter(loader_b)
        batches_per_epoch = min(len(loader_a), len(loader_b))

        for _ in trange(batches_per_epoch):
            imgs_a = next(iter_a).to(device)
            imgs_b = next(iter_b).to(device)

            fake_b = model.G_AB(imgs_a)
            rec_a = model.G_BA(fake_b)
            fake_a = model.G_BA(imgs_b)
            rec_b = model.G_AB(fake_a)

            a_real_pred = model.D_A(imgs_a)
            a_fake_pred = model.D_A(fake_a)
            b_real_pred = model.D_B(imgs_b)
            b_fake_pred = model.D_B(fake_b)

            loss_d_a = criterion_d(a_real_pred, a_fake_pred)
            loss_d_b = criterion_d(b_real_pred, b_fake_pred)
            loss_d = loss_d_a + loss_d_b

            loss_g = criterion_g(
                a_fake_pred,
                b_fake_pred,
                imgs_a,
                rec_a,
                imgs_b,
                rec_b,
            )

            val_data["loss D"].append(loss_d.item())
            val_data["loss G"].append(loss_g.item())

            a_real_pred = _reduce_patch_predictions(a_real_pred)
            b_real_pred = _reduce_patch_predictions(b_real_pred)
            a_fake_pred = _reduce_patch_predictions(a_fake_pred)
            b_fake_pred = _reduce_patch_predictions(b_fake_pred)

            is_mse_pred = a_real_pred.shape[-1] == 1

            if is_mse_pred:
                a_real_pred = a_real_pred[:, 0]
                b_real_pred = b_real_pred[:, 0]
                a_fake_pred = a_fake_pred[:, 0]
                b_fake_pred = b_fake_pred[:, 0]
            else:
                a_real_pred = F.softmax(a_real_pred, dim=1)[:, 1]
                b_real_pred = F.softmax(b_real_pred, dim=1)[:, 1]
                a_fake_pred = F.softmax(a_fake_pred, dim=1)[:, 1]
                b_fake_pred = F.softmax(b_fake_pred, dim=1)[:, 1]

            val_data["real pred A"].extend(a_real_pred.cpu().detach().tolist())
            val_data["real pred B"].extend(b_real_pred.cpu().detach().tolist())
            val_data["fake pred A"].extend(a_fake_pred.cpu().detach().tolist())
            val_data["fake pred B"].extend(b_fake_pred.cpu().detach().tolist())

        val_data["loss D"] = np.mean(val_data["loss D"])
        val_data["loss G"] = np.mean(val_data["loss G"])

    return val_data


def _take_images_from_loader(loader, num_images, device):
    batches = []
    total = 0

    for batch in loader:
        batch = batch.to(device)
        need = min(num_images - total, batch.shape[0])
        batches.append(batch[:need])
        total += need
        if total >= num_images:
            break

    if total == 0:
        raise ValueError("Loader is empty")

    return torch.cat(batches, dim=0)


def draw_imgs(model, num_images, loader_a, loader_b, de_norm_a, de_norm_b, device):
    model.eval()
    with torch.no_grad():
        imgs_a = _take_images_from_loader(loader_a, num_images, device)
        imgs_b = _take_images_from_loader(loader_b, num_images, device)

        num_images = min(num_images, imgs_a.shape[0], imgs_b.shape[0])

        fake_a = model.G_BA(imgs_b)
        fake_b = model.G_AB(imgs_a)
        rec_a = model.G_BA(fake_b)
        rec_b = model.G_AB(fake_a)

        fig, ax = plt.subplots(num_images, 3, figsize=(25, 5 * num_images))
        plt.suptitle("Images from A", y=0.92)

        for ind in range(num_images):
            plt.subplot(num_images, 3, ind * 3 + 1)
            plt.title("Original from A")
            plt.imshow(de_norm_a(imgs_a[ind], normalized=True))
            plt.xticks([])
            plt.yticks([])

            plt.subplot(num_images, 3, ind * 3 + 2)
            plt.title("Translated to B")
            plt.imshow(de_norm_b(fake_b[ind], normalized=True))
            plt.xticks([])
            plt.yticks([])

            plt.subplot(num_images, 3, ind * 3 + 3)
            plt.title("Reconstructed A")
            plt.imshow(de_norm_a(rec_a[ind], normalized=True))
            plt.xticks([])
            plt.yticks([])

        fig, ax = plt.subplots(num_images, 3, figsize=(25, 5 * num_images))
        plt.suptitle("Images from B", y=0.92)

        for ind in range(num_images):
            plt.subplot(num_images, 3, ind * 3 + 1)
            plt.title("Original from B")
            plt.imshow(de_norm_b(imgs_b[ind], normalized=True))
            plt.xticks([])
            plt.yticks([])

            plt.subplot(num_images, 3, ind * 3 + 2)
            plt.title("Translated to A")
            plt.imshow(de_norm_a(fake_a[ind], normalized=True))
            plt.xticks([])
            plt.yticks([])

            plt.subplot(num_images, 3, ind * 3 + 3)
            plt.title("Reconstructed B")
            plt.imshow(de_norm_b(rec_b[ind], normalized=True))
            plt.xticks([])
            plt.yticks([])

        plt.show()


def beautiful_int(i):
    i = str(i)
    return ".".join(reversed([i[max(j, 0):j+3] for j in range(len(i) - 3, -3, -3)]))


def model_num_params(model, verbose_all=True, verbose_only_learnable=False):
    sum_params = 0
    sum_learnable_params = 0
    submodules = defaultdict(lambda: [0, 0])
    for name, param in model.named_parameters():
        num_params = param.numel()
        if verbose_all or (verbose_only_learnable and param.requires_grad):
            print(
                colored(
                    '{: <65} ~  {: <9} params ~ grad: {}'.format(
                        name,
                        beautiful_int(num_params),
                        param.requires_grad,
                    ),
                    {True: "green", False: "red"}[param.requires_grad],
                )
            )
        sum_params += num_params
        sm = name.split(".")[0]
        submodules[sm][0] += num_params
        if param.requires_grad:
            sum_learnable_params += num_params
            submodules[sm][1] += num_params
    print(
        f'\nIn total:\n  - {beautiful_int(sum_params)} params\n  - {beautiful_int(sum_learnable_params)} learnable params'
    )

    for sm, v in submodules.items():
        print(
            f"\n . {sm}:\n .   - {beautiful_int(submodules[sm][0])} params\n .   - {beautiful_int(submodules[sm][1])} learnable params"
        )
    return sum_params, sum_learnable_params


def get_model_name(chkp_folder, model_name=None):
    if model_name is None:
        if os.path.exists(chkp_folder):
            num_starts = len(os.listdir(chkp_folder)) + 1
        else:
            num_starts = 1
        model_name = f'model#{num_starts}'
    else:
        if "#" not in model_name:
            model_name += "#0"
    changed = False
    while os.path.exists(os.path.join(chkp_folder, model_name + '.pt')):
        model_name, ind = model_name.split("#")
        model_name += f"#{int(ind) + 1}"
        changed = True
    if changed:
        warnings.warn(f"Selected model_name was used already! To avoid possible overwrite - model_name changed to {model_name}")
    return model_name


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def create_model_and_optimizer(
    model_class,
    model_params,
    lr=2e-4,
    betas=(0.5, 0.999),
    weight_decay=0.0,
    device=None,
):
    device = _resolve_device(device)
    model = model_class(**model_params).to(device)

    optimizer_d = torch.optim.Adam(
        itertools.chain(
            model.D_A.parameters(),
            model.D_B.parameters(),
        ),
        lr=lr,
        betas=betas,
        weight_decay=weight_decay,
    )
    optimizer_g = torch.optim.Adam(
        itertools.chain(
            model.G_AB.parameters(),
            model.G_BA.parameters(),
        ),
        lr=lr,
        betas=betas,
        weight_decay=weight_decay,
    )
    return model, optimizer_d, optimizer_g


def _step_scheduler(scheduler, metric=None):
    if scheduler is None:
        return
    try:
        if metric is None:
            scheduler.step()
        else:
            scheduler.step(metric)
    except TypeError:
        scheduler.step()


def learning_loop(
    model,
    optimizer_g,
    g_iters_per_epoch,
    optimizer_d,
    d_iters_per_epoch,
    train_loader_a,
    train_loader_b,
    val_loader_a,
    val_loader_b,
    criterion_d,
    criterion_g,
    de_norm_a,
    de_norm_b,
    scheduler_d=None,
    scheduler_g=None,
    min_lr=None,
    epochs=10,
    val_every=1,
    draw_every=1,
    model_name=None,
    chkp_folder="./chkps",
    images_per_validation=3,
    plots=None,
    starting_epoch=0,
    device=None,
    image_pool_size=50,
):
    device = _resolve_device(device, model)
    model_name = get_model_name(chkp_folder, model_name)

    if plots is None:
        plots = {
            'train G': [],
            'train D': [],
            'val D': [],
            'val G': [],
            'lr G': [],
            'lr D': [],
            'hist real A': [],
            'hist gen A': [],
            'hist real B': [],
            'hist gen B': [],
            'best val G': np.inf,
        }

    val_epochs = list(range(val_every, val_every * len(plots['val D']) + 1, val_every))
    fake_a_pool = ImagePool(pool_size=image_pool_size)
    fake_b_pool = ImagePool(pool_size=image_pool_size)

    for epoch in np.arange(1, epochs + 1) + starting_epoch:
        print(f'#{epoch}/{epochs}:')

        plots['lr G'].append(get_lr(optimizer_g))
        plots['lr D'].append(get_lr(optimizer_d))

        print(
            f'train CycleGAN epoch '
            f'(generator updates per batch = {g_iters_per_epoch}, discriminator updates per batch = {d_iters_per_epoch})'
        )
        model, optimizer_d, optimizer_g, loss_d, loss_g = train_cyclegan_epoch(
            model=model,
            optimizer_g=optimizer_g,
            g_iters_per_epoch=g_iters_per_epoch,
            optimizer_d=optimizer_d,
            d_iters_per_epoch=d_iters_per_epoch,
            loader_a=train_loader_a,
            loader_b=train_loader_b,
            criterion_d=criterion_d,
            criterion_g=criterion_g,
            fake_a_pool=fake_a_pool,
            fake_b_pool=fake_b_pool,
            device=device,
        )
        plots['train D'].append(loss_d)
        plots['train G'].append(loss_g)

        val_loss_g_for_scheduler = None
        if not (epoch % val_every):
            print('validate')
            val_data = val(model, val_loader_a, val_loader_b, criterion_d, criterion_g, device=device)
            plots['val D'].append(val_data['loss D'])
            plots['val G'].append(val_data['loss G'])
            plots['hist real A'].append(val_data['real pred A'])
            plots['hist gen A'].append(val_data['fake pred A'])
            plots['hist real B'].append(val_data['real pred B'])
            plots['hist gen B'].append(val_data['fake pred B'])
            val_epochs.append(epoch)
            val_loss_g_for_scheduler = val_data['loss G']

            if val_data['loss G'] < plots['best val G']:
                plots['best val G'] = val_data['loss G']

                if not os.path.exists(chkp_folder):
                    os.makedirs(chkp_folder)

                torch.save(
                    {
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_d_state_dict': optimizer_d.state_dict(),
                        'optimizer_g_state_dict': optimizer_g.state_dict(),
                        'scheduler_d_state_dict': scheduler_d.state_dict() if scheduler_d is not None else None,
                        'scheduler_g_state_dict': scheduler_g.state_dict() if scheduler_g is not None else None,
                        'plots': plots,
                    },
                    os.path.join(chkp_folder, model_name + '.pt'),
                )

        _step_scheduler(scheduler_d, metric=val_loss_g_for_scheduler)
        _step_scheduler(scheduler_g, metric=val_loss_g_for_scheduler)

        if not (epoch % draw_every):
            clear_output(True)

            hh = 2
            ww = 2
            plt_ind = 1
            fig, ax = plt.subplots(hh, ww, figsize=(25, 12))
            fig.suptitle(f'#{epoch}/{epochs}:')

            plt.subplot(hh, ww, plt_ind)
            plt.title('discriminators losses')
            plt.plot(np.arange(1, len(plots['train D']) + 1), plots['train D'], 'r.-', label='train', alpha=0.7)
            if len(plots['val D']) > 0:
                plt.plot(val_epochs, plots['val D'], 'g.-', label='val', alpha=0.7)
            plt.grid()
            plt.legend()
            plt_ind += 1

            plt.subplot(hh, ww, plt_ind)
            plt.title('generators losses')
            plt.plot(np.arange(1, len(plots['train G']) + 1), plots['train G'], 'r.-', label='train', alpha=0.7)
            if len(plots['val G']) > 0:
                plt.plot(val_epochs, plots['val G'], 'g.-', label='val', alpha=0.7)
            plt.grid()
            plt.legend()
            plt_ind += 1

            plt.subplot(hh, ww, plt_ind)
            plt.title('Discriminator A predictions')
            if len(plots['hist real A']) > 0:
                plt.hist(plots['hist real A'][-1], bins=50, density=True, label='real', color='green', alpha=0.7)
                plt.hist(plots['hist gen A'][-1], bins=50, density=True, label='generated', color='red', alpha=0.7)
            plt.xlim((-0.05, 1.05))
            plt.xticks(ticks=np.arange(0, 1.05, 0.1))
            plt.legend()
            plt_ind += 1

            plt.subplot(hh, ww, plt_ind)
            plt.title('Discriminator B predictions')
            if len(plots['hist real B']) > 0:
                plt.hist(plots['hist real B'][-1], bins=50, density=True, label='real', color='green', alpha=0.7)
                plt.hist(plots['hist gen B'][-1], bins=50, density=True, label='generated', color='red', alpha=0.7)
            plt.xlim((-0.05, 1.05))
            plt.xticks(ticks=np.arange(0, 1.05, 0.1))
            plt.legend()
            plt_ind += 1

            plt.show()

            if len(plots['val D']) > 0:
                draw_imgs(model, images_per_validation, val_loader_a, val_loader_b, de_norm_a, de_norm_b, device)

        if min_lr and get_lr(optimizer_d) <= min_lr:
            print(f'Learning process ended with early stop for discriminator after epoch {epoch}')
            break

        if min_lr and get_lr(optimizer_g) <= min_lr:
            print(f'Learning process ended with early stop for generator after epoch {epoch}')
            break

    return model, optimizer_d, optimizer_g, plots


__all__ = [
    'DatasetsClass',
    'DataLoadersClass',
    'ImageDatasetNoLabel',
    'get_channel_statistics',
    'get_transforms',
    'show_examples',
    'ResidualBlock',
    'ResNetGenerator',
    'PatchGANDiscriminator',
    'ImagePool',
    'CycleGAN',
    'CycleConsistencyLoss',
    'AdversarialLossCE',
    'AdversarialLossMSE',
    'FullDiscriminatorLoss',
    'FullGeneratorLoss',
    'train_discriminators',
    'train_generators',
    'train_cyclegan_epoch',
    'val',
    'draw_imgs',
    'beautiful_int',
    'model_num_params',
    'get_model_name',
    'get_lr',
    'create_model_and_optimizer',
    'learning_loop',
]
