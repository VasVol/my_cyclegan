import argparse
import itertools
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from model import (
    ImageDatasetNoLabel,
    DatasetsClass,
    DataLoadersClass,
    get_transforms,
    CycleGAN,
    FullDiscriminatorLoss,
    FullGeneratorLoss,
    create_model_and_optimizer,
    model_num_params,
)


class ImagePool:
    def __init__(self, pool_size=50):
        self.pool_size = pool_size
        self.images = []

    def query(self, images):
        if self.pool_size <= 0:
            return images

        returned_images = []
        for image in images:
            image = image.detach().unsqueeze(0)
            if len(self.images) < self.pool_size:
                self.images.append(image)
                returned_images.append(image)
            else:
                if np.random.rand() > 0.5:
                    idx = np.random.randint(0, self.pool_size)
                    old_image = self.images[idx].clone()
                    self.images[idx] = image
                    returned_images.append(old_image)
                else:
                    returned_images.append(image)
        return torch.cat(returned_images, dim=0)


def resolve_device(device_str=None):
    if device_str is not None:
        return torch.device(device_str)
    return torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def take_images(loader, num_images, device):
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
        raise ValueError('Loader is empty')
    return torch.cat(batches, dim=0)


def summarize_patch_predictions(pred):
    if pred.ndim > 2:
        pred = pred.mean(dim=tuple(range(2, pred.ndim)))
    if pred.ndim == 2 and pred.shape[1] == 1:
        pred = pred[:, 0]
    return pred.detach().cpu().tolist()


def validate(model, loader_a, loader_b, criterion_d, criterion_g, device):
    model.eval()
    val_data = {
        'loss D': [],
        'loss G': [],
        'real pred A': [],
        'fake pred A': [],
        'real pred B': [],
        'fake pred B': [],
    }

    with torch.no_grad():
        iter_a = iter(loader_a)
        iter_b = iter(loader_b)
        batches_per_epoch = min(len(loader_a), len(loader_b))

        for _ in tqdm(range(batches_per_epoch), desc='validate', leave=False):
            imgs_a = next(iter_a).to(device)
            imgs_b = next(iter_b).to(device)

            fake_b = model.G_AB(imgs_a)
            rec_a = model.G_BA(fake_b)
            fake_a = model.G_BA(imgs_b)
            rec_b = model.G_AB(fake_a)

            real_pred_a = model.D_A(imgs_a)
            fake_pred_a = model.D_A(fake_a)
            real_pred_b = model.D_B(imgs_b)
            fake_pred_b = model.D_B(fake_b)

            loss_d_a = criterion_d(real_pred_a, fake_pred_a)
            loss_d_b = criterion_d(real_pred_b, fake_pred_b)
            loss_d = 0.5 * (loss_d_a + loss_d_b)

            loss_g = criterion_g(
                fake_pred_a,
                fake_pred_b,
                imgs_a,
                rec_a,
                imgs_b,
                rec_b,
            )

            val_data['loss D'].append(loss_d.item())
            val_data['loss G'].append(loss_g.item())
            val_data['real pred A'].extend(summarize_patch_predictions(real_pred_a))
            val_data['fake pred A'].extend(summarize_patch_predictions(fake_pred_a))
            val_data['real pred B'].extend(summarize_patch_predictions(real_pred_b))
            val_data['fake pred B'].extend(summarize_patch_predictions(fake_pred_b))

    val_data['loss D'] = float(np.mean(val_data['loss D']))
    val_data['loss G'] = float(np.mean(val_data['loss G']))
    return val_data


def save_dashboard(plots, epoch, out_path):
    val_epochs = np.arange(1, len(plots['val D']) + 1)

    fig, axes = plt.subplots(2, 2, figsize=(18, 10))
    fig.suptitle(f'Epoch {epoch}', fontsize=16)

    axes[0, 0].set_title('Discriminator losses')
    axes[0, 0].plot(np.arange(1, len(plots['train D']) + 1), plots['train D'], 'r.-', label='train', alpha=0.7)
    if len(plots['val D']) > 0:
        axes[0, 0].plot(val_epochs, plots['val D'], 'g.-', label='val', alpha=0.7)
    axes[0, 0].grid()
    axes[0, 0].legend()

    axes[0, 1].set_title('Generator losses')
    axes[0, 1].plot(np.arange(1, len(plots['train G']) + 1), plots['train G'], 'r.-', label='train', alpha=0.7)
    if len(plots['val G']) > 0:
        axes[0, 1].plot(val_epochs, plots['val G'], 'g.-', label='val', alpha=0.7)
    axes[0, 1].grid()
    axes[0, 1].legend()

    axes[1, 0].set_title('Discriminator A predictions')
    if len(plots['hist real A']) > 0:
        axes[1, 0].hist(plots['hist real A'][-1], bins=50, density=True, label='real', color='green', alpha=0.7)
        axes[1, 0].hist(plots['hist gen A'][-1], bins=50, density=True, label='generated', color='red', alpha=0.7)
    axes[1, 0].set_xlim((-0.25, 1.25))
    axes[1, 0].legend()

    axes[1, 1].set_title('Discriminator B predictions')
    if len(plots['hist real B']) > 0:
        axes[1, 1].hist(plots['hist real B'][-1], bins=50, density=True, label='real', color='green', alpha=0.7)
        axes[1, 1].hist(plots['hist gen B'][-1], bins=50, density=True, label='generated', color='red', alpha=0.7)
    axes[1, 1].set_xlim((-0.25, 1.25))
    axes[1, 1].legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def save_translation_grid(model, loader_a, loader_b, de_norm_a, de_norm_b, device, num_images, out_path):
    model.eval()
    with torch.no_grad():
        imgs_a = take_images(loader_a, num_images, device)
        imgs_b = take_images(loader_b, num_images, device)

        num_images = min(num_images, imgs_a.shape[0], imgs_b.shape[0])
        imgs_a = imgs_a[:num_images]
        imgs_b = imgs_b[:num_images]

        fake_b = model.G_AB(imgs_a)
        rec_a = model.G_BA(fake_b)
        fake_a = model.G_BA(imgs_b)
        rec_b = model.G_AB(fake_a)

    fig, axes = plt.subplots(num_images * 2, 3, figsize=(15, 5 * num_images))
    if num_images == 1:
        axes = np.expand_dims(axes, 0)
    fig.suptitle('CycleGAN translations', fontsize=16)

    for ind in range(num_images):
        row_a = ind * 2
        row_b = row_a + 1

        axes[row_a, 0].imshow(de_norm_a(imgs_a[ind], normalized=True))
        axes[row_a, 0].set_title('A: original')
        axes[row_a, 1].imshow(de_norm_b(fake_b[ind], normalized=True))
        axes[row_a, 1].set_title('A → B')
        axes[row_a, 2].imshow(de_norm_a(rec_a[ind], normalized=True))
        axes[row_a, 2].set_title('A → B → A')

        axes[row_b, 0].imshow(de_norm_b(imgs_b[ind], normalized=True))
        axes[row_b, 0].set_title('B: original')
        axes[row_b, 1].imshow(de_norm_a(fake_a[ind], normalized=True))
        axes[row_b, 1].set_title('B → A')
        axes[row_b, 2].imshow(de_norm_b(rec_b[ind], normalized=True))
        axes[row_b, 2].set_title('B → A → B')

        for col in range(3):
            axes[row_a, col].set_xticks([])
            axes[row_a, col].set_yticks([])
            axes[row_b, col].set_xticks([])
            axes[row_b, col].set_yticks([])

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def save_checkpoint(path, epoch, model, optimizer_d, optimizer_g, scheduler_d, scheduler_g, plots, args):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_d_state_dict': optimizer_d.state_dict(),
            'optimizer_g_state_dict': optimizer_g.state_dict(),
            'scheduler_d_state_dict': scheduler_d.state_dict() if scheduler_d is not None else None,
            'scheduler_g_state_dict': scheduler_g.state_dict() if scheduler_g is not None else None,
            'plots': plots,
            'args': vars(args),
        },
        path,
    )


def build_dataloaders(dataset_root, image_size, resize_size, batch_size, num_workers):
    train_transform_a, val_transform_a, de_normalize_a = get_transforms(None, None, image_size=image_size, resize_size=resize_size)
    train_transform_b, val_transform_b, de_normalize_b = get_transforms(None, None, image_size=image_size, resize_size=resize_size)

    ds = DatasetsClass(
        train_a=ImageDatasetNoLabel(Path(dataset_root) / 'trainA', transform=train_transform_a),
        train_b=ImageDatasetNoLabel(Path(dataset_root) / 'trainB', transform=train_transform_b),
        test_a=ImageDatasetNoLabel(Path(dataset_root) / 'testA', transform=val_transform_a),
        test_b=ImageDatasetNoLabel(Path(dataset_root) / 'testB', transform=val_transform_b),
    )

    dataloaders = DataLoadersClass(
        train_a=DataLoader(ds.train_a, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=num_workers, pin_memory=True),
        train_b=DataLoader(ds.train_b, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=num_workers, pin_memory=True),
        test_a=DataLoader(ds.test_a, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=num_workers, pin_memory=True),
        test_b=DataLoader(ds.test_b, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=num_workers, pin_memory=True),
    )
    return dataloaders, de_normalize_a, de_normalize_b


def train(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = resolve_device(args.device)
    output_dir = Path(args.output_dir)
    checkpoints_dir = output_dir / 'checkpoints'
    dashboards_dir = output_dir / 'dashboards'
    translations_dir = output_dir / 'translations'
    for d in [output_dir, checkpoints_dir, dashboards_dir, translations_dir]:
        d.mkdir(parents=True, exist_ok=True)

    dataloaders, de_normalize_a, de_normalize_b = build_dataloaders(
        dataset_root=args.dataset_root,
        image_size=args.image_size,
        resize_size=args.resize_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    model_params = dict(
        img_channels_a=3,
        img_channels_b=3,
        ngf=args.ngf,
        ndf=args.ndf,
        image_size=args.image_size,
        num_res_blocks=9 if args.image_size >= 256 else 6,
        init_gain=0.02,
    )
    model, optimizer_d, optimizer_g = create_model_and_optimizer(
        model_class=CycleGAN,
        model_params=model_params,
        lr=args.lr,
        betas=(0.5, 0.999),
        device=device,
    )

    def lr_lambda(epoch_idx):
        if epoch_idx < args.decay_start_epoch:
            return 1.0
        decay_epochs = max(1, args.epochs - args.decay_start_epoch)
        return max(0.0, 1.0 - (epoch_idx - args.decay_start_epoch) / decay_epochs)

    scheduler_d = torch.optim.lr_scheduler.LambdaLR(optimizer_d, lr_lambda=lr_lambda)
    scheduler_g = torch.optim.lr_scheduler.LambdaLR(optimizer_g, lr_lambda=lr_lambda)

    criterion_d = FullDiscriminatorLoss(is_mse=True)
    criterion_g = FullGeneratorLoss(lambda_value=args.lambda_cycle, is_mse=True)

    model_num_params(model, verbose_all=False)

    fake_a_pool = ImagePool(args.image_pool_size)
    fake_b_pool = ImagePool(args.image_pool_size)

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
        'best val G': float('inf'),
    }

    best_path = checkpoints_dir / 'best.pt'
    last_path = checkpoints_dir / 'last.pt'

    for epoch in range(1, args.epochs + 1):
        model.train()
        plots['lr G'].append(optimizer_g.param_groups[0]['lr'])
        plots['lr D'].append(optimizer_d.param_groups[0]['lr'])

        losses_g = []
        losses_d = []
        iter_a = iter(dataloaders.train_a)
        iter_b = iter(dataloaders.train_b)
        batches_per_epoch = min(len(dataloaders.train_a), len(dataloaders.train_b))

        progress = tqdm(range(batches_per_epoch), desc=f'epoch {epoch}/{args.epochs}')
        for _ in progress:
            imgs_a = next(iter_a).to(device, non_blocking=True)
            imgs_b = next(iter_b).to(device, non_blocking=True)

            model.set_requires_grad([model.D_A, model.D_B], False)
            optimizer_g.zero_grad(set_to_none=True)

            fake_b = model.G_AB(imgs_a)
            rec_a = model.G_BA(fake_b)
            fake_a = model.G_BA(imgs_b)
            rec_b = model.G_AB(fake_a)

            fake_pred_a = model.D_A(fake_a)
            fake_pred_b = model.D_B(fake_b)

            loss_g = criterion_g(fake_pred_a, fake_pred_b, imgs_a, rec_a, imgs_b, rec_b)
            loss_g.backward()
            optimizer_g.step()

            model.set_requires_grad([model.D_A, model.D_B], True)
            optimizer_d.zero_grad(set_to_none=True)

            fake_a_for_d = fake_a_pool.query(fake_a.detach())
            fake_b_for_d = fake_b_pool.query(fake_b.detach())

            real_pred_a = model.D_A(imgs_a)
            fake_pred_a_d = model.D_A(fake_a_for_d)
            real_pred_b = model.D_B(imgs_b)
            fake_pred_b_d = model.D_B(fake_b_for_d)

            loss_d_a = criterion_d(real_pred_a, fake_pred_a_d)
            loss_d_b = criterion_d(real_pred_b, fake_pred_b_d)
            loss_d = 0.5 * (loss_d_a + loss_d_b)
            loss_d.backward()
            optimizer_d.step()

            losses_g.append(loss_g.item())
            losses_d.append(loss_d.item())
            progress.set_postfix(loss_g=f'{loss_g.item():.4f}', loss_d=f'{loss_d.item():.4f}')

        train_g = float(np.mean(losses_g))
        train_d = float(np.mean(losses_d))
        plots['train G'].append(train_g)
        plots['train D'].append(train_d)

        val_data = validate(model, dataloaders.test_a, dataloaders.test_b, criterion_d, criterion_g, device)
        plots['val D'].append(val_data['loss D'])
        plots['val G'].append(val_data['loss G'])
        plots['hist real A'].append(val_data['real pred A'])
        plots['hist gen A'].append(val_data['fake pred A'])
        plots['hist real B'].append(val_data['real pred B'])
        plots['hist gen B'].append(val_data['fake pred B'])

        scheduler_d.step()
        scheduler_g.step()

        save_dashboard(plots, epoch, dashboards_dir / f'epoch_{epoch:03d}.png')
        save_translation_grid(
            model,
            dataloaders.test_a,
            dataloaders.test_b,
            de_normalize_a,
            de_normalize_b,
            device,
            args.images_per_validation,
            translations_dir / f'epoch_{epoch:03d}.png',
        )

        save_checkpoint(last_path, epoch, model, optimizer_d, optimizer_g, scheduler_d, scheduler_g, plots, args)
        if val_data['loss G'] < plots['best val G']:
            plots['best val G'] = val_data['loss G']
            save_checkpoint(best_path, epoch, model, optimizer_d, optimizer_g, scheduler_d, scheduler_g, plots, args)

        print(
            f'epoch {epoch:03d} | '
            f'train G: {train_g:.4f} | train D: {train_d:.4f} | '
            f'val G: {val_data["loss G"]:.4f} | val D: {val_data["loss D"]:.4f} | '
            f'lr: {optimizer_g.param_groups[0]["lr"]:.6f}'
        )

    print(f'Artifacts saved to: {output_dir.resolve()}')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-root', type=str, default='./datasets/img2img/horse2zebra')
    parser.add_argument('--output-dir', type=str, default='./runs/horse2zebra')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--decay-start-epoch', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--lambda-cycle', type=float, default=10.0)
    parser.add_argument('--image-size', type=int, default=256)
    parser.add_argument('--resize-size', type=int, default=286)
    parser.add_argument('--ngf', type=int, default=64)
    parser.add_argument('--ndf', type=int, default=64)
    parser.add_argument('--image-pool-size', type=int, default=50)
    parser.add_argument('--images-per-validation', type=int, default=3)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default=None)
    return parser.parse_args()


if __name__ == '__main__':
    train(parse_args())
