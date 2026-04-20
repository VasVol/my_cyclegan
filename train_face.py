import argparse
from pathlib import Path
import random

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
import cv2

from model import (
    DataLoadersClass,
    DatasetsClass,
    get_transforms,
    CycleGAN,
    FullDiscriminatorLoss,
    FullGeneratorLoss,
    create_model_and_optimizer,
    model_num_params,
)


ALLOWED_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}


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


class ImagePathsDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        super().__init__()
        self.image_paths = list(image_paths)
        self.transform = transform

    def __getitem__(self, index):
        img_path = self.image_paths[index]
        image = cv2.imread(str(img_path))
        if image is None:
            raise ValueError(f'Failed to read image: {img_path}')
        image = image[:, :, ::-1].copy()
        if self.transform is not None:
            image = self.transform(image)
        return image

    def __len__(self):
        return len(self.image_paths)


def resolve_device(device_str=None):
    if device_str is not None:
        return torch.device(device_str)
    return torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def list_images(root: Path):
    return sorted(
        p for p in root.rglob('*')
        if p.is_file() and p.suffix.lower() in ALLOWED_EXTS
    )


def collect_sketch_images(dataset_root: Path):
    sketch_root = dataset_root / '_raw' / 'person_face_sketches'
    candidates = [
        sketch_root / 'train' / 'sketches',
        sketch_root / 'val' / 'sketches',
        sketch_root / 'test' / 'sketches',
    ]
    image_paths = []
    for folder in candidates:
        if folder.exists():
            image_paths.extend(list_images(folder))
    return sorted(image_paths)


def collect_cartoon_images(dataset_root: Path):
    cartoon_root = dataset_root / '_raw' / 'cartoon_faces' / 'cartoonset100k_jpg'
    if not cartoon_root.exists():
        raise FileNotFoundError(f'Cartoon root not found: {cartoon_root}')
    return list_images(cartoon_root)


def sample_equal_domains(sketch_paths, cartoon_paths, total_per_domain, train_count, test_count, seed):
    if train_count + test_count > total_per_domain:
        raise ValueError('train_count + test_count must be <= total_per_domain')

    rng = random.Random(seed)

    sketch_paths = list(sketch_paths)
    cartoon_paths = list(cartoon_paths)
    rng.shuffle(sketch_paths)
    rng.shuffle(cartoon_paths)

    if len(sketch_paths) < total_per_domain:
        raise RuntimeError(
            f'Not enough sketch images: found {len(sketch_paths)}, need {total_per_domain}'
        )
    if len(cartoon_paths) < total_per_domain:
        raise RuntimeError(
            f'Not enough cartoon images: found {len(cartoon_paths)}, need {total_per_domain}'
        )

    sketch_selected = sketch_paths[:total_per_domain]
    cartoon_selected = cartoon_paths[:total_per_domain]

    train_a = sketch_selected[:train_count]
    test_a = sketch_selected[train_count:train_count + test_count]
    train_b = cartoon_selected[:train_count]
    test_b = cartoon_selected[train_count:train_count + test_count]

    return train_a, test_a, train_b, test_b


def save_split_manifest(train_a, test_a, train_b, test_b, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    manifests = {
        'trainA.txt': train_a,
        'testA.txt': test_a,
        'trainB.txt': train_b,
        'testB.txt': test_b,
    }
    for name, items in manifests.items():
        with open(output_dir / name, 'w', encoding='utf-8') as f:
            for item in items:
                f.write(str(item) + '\n')


def load_split_manifest(manifest_dir: Path):
    manifests = {
        'train_a': manifest_dir / 'trainA.txt',
        'test_a': manifest_dir / 'testA.txt',
        'train_b': manifest_dir / 'trainB.txt',
        'test_b': manifest_dir / 'testB.txt',
    }
    if not all(path.exists() for path in manifests.values()):
        return None

    split_paths = {}
    for key, path in manifests.items():
        with open(path, 'r', encoding='utf-8') as f:
            split_paths[key] = [Path(line.strip()) for line in f if line.strip()]
    return split_paths


def build_dataloaders(dataset_root, image_size, resize_size, batch_size, num_workers, total_per_domain, train_count, test_count, seed, split_paths=None):
    if split_paths is None:
        sketch_paths = collect_sketch_images(Path(dataset_root))
        cartoon_paths = collect_cartoon_images(Path(dataset_root))

        train_a_paths, test_a_paths, train_b_paths, test_b_paths = sample_equal_domains(
            sketch_paths,
            cartoon_paths,
            total_per_domain=total_per_domain,
            train_count=train_count,
            test_count=test_count,
            seed=seed,
        )
        split_paths = {
            'train_a': train_a_paths,
            'test_a': test_a_paths,
            'train_b': train_b_paths,
            'test_b': test_b_paths,
        }
    else:
        for key, paths in split_paths.items():
            missing = [str(p) for p in paths if not Path(p).exists()]
            if missing:
                raise FileNotFoundError(
                    f'Missing files in manifest for {key}. First missing path: {missing[0]}'
                )

    train_transform_a, val_transform_a, de_normalize_a = get_transforms(None, None, image_size=image_size, resize_size=resize_size)
    train_transform_b, val_transform_b, de_normalize_b = get_transforms(None, None, image_size=image_size, resize_size=resize_size)

    ds = DatasetsClass(
        train_a=ImagePathsDataset(split_paths['train_a'], transform=train_transform_a),
        train_b=ImagePathsDataset(split_paths['train_b'], transform=train_transform_b),
        test_a=ImagePathsDataset(split_paths['test_a'], transform=val_transform_a),
        test_b=ImagePathsDataset(split_paths['test_b'], transform=val_transform_b),
    )

    pin_memory = torch.cuda.is_available()
    dataloaders = DataLoadersClass(
        train_a=DataLoader(ds.train_a, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=num_workers, pin_memory=pin_memory),
        train_b=DataLoader(ds.train_b, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=num_workers, pin_memory=pin_memory),
        test_a=DataLoader(ds.test_a, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=num_workers, pin_memory=pin_memory),
        test_b=DataLoader(ds.test_b, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=num_workers, pin_memory=pin_memory),
    )

    return dataloaders, de_normalize_a, de_normalize_b, split_paths


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
        axes[row_a, 0].set_title('A: original sketch')
        axes[row_a, 1].imshow(de_norm_b(fake_b[ind], normalized=True))
        axes[row_a, 1].set_title('A → B')
        axes[row_a, 2].imshow(de_norm_a(rec_a[ind], normalized=True))
        axes[row_a, 2].set_title('A → B → A')

        axes[row_b, 0].imshow(de_norm_b(imgs_b[ind], normalized=True))
        axes[row_b, 0].set_title('B: original cartoon')
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


def maybe_resume(args, model, optimizer_d, optimizer_g, scheduler_d, scheduler_g, default_resume_path: Path):
    resume_path = None
    if args.resume is not None:
        resume_path = Path(args.resume)
    elif args.auto_resume and default_resume_path.exists():
        resume_path = default_resume_path

    if resume_path is None:
        return 0, None, None

    if not resume_path.exists():
        raise FileNotFoundError(f'Resume checkpoint not found: {resume_path}')

    checkpoint = torch.load(resume_path, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer_d.load_state_dict(checkpoint['optimizer_d_state_dict'])
    optimizer_g.load_state_dict(checkpoint['optimizer_g_state_dict'])

    if scheduler_d is not None and checkpoint.get('scheduler_d_state_dict') is not None:
        scheduler_d.load_state_dict(checkpoint['scheduler_d_state_dict'])
    if scheduler_g is not None and checkpoint.get('scheduler_g_state_dict') is not None:
        scheduler_g.load_state_dict(checkpoint['scheduler_g_state_dict'])

    start_epoch = int(checkpoint.get('epoch', 0))
    plots = checkpoint.get('plots', None)
    checkpoint_args = checkpoint.get('args', None)

    print(f'Resumed from checkpoint: {resume_path}')
    print(f'Resumed epoch: {start_epoch}')
    return start_epoch, plots, checkpoint_args


def train(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    device = resolve_device(args.device)
    output_dir = Path(args.output_dir)
    checkpoints_dir = output_dir / 'checkpoints'
    dashboards_dir = output_dir / 'dashboards'
    translations_dir = output_dir / 'translations'
    manifests_dir = output_dir / 'splits'
    for d in [output_dir, checkpoints_dir, dashboards_dir, translations_dir, manifests_dir]:
        d.mkdir(parents=True, exist_ok=True)

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

    last_path = checkpoints_dir / 'last.pt'
    best_path = checkpoints_dir / 'best.pt'

    start_epoch, plots, checkpoint_args = maybe_resume(
        args,
        model,
        optimizer_d,
        optimizer_g,
        scheduler_d,
        scheduler_g,
        last_path,
    )

    split_paths = load_split_manifest(manifests_dir)
    if split_paths is None and checkpoint_args is not None:
        print('Split manifests not found, rebuilding splits from current arguments and seed.')

    dataloaders, de_normalize_a, de_normalize_b, split_paths = build_dataloaders(
        dataset_root=args.dataset_root,
        image_size=args.image_size,
        resize_size=args.resize_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        total_per_domain=args.total_per_domain,
        train_count=args.train_count,
        test_count=args.test_count,
        seed=args.seed,
        split_paths=split_paths,
    )
    save_split_manifest(
        split_paths['train_a'],
        split_paths['test_a'],
        split_paths['train_b'],
        split_paths['test_b'],
        manifests_dir,
    )

    print(f'trainA size: {len(dataloaders.train_a.dataset)}')
    print(f'trainB size: {len(dataloaders.train_b.dataset)}')
    print(f'testA size : {len(dataloaders.test_a.dataset)}')
    print(f'testB size : {len(dataloaders.test_b.dataset)}')

    criterion_d = FullDiscriminatorLoss(is_mse=True)
    criterion_g = FullGeneratorLoss(lambda_value=args.lambda_cycle, is_mse=True)

    model_num_params(model, verbose_all=False)

    fake_a_pool = ImagePool(args.image_pool_size)
    fake_b_pool = ImagePool(args.image_pool_size)

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
            'best val G': float('inf'),
        }
    else:
        plots.setdefault('best val G', min(plots.get('val G', [float('inf')])) if len(plots.get('val G', [])) > 0 else float('inf'))

    if start_epoch >= args.epochs:
        print(f'Checkpoint is already at epoch {start_epoch}, which is >= requested epochs={args.epochs}. Nothing to do.')
        print(f'Artifacts saved to: {output_dir.resolve()}')
        return

    for epoch in range(start_epoch + 1, args.epochs + 1):
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
    parser.add_argument('--dataset-root', type=str, default='./datasets/img2img/sketch2cartoon_kaggle')
    parser.add_argument('--output-dir', type=str, default='./runs/sketch2cartoon_faces')
    parser.add_argument('--total-per-domain', type=int, default=679)
    parser.add_argument('--train-count', type=int, default=600)
    parser.add_argument('--test-count', type=int, default=79)
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
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--auto-resume', action='store_true', default=True)
    parser.add_argument('--no-auto-resume', action='store_false', dest='auto_resume')
    return parser.parse_args()


if __name__ == '__main__':
    train(parse_args())
