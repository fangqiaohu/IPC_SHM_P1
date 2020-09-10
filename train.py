import argparse
import logging
import os

import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm

from unet import UNet

from torch.utils.tensorboard import SummaryWriter
from utils.dataset import BasicDataset
from torch.utils.data import DataLoader
from prefetch_generator import BackgroundGenerator

import time

dir_img_train = 'data/train/images_sub/'
dir_mask_train = 'data/train/labels_sub/'
dir_checkpoint = 'checkpoints/'


# DEBUG: accelerate data loading
class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


def train_net(net, device, epochs, batch_size, lr, lr_decay, img_scale, pos_weight):

    dataset_train = BasicDataset(dir_img_train, dir_mask_train, img_scale)

    n_train = len(dataset_train)

    train_loader = DataLoaderX(dataset_train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)

    writer = SummaryWriter(comment=f'LR_{lr}_BS_{batch_size}_SCALE_{img_scale}')
    global_step = 0

    optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight).to(device))
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=lr_decay)

    for epoch in range(epochs):
        net.train()
        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                imgs = batch['image']
                true_masks = batch['mask']
                imgs = imgs.to(device=device, dtype=torch.float32, non_blocking=True)
                mask_type = torch.float32 if net.n_classes == 1 else torch.long
                true_masks = true_masks.to(device=device, dtype=mask_type, non_blocking=True)

                masks_pred = net(imgs)
                loss = criterion(masks_pred, true_masks)
                epoch_loss += loss.item()
                writer.add_scalar('Loss/train', loss.item(), global_step)

                pbar.set_postfix(**{'loss (batch)': loss.item()})

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_value_(net.parameters(), 0.1)
                optimizer.step()

                pbar.update(imgs.shape[0])
                global_step += 1

        lr_scheduler.step()  # update the learning rate

        if not os.path.exists(dir_checkpoint):
            os.mkdir(dir_checkpoint)
        torch.save(net.state_dict(),
                   dir_checkpoint + f'CP_epoch{epoch + 1}.pth')

    writer.close()


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=10,
                        help='Number of epochs', dest='epochs')
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=16,
                        help='Batch size', dest='batchsize')
    parser.add_argument('-l', '--learning-rate', metavar='LR', type=float, nargs='?', default=0.001,
                        help='Learning rate', dest='lr')
    parser.add_argument('-d', '--learning-rate-decay-rate', type=float, nargs='?', default=0.5,
                        help='Learning rate decay rate', dest='lr_decay')
    parser.add_argument('-f', '--load', dest='load', type=str, default=False,
                        help='Load model from a .pth file')
    parser.add_argument('-s', '--scale', dest='scale', type=float, default=0.5,
                        help='Downscaling factor of the images')
    parser.add_argument('-w', '--positive-weight', dest='pos_weight', type=float, default=2,
                        help='Postive weight')

    return parser.parse_args()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    net = UNet(n_channels=3, n_classes=1, bilinear=True)

    if args.load:
        net.load_state_dict(
            torch.load(args.load, map_location=device)
        )
        logging.info(f'Model loaded from {args.load}')

    net.to(device=device)
    # faster convolutions, but more memory
    torch.backends.cudnn.benchmark = True

    train_net(net=net,
              epochs=args.epochs,
              batch_size=args.batchsize,
              lr=args.lr,
              lr_decay = args.lr_decay,
              device=device,
              img_scale=args.scale,
              pos_weight=args.pos_weight)


"""
Full model
weight = 2, epoch iou:
1  0.204836
2  0.236563
3  0.361155
4  0.408731
5  0.405541
6  0.391038
7  0.406958
8  0.401253
9  0.399447
10 0.400156
"""

"""
Baseline
weight = 2, epoch iou:
1  0 (0.5 thresh); 0.001181351 (otsu)
2  0.086528649 (0.5 thresh); 0.136570811 (otsu)
3  0.033477568 (0.5 thresh); 0.183179189 (otsu) 
4  0.237977692 (0.5 thresh)
5  0.278899211
6  0.291195946
7  0.306889744
8  0.315088974
9  0.31179725
10 0.310478378
"""

"""
Baseline + SASA
weight = 2, epoch iou:
1  0.05475
2  0.16154
3  0.21245
4  0.25796
5  0.29448
6  0.33743
7  0.34215
8  0.34178
9  0.33986
10 0.34009
"""


"""
Baseline + RED
weight = 2, epoch iou:
1  0.207746216; 0.138917568
2  0.238747105; 0.194217368
3  0.313589744; 0.31514
4  0.304346316; 0.322787105
5  0.33923475; 0.34974725
6  0.354031842; 0.362694211
7  0.357951282; 0.372628462
8  0.3572975; 0.37167425
9  0.363585; 0.3808565
10 0.3531; 0.360677949
"""
