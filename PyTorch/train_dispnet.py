import time
import argparse
import datetime

import torch
import torch.nn as nn
import torch.nn.utils as utils
import torchvision.utils as vutils    
from tensorboardX import SummaryWriter

from model import PTModel, DispNetS
from loss import ssim
from data import getTrainingTestingData
from utils import AverageMeter, DepthNorm, colorize, init_or_load_model

def main():
    # Arguments
    parser = argparse.ArgumentParser(description='High Quality Monocular Depth Estimation via Transfer Learning')
    parser.add_argument('--epochs', default=10, type=int, help='number of total epochs to run')
    parser.add_argument('--lr', '--learning-rate', default=0.0001, type=float, help='initial learning rate')
    parser.add_argument('--bs', default=4, type=int, help='batch size')
    parser.add_argument(
        "--save", "-s", default="checkpoint/", type=str, help="location to save checkpoints in"
    )
    parser.add_argument(
        "--checkpoint",
        "-c",
        default="",
        type=str,
        help="path to last saved checkpoint to resume training from",
    )
    parser.add_argument(
        "--device",
        "-d",
        default="cuda",
        type=str,
        help="device to run training on. Use CUDA",
    )
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create model
    # loading from checkpoint if provided
    if len(args.checkpoint) > 0:
        print("Loading from checkpoint ...")

        model, optimizer, start_epoch = init_or_load_model(
            depthmodel=DispNetS,
            epochs=args.epochs,
            lr=args.lr,
            ckpt=args.checkpoint,
            device=device,
        )
        print("Resuming from: epoch #{}".format(start_epoch))

    else:
        print("Initializing fresh model ...")

        model, optimizer, start_epoch = init_or_load_model(
            depthmodel=DispNetS,
            epochs=args.epochs,
            lr=args.lr,
            ckpt=None,
            device=device,
        )

    print('Model created.')

    # Training parameters
    optimizer = torch.optim.Adam( model.parameters(), args.lr )
    batch_size = args.bs
    prefix = 'dispnet_' + str(batch_size)

    # Load data
    train_loader, test_loader = getTrainingTestingData(batch_size=batch_size)

    # Logging
    writer = SummaryWriter(comment='{}-lr{}-e{}-bs{}'.format(prefix, args.lr, args.epochs, args.bs), flush_secs=30)

    # Loss
    l1_criterion = nn.L1Loss()

    # Start training...
    for epoch in range(start_epoch, args.epochs+1):
        batch_time = AverageMeter()
        losses = AverageMeter()
        N = len(train_loader)

        # Switch to train mode
        model.train()

        end = time.time()

        for i, sample_batched in enumerate(train_loader):
            optimizer.zero_grad()

            # Prepare sample and target
            image = torch.autograd.Variable(sample_batched['image'].cuda())
            depth = torch.autograd.Variable(sample_batched['depth'].cuda(non_blocking=True))

            # Normalize depth
            depth_n = DepthNorm( depth )

            # Predict
            output = model(image)
            pred_depth = [1/disp for disp in output[0:3]]

            # Compute the loss
            l_depth = l1_criterion(pred_depth[1], depth_n)
            l_ssim = torch.clamp((1 - ssim(pred_depth[1], depth_n, val_range = 1000.0 / 10.0)) * 0.5, 0, 1)

            loss = (1.0 * l_ssim) + (0.1 * l_depth)

            # Update step
            losses.update(loss.data.item(), image.size(0))
            loss.backward()
            optimizer.step()

            # Measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            eta = str(datetime.timedelta(seconds=int(batch_time.val*(N - i))))
        
            # Log progress
            niter = epoch*N+i
            if i % 5 == 0:
                # Print to console
                print('Epoch: [{0}][{1}/{2}]\t'
                'Time {batch_time.val:.3f} ({batch_time.sum:.3f})\t'
                'ETA {eta}\t'
                'Loss {loss.val:.4f} ({loss.avg:.4f})'
                .format(epoch, i, N, batch_time=batch_time, loss=losses, eta=eta))

                # Log to tensorboard
                writer.add_scalar('Train/Loss', losses.val, niter)

            if i % 300 == 0:
                LogProgress(model, writer, test_loader, niter)
                

            if i % 1000 == 0:
                torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optim_state_dict": optimizer.state_dict(),
                    "loss": losses.avg,
                },
                args.save + "ckpt_{}.pth".format(epoch, int(losses.avg * 100)),
                )   
        
        # Record epoch's intermediate results
        LogProgress(model, writer, test_loader, niter)
        writer.add_scalar('Train/Loss.avg', losses.avg, epoch)

def LogProgress(model, writer, test_loader, epoch):
    model.eval()
    sequential = test_loader
    sample_batched = next(iter(sequential))
    image = torch.autograd.Variable(sample_batched['image'].cuda())
    depth = torch.autograd.Variable(sample_batched['depth'].cuda(non_blocking=True))
    if epoch == 0: writer.add_image('Train.1.Image', vutils.make_grid(image.data, nrow=6, normalize=True), epoch)
    if epoch == 0: writer.add_image('Train.2.Depth', colorize(vutils.make_grid(depth.data, nrow=6, normalize=False)), epoch)
    pred_disp = model(image)
    pred_depth = [1/disp for disp in pred_disp[0:3]]
    output = DepthNorm( pred_depth[1] )
    writer.add_image('Train.3.Ours', colorize(vutils.make_grid(output.data, nrow=6, normalize=False)), epoch)
    writer.add_image('Train.4.Diff', colorize(vutils.make_grid(torch.abs(output-depth).data, nrow=6, normalize=False)), epoch)
    del image
    del depth
    del output

if __name__ == '__main__':
    main()
