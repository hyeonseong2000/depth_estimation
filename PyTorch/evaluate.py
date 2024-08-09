import argparse
import time
import os
from data import *
from utils import *
from model import DispNetS

def validate(val_loader, model, epoch, output_directory=""):
    average_meter = AverageMeter()
    model.eval()  # switch to evaluate mode
    end = time.time()

    eval_file = output_directory + 'evaluation.txt'
    f = open(eval_file, "w+")
    f.write("Max_Error  Depth   \r\n")
    for i, sample_batched in enumerate(val_loader):

        image = torch.autograd.Variable(sample_batched['image'].cuda())
        depth = torch.autograd.Variable(sample_batched['depth'].cuda(non_blocking=True))

        # Normalize depth
        depth_n = DepthNorm( depth )
        # torch.cuda.synchronize()
        data_time = time.time() - end

        # compute output
        end = time.time()
        with torch.no_grad():
            output = model(image)
            pred_depth = [1/disp for disp in output]
            pred = pred_depth[1]
        
        #print("image shape before", image.shape)
        # normalization for the model
        image = image[:, :, ::2, ::2]
        #depth = depth[:, :, ::2, ::2]
        #print("image shape after", image.shape)
        abs_err = (depth_n.data - pred.data).abs().cpu()
        max_err_ind = np.unravel_index(np.argmax(abs_err, axis=None), abs_err.shape)

        max_err_depth = depth_n.data[max_err_ind]
        max_err = abs_err[max_err_ind]
        f.write(f'{max_err}  {max_err_depth}   \r\n')

        # torch.cuda.synchronize()
        gpu_time = time.time() - end

        # measure accuracy and record loss
        result = Result()
        result.evaluate(pred.data, depth_n.data)
        average_meter.update2(result, gpu_time, data_time, image.size(0))
        end = time.time()

        # save 8 images for visualization
        skip = 50
        output_directory = os.path.join(os.path.abspath(os.path.dirname(__file__)), "results")

        if i == 0:
            # print(f'{image.shape} {depth_n.shape} {pred.shape}')
            img_merge = merge_into_row_with_gt(image, depth_n, pred, (depth_n - pred).abs())
        elif (i < 8 * skip) and (i % skip == 0):
            row = merge_into_row_with_gt(image, depth_n, pred, (depth_n - pred).abs())
            img_merge = add_row(img_merge, row)
        elif i == 8 * skip:
            filename = output_directory + '/comparison_' + str(epoch) + '.png'
            save_image(img_merge, filename)

        # if (i + 1) % skip == 0:
        #     print('Test: [{0}/{1}]\t'
        #           't_GPU={gpu_time:.3f}({average.gpu_time:.3f})\n\t'
        #           'RMSE={result.rmse:.2f}({average.rmse:.2f}) '
        #           'MAE={result.mae:.2f}({average.mae:.2f}) '
        #           'Delta1={result.delta1:.3f}({average.delta1:.3f}) '
        #           'Delta2={result.delta2:.3f}({average.delta2:.3f}) '
        #           'Delta3={result.delta3:.3f}({average.delta3:.3f}) '
        #           'REL={result.absrel:.3f}({average.absrel:.3f}) '
        #           'Lg10={result.lg10:.3f}({average.lg10:.3f}) '.format(
        #         i + 1, len(val_loader), gpu_time=gpu_time, result=result, average=average_meter.average()))
    f.close()
    avg = average_meter.average()

    print('\n*\n'
          'RMSE={average.rmse:.3f}\n'
          'MAE={average.mae:.3f}\n'
          'Delta1={average.delta1:.3f}\n'
          'Delta2={average.delta2:.3f}\n'
          'Delta3={average.delta3:.3f}\n'
          'REL={average.absrel:.3f}\n'
          'Lg10={average.lg10:.3f}\n'
          't_GPU={time:.3f}\n'.format(average=avg, time=avg.gpu_time))
    return avg, img_merge


def main():
    # Arguments
    parser = argparse.ArgumentParser(description='High Quality Monocular Depth Estimation via Transfer Learning')
    parser.add_argument('--path', default="checkpoint/ckpt_20.pth", type=str,
                        help='model path')
    parser.add_argument('--bs', default=1, type=int, help='batch size')

    parser.add_argument('--modality', '-m', metavar='MODALITY', default='rgb',
                        help='modality: ')
    parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',
                        help='number of data loading workers (default: 16)')
    args = parser.parse_args()

    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Data loading code
    print("=> creating data loaders...")

    val_loader = getEvaluateData(batch_size=args.bs)
    
    print("=> data loaders created.")
    
    assert os.path.isfile(args.path), "=> no model found at '{}'".format(args.path)
    print("=> loading model '{}'".format(args.path))
    checkpoint = torch.load(args.path)

    model = DispNetS()
    args.start_epoch = checkpoint["epoch"]
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    print("=> loaded best model (epoch {})".format(checkpoint['epoch']))
   
    output_directory = os.path.join(os.path.dirname(__file__), "results")
    validate(val_loader, model, args.start_epoch, output_directory)

    return


if __name__ == '__main__':
    main()
