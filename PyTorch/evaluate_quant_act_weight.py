import argparse
import time
import os
from data import *
from utils import *
from model import DispNetS, DispNetS_Q
import torch
import torch.nn as nn

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np



def validate(val_loader, model, epoch, exp, mant, type, output_directory=""):
    average_meter = AverageMeter()
    model.eval()  # switch to evaluate mode
    end = time.time()

    eval_file = output_directory + '/evaluation.txt'
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
            #print(f'{image.shape} {depth_n.shape} {pred.shape}')
            img_merge = merge_into_row_with_gt(image, depth_n, pred, (depth_n - pred).abs())
        elif (i < 8 * skip) and (i % skip == 0):
            row = merge_into_row_with_gt(image, depth_n, pred, (depth_n - pred).abs())
            img_merge = add_row(img_merge, row)
        elif i == 8 * skip:
            filename = output_directory + '/comparison_quant_act_weight_' + str(epoch) + '_exp_' + str(exp) + '_mant_' + str(mant) + '_' + str(type) + '.png'
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
    parser.add_argument('--dtype',      type=str    , default='fxp' ,             help='Choose Data Type to Quantize: "fxp" or "fp"')
    parser.add_argument('--exp',          type=int    , default=5    , help='Exponent/Integer Bit-width') 
    parser.add_argument('--mant',          type=int    , default = 10 , help = 'Mantissa/Fractional Bit-width')
    parser.add_argument('--mode', type=str    , default = 'trunc' , help = "Quantization Rule: 'trunc' or 'round' or 'stochastic")
    parser.add_argument('--wl', type=int    , default = '16' , help = "Word Length")

    args = parser.parse_args()
    exps = []
    mants = []
    rmses = []
    deltas = []
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Data loading code
    print("=> creating data loaders...")

    val_loader = getEvaluateData(batch_size=args.bs)
    
    print("=> data loaders created.")
    
    assert os.path.isfile(args.path), "=> no model found at '{}'".format(args.path)
    print("=> loading model '{}'".format(args.path))
    checkpoint = torch.load(args.path)

    

    for exp in range(1,args.exp + 1):
        for mant in range(1,args.wl - exp +1):
            model = DispNetS_Q(type = args.dtype, n_exp = exp, n_man =mant, mode = args.mode, device = "cuda" )
            args.start_epoch = checkpoint["epoch"]
            model.load_state_dict(checkpoint["model_state_dict"])
            model = model.to(device)
            print("=> loaded best model (epoch {})".format(checkpoint['epoch']))
        
            ## For weight Quantization!
            print("=> Proceed Weight Quantization")
            for _, child in model.named_children():
                

                if isinstance(child, nn.Sequential):
                    for _, sub_child in child.named_children():
                        if isinstance(sub_child, nn.Conv2d):
                            
                            sub_child_quant_weight = quantization(sub_child.weight.data, args.dtype, exp, mant, args.mode)
                            sub_child_quant_bias = quantization(sub_child.bias.data, args.dtype, exp, mant, args.mode)
                            sub_child.weight.data = sub_child_quant_weight.to(device)
                            sub_child.bias.data = sub_child_quant_bias.to(device)
                            
                        
                        if isinstance(sub_child, nn.ConvTranspose2d):
                            
                            sub_child_quant_weight = quantization(sub_child.weight.data, args.dtype, exp, mant, args.mode)
                            sub_child_quant_bias = quantization(sub_child.bias.data, args.dtype, exp, mant, args.mode)
                            sub_child.weight.data = sub_child_quant_weight.to(device)
                            sub_child.bias.data = sub_child_quant_bias.to(device)
                        

            print("=> Complete Weight Quantization")
            output_directory = os.path.join(os.path.dirname(__file__), "results")
            print("result of type:{} exp(int):{} mant(frac):{} mode:{}".format(args.dtype, exp, mant, args.mode))
            avgs, _ = validate(val_loader, model, args.start_epoch, exp, mant, args.dtype, output_directory)
            exps.append(exp)
            mants.append(mant)
            rmses.append(avgs.rmse)
            deltas.append(avgs.delta1)


    
    exps_nd = np.array(exps)
    mants_nd = np.array(mants)
    delats_nd = np.array(deltas)
    
    rmses_nd = np.array(rmses)
    rmses_nd = np.clip(rmses_nd, 0, 5000)
    np.nan_to_num(rmses_nd, copy=False)
    rmses = rmses_nd.tolist()

    
    print(exps)
    print(mants)
    print(rmses)
    print(deltas)
    # save result data with csv format
    import csv

    with open("./results/results_act_weight_{}_.csv".format(args.dtype),'w',newline='', encoding = 'utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(exps)
        writer.writerow(mants)
        writer.writerow(rmses)
        writer.writerow(deltas)

    # For plot 3D data graph
    fig1 = plt.figure(1)
    fig2 = plt.figure(2)
    ax1 = fig1.add_subplot(111, projection='3d')
    ax2 = fig2.add_subplot(111, projection='3d')
    fontlabel = {"fontsize":"large", "color":"black", "fontweight":"bold"}
    xlabel = "exponent" if args.dtype == "fp" else "integer"
    ylabel = "mantissa" if args.dtype == "fp" else "fraction"
    ax1.set_xlabel(xlabel, fontdict=fontlabel, labelpad=16)
    ax1.set_ylabel(ylabel, fontdict=fontlabel, labelpad=16)
    ax1.set_title("RMSE", fontdict=fontlabel)
    ax2.set_xlabel(xlabel, fontdict=fontlabel, labelpad=16)
    ax2.set_ylabel(ylabel, fontdict=fontlabel, labelpad=16)
    ax2.set_title("Delta1", fontdict=fontlabel)

    cmaps = plt.get_cmap("plasma")
    ax1.plot_trisurf(exps, mants, rmses, cmap=cmaps, alpha=0.8)
    ax2.plot_trisurf(exps, mants, deltas, cmap=cmaps, alpha=0.8)

    plt.show()



    
    return


if __name__ == '__main__':
    main()
