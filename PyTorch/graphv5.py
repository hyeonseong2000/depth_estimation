from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib as mpl
import numpy as np
import math
import csv

def main():
        
    # exps_load = []
    # mants_load = []
    # rmses_load = []
    # deltas_load = []
    with open("./results/results_act_weight_full_fp.csv",'r',newline='') as f:
        reader = csv.reader(f)
        for i,line in enumerate(reader):
            if i==0:
                exps_en_load = line
            elif i==1 :
                mants_en_load = line
            elif i==2:
                exps_de_load = line
            elif i==3 :
                mants_de_load = line
            elif i==4 :
                rmses_load = line
            elif i==5 :
                deltas_load = line
    
    exps_en_load = [float(item) for item in exps_en_load]
    mants_en_load = [float(item) for item in mants_en_load]
    exps_de_load = [float(item) for item in exps_de_load]
    mants_de_load = [float(item) for item in mants_de_load]
    rmses_load = [float(item) for item in rmses_load]
    deltas_load = [float(item) for item in deltas_load]

    wl_en = np.array(exps_en_load) + np.array(mants_en_load) 
    wl_de = np.array(exps_de_load) + np.array(mants_de_load)

    wl_en_load = wl_en.tolist()
    wl_de_load = wl_de.tolist()
    
    en_wl = [2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]

    rmse_de_2 = [rmses_load[16*i+0] for i in range(0,15)]
    rmse_de_3 = [rmses_load[16*i+1] for i in range(0,15)]
    rmse_de_4 = [rmses_load[16*i+2] for i in range(0,15)]
    rmse_de_5 = [rmses_load[16*i+3] for i in range(0,15)]
    rmse_de_6 = [rmses_load[16*i+4] for i in range(0,15)]
    rmse_de_7 = [rmses_load[16*i+5] for i in range(0,15)]
    rmse_de_8 = [rmses_load[16*i+6] for i in range(0,15)]
    rmse_de_9 = [rmses_load[16*i+7] for i in range(0,15)]
    rmse_de_10 = [rmses_load[16*i+8] for i in range(0,15)]
    rmse_de_11 = [rmses_load[16*i+9] for i in range(0,15)]
    rmse_de_12 = [rmses_load[16*i+10] for i in range(0,15)]
    rmse_de_13 = [rmses_load[16*i+11] for i in range(0,15)]
    rmse_de_14 = [rmses_load[16*i+12] for i in range(0,15)]
    rmse_de_15 = [rmses_load[16*i+13] for i in range(0,15)]
    rmse_de_16 = [rmses_load[16*i+14] for i in range(0,15)]

    delta_de_2 = [deltas_load[16*i+0] for i in range(0,15)]
    delta_de_3 = [deltas_load[16*i+1] for i in range(0,15)]
    delta_de_4 = [deltas_load[16*i+2] for i in range(0,15)]
    delta_de_5 = [deltas_load[16*i+3] for i in range(0,15)]
    delta_de_6 = [deltas_load[16*i+4] for i in range(0,15)]
    delta_de_7 = [deltas_load[16*i+5] for i in range(0,15)]
    delta_de_8 = [deltas_load[16*i+6] for i in range(0,15)]
    delta_de_9 = [deltas_load[16*i+7] for i in range(0,15)]
    delta_de_10 = [deltas_load[16*i+8] for i in range(0,15)]
    delta_de_11 = [deltas_load[16*i+9] for i in range(0,15)]
    delta_de_12 = [deltas_load[16*i+10] for i in range(0,15)]
    delta_de_13 = [deltas_load[16*i+11] for i in range(0,15)]
    delta_de_14 = [deltas_load[16*i+12] for i in range(0,15)]
    delta_de_15 = [deltas_load[16*i+13] for i in range(0,15)]
    delta_de_16 = [deltas_load[16*i+14] for i in range(0,15)]
    
    noquant_rmse = [285.6, 285.6,285.6,285.6, 285.6, 285.6,285.6,285.6,285.6, 285.6,285.6,285.6,285.6, 285.6,285.6 ]
    noquant_delta = [0.982, 0.982,0.982,0.982, 0.982, 0.982,0.982,0.982,0.982, 0.982,0.982,0.982,0.982, 0.982,0.982 ]


    # For plot 3D data graph
    
    norm1 = mpl.colors.Normalize(vmin=0, vmax=5000)
    norm2 = mpl.colors.Normalize(vmin=0, vmax=1)
    cmaps = plt.get_cmap("plasma")
    fig1 = plt.figure(1)
    fig2 = plt.figure(2)

    ax1 = fig1.add_subplot(111)
    ax2 = fig2.add_subplot(111)
 
    
    fontlabel = {"fontsize":"large", "color":"black", "fontweight":"bold"}
    xlabel = "Encoder_WL" 
    ylabel = "RMSE"
    
    
    ax1.set_xlabel(xlabel, fontdict=fontlabel, labelpad=16)
    ax1.set_ylabel("RMSE", fontdict=fontlabel, labelpad=16)
    ax1.set_title("RMSE with word length", fontdict=fontlabel)
    ax1.plot(en_wl, rmse_de_2, label = 'decoder_wl_2', marker='o'  ,ls='-', markersize=5)
    ax1.plot(en_wl, rmse_de_3, label = 'decoder_wl_3', marker='v'  ,ls='-', markersize=5)
    ax1.plot(en_wl, rmse_de_4, label = 'decoder_wl_4', marker='s'  ,ls='-', markersize=5)
    ax1.plot(en_wl, rmse_de_5, label = 'decoder_wl_5', marker='h'  ,ls='-', markersize=5)
    ax1.plot(en_wl, rmse_de_6, label = 'decoder_wl_6', marker='o'  ,ls='-', markersize=5)
    ax1.plot(en_wl, rmse_de_7, label = 'decoder_wl_7', marker='v'  ,ls='-', markersize=5)
    ax1.plot(en_wl, rmse_de_8, label = 'decoder_wl_8', marker='s'  ,ls='-', markersize=5)
    ax1.plot(en_wl, rmse_de_9, label = 'decoder_wl_9', marker='h'  ,ls='-', markersize=5)
    ax1.plot(en_wl, rmse_de_10, label = 'decoder_wl_10', marker='o'  ,ls='-', markersize=5)
    ax1.plot(en_wl, rmse_de_11, label = 'decoder_wl_11', marker='v'  ,ls='-', markersize=5)
    ax1.plot(en_wl, rmse_de_12, label = 'decoder_wl_12', marker='s'  ,ls='-', markersize=5)
    ax1.plot(en_wl, rmse_de_13, label = 'decoder_wl_13', marker='h'  ,ls='-', markersize=5)
    ax1.plot(en_wl, rmse_de_14, label = 'decoder_wl_14', marker='o'  ,ls='-', markersize=5)
    ax1.plot(en_wl, rmse_de_15, label = 'decoder_wl_15', marker='v'  ,ls='-', markersize=5)
    ax1.plot(en_wl, rmse_de_16, label = 'decoder_wl_16', marker='s'  ,ls='-', markersize=5)
    ax1.plot(en_wl, noquant_rmse, label = 'No_quant.(Baseline)', marker='s'  ,ls='-', markersize=5)
    
    ax1.legend()

    ax2.set_xlabel(xlabel, fontdict=fontlabel, labelpad=16)
    ax2.set_ylabel("Delta1", fontdict=fontlabel, labelpad=16)
    ax2.set_title("Delta1 with word length", fontdict=fontlabel)
    ax2.plot(en_wl, delta_de_2, label = 'decoder_wl_2', marker='o'  ,ls='-', markersize=5)
    ax2.plot(en_wl, delta_de_3, label = 'decoder_wl_3', marker='v'  ,ls='-', markersize=5)
    ax2.plot(en_wl, delta_de_4, label = 'decoder_wl_4', marker='s'  ,ls='-', markersize=5)
    ax2.plot(en_wl, delta_de_5, label = 'decoder_wl_5', marker='h'  ,ls='-', markersize=5)
    ax2.plot(en_wl, delta_de_6, label = 'decoder_wl_6', marker='o'  ,ls='-', markersize=5)
    ax2.plot(en_wl, delta_de_7, label = 'decoder_wl_7', marker='v'  ,ls='-', markersize=5)
    ax2.plot(en_wl, delta_de_8, label = 'decoder_wl_8', marker='s'  ,ls='-', markersize=5)
    ax2.plot(en_wl, delta_de_9, label = 'decoder_wl_9', marker='h'  ,ls='-', markersize=5)
    ax2.plot(en_wl, delta_de_10, label = 'decoder_wl_10', marker='o'  ,ls='-', markersize=5)
    ax2.plot(en_wl, delta_de_11, label = 'decoder_wl_11', marker='v'  ,ls='-', markersize=5)
    ax2.plot(en_wl, delta_de_12, label = 'decoder_wl_12', marker='s'  ,ls='-', markersize=5)
    ax2.plot(en_wl, delta_de_13, label = 'decoder_wl_13', marker='h'  ,ls='-', markersize=5)
    ax2.plot(en_wl, delta_de_14, label = 'decoder_wl_14', marker='o'  ,ls='-', markersize=5)
    ax2.plot(en_wl, delta_de_15, label = 'decoder_wl_15', marker='v'  ,ls='-', markersize=5)
    ax2.plot(en_wl, delta_de_16, label = 'decoder_wl_16', marker='s'  ,ls='-', markersize=5)
    ax2.plot(en_wl, noquant_delta, label = 'No_quant.(Baseline)', marker='s'  ,ls='-', markersize=5)
    

    
    
    
    ax2.legend()
    plt.show()

if __name__ == '__main__':
    main()