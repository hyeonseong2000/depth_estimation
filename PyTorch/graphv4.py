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
    with open("./results/results_act_weight_full_fxp.csv",'r',newline='') as f:
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
    ylabel = "Decoder_WL"
    
    ax1.set_xlabel(xlabel, fontdict=fontlabel, labelpad=16)
    ax1.set_ylabel(ylabel, fontdict=fontlabel, labelpad=16)
    ax1.set_title("RMSE", fontdict=fontlabel)
    ax1.scatter(wl_en_load, wl_de_load, c=rmses_load, cmap=cmaps)
    fig1.colorbar(mpl.cm.ScalarMappable(norm=norm1, cmap=cmaps))

    
    ax2.set_xlabel(xlabel, fontdict=fontlabel, labelpad=16)
    ax2.set_ylabel(ylabel, fontdict=fontlabel, labelpad=16)
    ax2.set_title("Delta1", fontdict=fontlabel)
    ax2.scatter(wl_en_load, wl_de_load, c=deltas_load, cmap=cmaps)
    fig2.colorbar(mpl.cm.ScalarMappable(norm=norm2, cmap=cmaps))


    
    
    #ax1.set_zscale('log')
    #ax2.set_zscale('log')
    
    
    
    plt.show()

if __name__ == '__main__':
    main()