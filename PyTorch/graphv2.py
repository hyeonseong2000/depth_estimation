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
    with open("./results/results_act_weight_fp_.csv",'r',newline='') as f:
        reader = csv.reader(f)
        for i,line in enumerate(reader):
            if (i == 0):
                exps_load = line
            elif i==1 :
                mants_load = line
            elif i==2 :
                rmses_load = line
            elif i==3 :
                deltas_load = line
    
    exps_load = [float(item) for item in exps_load]
    mants_load = [float(item) for item in mants_load]
    rmses_load = [float(item) for item in rmses_load]
    deltas_load = [float(item) for item in deltas_load]

    
    

    # For plot 3D data graph
    
    norm1 = mpl.colors.Normalize(vmin=0, vmax=5000)
    norm2 = mpl.colors.Normalize(vmin=0, vmax=1)
    cmaps = plt.get_cmap("plasma")
    fig1 = plt.figure(1)
    fig2 = plt.figure(2)

    ax1 = fig1.add_subplot(111)
    ax2 = fig2.add_subplot(111)
 
    
    fontlabel = {"fontsize":"large", "color":"black", "fontweight":"bold"}
    xlabel = "exponent" 
    ylabel = "mantissa"
    
    ax1.set_xlabel(xlabel, fontdict=fontlabel, labelpad=16)
    ax1.set_ylabel(ylabel, fontdict=fontlabel, labelpad=16)
    ax1.set_title("RMSE", fontdict=fontlabel)
    ax1.scatter(exps_load, mants_load, c=rmses_load, cmap=cmaps)
    fig1.colorbar(mpl.cm.ScalarMappable(norm=norm1, cmap=cmaps))

    
    ax2.set_xlabel(xlabel, fontdict=fontlabel, labelpad=16)
    ax2.set_ylabel(ylabel, fontdict=fontlabel, labelpad=16)
    ax2.set_title("Delta1", fontdict=fontlabel)
    ax2.scatter(exps_load, mants_load, c=deltas_load, cmap=cmaps)
    fig2.colorbar(mpl.cm.ScalarMappable(norm=norm2, cmap=cmaps))


    
    
    #ax1.set_zscale('log')
    #ax2.set_zscale('log')
    
    
    
    plt.show()

if __name__ == '__main__':
    main()