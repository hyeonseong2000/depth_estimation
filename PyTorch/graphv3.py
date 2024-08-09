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

    wl = [2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]

    act_fxp_rmse = [4160.618572	,4215.745188	,4157.699801	,4133.767208	,2391.012473,	1149.680906,	770.9976174	,484.4678304,	437.9440399,	342.7601932,	327.2843164	,298.9669432	,290.8353861	,288.3884937	,288.3294847]
    act_fxp_delta = [0	,0.000117112,	7.38E-05,	5.69E-05,	0.073831155,	0.655280115,	0.860938846,	0.906117808,	0.950011075,	0.952777983,	0.976037829	,0.976917927	,0.976969669,	0.982462605	,0.98241321]
    
    act_fp_rmse = [4462.015297,	5000	,4128.115386,	2602.415668	,498.1396742,	361.8756511,	313.1676184	,296.4409375,	296.4548935	,290.9595588	,290.9580221	,286.7390013	,286.7390013	,285.9702893	,285.9702893]
    act_fp_delta = [1.89E-08	,0.000295864,	0.000197319,	0.055622163,	0.941600714	,0.951245328	,0.981437331	,0.98214494	,0.982134682,	0.98232604,	0.982323351,	0.982396611,	0.982396611,	0.98244934,	0.98244934]
    

    act_weight_fxp_rmse = [4160.618572	,4246.214942,	4280.576635	,4583.928291	,4522.636007	,1405.703464,	1835.20277897994,	1206.310442	,1139.900374,	567.4922201	,562.4654026	,326.2096165,	322.3851725,	300.4653874	,298.4515124]
    act_weight_fxp_delta = [0,	0,	0,	0	,0	,0.498043311	,0.130355502358598	,0.575404757,	0.676355299,	0.94443055,	0.960933793,	0.970717966,	0.979857002,	0.977486719,	0.982397521]
    
    act_weight_fp_rmse = [4462.04077	,4505.492995,	4522.898887	,4583.928291	,3436.086967,	1068.506467	,318.7054188	,317.956933,	296.4662567	,292.4334209	,292.4185004	,286.5403518	,286.5403518	,286.2933961,	286.2933961]
    act_weight_fp_delta = [0	,0	,0	,0,	0.062522315,	0.41188665,	0.981152934,	0.981141294,	0.982149312,	0.982343053,	0.982340272,	0.982394872,	0.982394872,	0.982446766,	0.982446766]
    
    

    # For plot 3D data graph
    
    norm1 = mpl.colors.Normalize(vmin=0, vmax=5000)
    norm2 = mpl.colors.Normalize(vmin=0, vmax=1)
    cmaps = plt.get_cmap("plasma")
    
    fig1 = plt.figure(1)
    fig2 = plt.figure(2)
   
    ax1 = fig1.add_subplot(111)
    ax2 = fig2.add_subplot(111)
    
    fontlabel = {"fontsize":"large", "color":"black", "fontweight":"bold"}
    xlabel = "word length" 
    ylabel = "RMSE"
    
    

    ax1.set_xlabel(xlabel, fontdict=fontlabel, labelpad=16)
    ax1.set_ylabel(ylabel, fontdict=fontlabel, labelpad=16)
    ax1.set_title("RMSE with word length", fontdict=fontlabel)
    ax1.plot(wl, act_fxp_rmse, label = 'act_fxp', marker='o'  ,ls='--', markersize=5)
    ax1.plot(wl, act_fp_rmse, label = 'act_fp', marker='v'  ,ls='-', markersize=5)
    ax1.plot(wl, act_weight_fxp_rmse, label = 'act_weight_fxp', marker='s'  ,ls='--', markersize=5)
    ax1.plot(wl, act_weight_fp_rmse, label = 'act_weight_fp', marker='h'  ,ls='-', markersize=5)
    plt.legend()

    ax2.set_xlabel(xlabel, fontdict=fontlabel, labelpad=16)
    ax2.set_ylabel("Delta1", fontdict=fontlabel, labelpad=16)
    ax2.set_title("Delta1 with word length", fontdict=fontlabel)
    ax2.plot(wl, act_fxp_delta, label = 'act_fxp', marker='o'  ,ls='--', markersize=5)
    ax2.plot(wl, act_fp_delta, label = 'act_fp', marker='v'  ,ls='-', markersize=5)
    ax2.plot(wl, act_weight_fxp_delta, label = 'act_weight_fxp', marker='s'  ,ls='--', markersize=5)
    ax2.plot(wl, act_weight_fp_delta, label = 'act_weight_fp', marker='h'  ,ls='-', markersize=5)
    
    
    
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()