

for ((int=0;int<=8;int++))
do
    for((frac=0; frac<=$[16-$int]; frac++))
    do
        CUDA_VISIBLE_DEVICES=1 python evaluate_quant.py --dtype "fxp" --exp $int --mant $frac --mode "stochastic"
    done
done





