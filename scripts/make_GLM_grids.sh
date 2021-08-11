#!/bin/bash
#resample GLM L2 data to GOES-16 CONUS 20km
for day in {153..274}
do
    for hour in 0{0..9} {10..23}
    do
        f="/public/home/zhangxin/new/papers/ERL_2021/data/GOES-16/GLM_L2/OR_GLM-L2-LCFA_G16_s2020${day}${hour}*"
        python make_GLM_grids.py --fixed_grid --split_events --goes_position east --goes_sector conus --dx=20 --dy=20 --dt=300 $f
    done
done
