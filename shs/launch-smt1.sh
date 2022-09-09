#!/bin/bash

#
# the following mapping is tied to "smt1"
# this mapping may not be optimal for a given application
#

APP=$1

grank=$PMIX_RANK
lrank=$(($PMIX_RANK%6))
export PAMI_ENABLE_STRIPING=0

#if [ $grank -eq 0 ]; then 
  #export NCCL_DEBUG_FILE="/mnt/bb/$USER/nccldeug.${LSB_JOBID}/nccldebug.${grank}"
#fi 

#LD_PRELOAD=/gpfs/alpine/lrn001/proj-shared/jqyin/directconv/directconv.so


case ${lrank} in
[0])
export PAMI_IBV_DEVICE_NAME=mlx5_0:1
export OMP_PLACES={0:7:4}
numactl --physcpubind=0,4,8,12,16,20,24 --membind=0 $APP 
#${APP}
  ;;
[1])
export PAMI_IBV_DEVICE_NAME=mlx5_1:1
export OMP_PLACES={28:7:4}
numactl --physcpubind=28,32,36,40,44,48,52 --membind=0 $APP 
#${APP}
  ;;
[2])
export PAMI_IBV_DEVICE_NAME=mlx5_0:1
export OMP_PLACES={56:7:4}
numactl --physcpubind=56,60,64,68,72,76,80 --membind=0 $APP 
#${APP}
  ;;
[3])
export PAMI_IBV_DEVICE_NAME=mlx5_3:1
export OMP_PLACES={88:7:4}
numactl --physcpubind=88,92,96,100,104,108,112 --membind=8 $APP 
#${APP}
  ;;
[4])
export PAMI_IBV_DEVICE_NAME=mlx5_2:1
export OMP_PLACES={116:7:4}
numactl --physcpubind=116,120,124,128,132,136,140 --membind=8 $APP 
#${APP}
  ;;
[5])
export PAMI_IBV_DEVICE_NAME=mlx5_3:1
export OMP_PLACES={144:7:4}
numactl --physcpubind=144,148,152,156,160,164,168 --membind=8 $APP 
#${APP}
  ;;
esac
