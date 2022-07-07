#!/bin/bash

#
# the following mapping is tied to "smt4"
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
export OMP_PLACES={0:28}
numactl --physcpubind=0-27 --membind=0 $APP
#${APP}
  ;;
[1])
export PAMI_IBV_DEVICE_NAME=mlx5_1:1
export OMP_PLACES={28:28}
numactl --physcpubind=28-55 --membind=0 $APP
#${APP}
  ;;
[2])
export PAMI_IBV_DEVICE_NAME=mlx5_0:1
export OMP_PLACES={56:28}
numactl --physcpubind=56-83 --membind=0 $APP
#${APP}
  ;;
[3])
export PAMI_IBV_DEVICE_NAME=mlx5_3:1
export OMP_PLACES={88:28}
numactl --physcpubind=88-115 --membind=8 $APP
#${APP}
  ;;
[4])
export PAMI_IBV_DEVICE_NAME=mlx5_2:1
export OMP_PLACES={116:28}
numactl --physcpubind=116-143 --membind=8 $APP
#${APP}
  ;;
[5])
export PAMI_IBV_DEVICE_NAME=mlx5_3:1
export OMP_PLACES={144:28}
numactl --physcpubind=144-171 --membind=8 $APP
#${APP}
  ;;
esac
