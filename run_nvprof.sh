#!/bin/bash

# we create a new short name for each experiment run
RUN_NAME="pygcn"
PYTHON=python3.6
LOG_DIR="GNN_Workload_Characterization/logs"
EXPERIMENT_HASH=$(date +%s%N | sha256sum | head -c 6)
EXP_NAME=${RUN_NAME}_11c740 #${EXPERIMENT_HASH}
OUTPUT_DIR="${LOG_DIR}/${EXP_NAME}"
CSV="${OUTPUT_DIR}/stats.csv"
ROOFLINE_PNG="${OUTPUT_DIR}/ofa.png"
MACHINE="1080ti"
mkdir -p $OUTPUT_DIR

echo "Running new experiments: $EXP_NAME"

if [ "$MACHINE" == "v100" ]; then
	NVPROF="/usr/local/cuda/bin/nvprof"
	DEVICE_NUM=2
elif [ "$MACHINE" == "TITAN Xp" ]; then
	NVPROF="/usr/local/cuda-11.1/bin/nvprof"
	DEVICE_NUM=0
elif [ "$MACHINE" == "1080ti" ]; then
	NVPROF="nvprof"
	DEVICE_NUM=0
	PYTHON="python3"
else
	NVPROF="nvprof"
	DEVICE_NUM=0
fi

OUTFILES=""

echo $DEVICE_NUM

logfile="$OUTPUT_DIR"/nvprof.out

pushd ../gnns/pygcn/pygcn
# CUDA_VISIBLE_DEVICES=$DEVICE_NUM "$NVPROF" \
# --metrics \
# dram_read_transactions,\
# dram_write_transactions,\
# flop_count_dp,\
# flop_count_sp,\
# flop_count_hp \
# --csv \
# --log-file "$logfile" \
# "$PYTHON" test.py

TIMEFILE="$OUTPUT_DIR"/time.txt
CUDA_VISIBLE_DEVICES=$DEVICE_NUM "$PYTHON" time_model.py --time_file $TIMEFILE
popd