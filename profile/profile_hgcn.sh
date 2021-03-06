#!/bin/bash

# we create a new short name for each experiment run
BATCH_NAME="hgcn"
PYTHON=python3.6
HOME=""
PROJECT_HOME=$HOME/gnns/GNN_Workload_Characterization
LOG_DIR=$PROJECT_HOME/"logs"
EXPERIMENT_HASH=$(date +%s%N | sha256sum | head -c 6)
EXP_NAME=${BATCH_NAME}_${EXPERIMENT_HASH}
OUTPUT_DIR="${LOG_DIR}/${EXP_NAME}"
WORKLOAD_DIR=$HOME/gnns/hgcn
TEST_PY=${WORKLOAD_DIR}/test.py
NVPROF_OUT=$OUTPUT_DIR/nvprof.out
STATS_CSV=$OUTPUT_DIR/stats.csv
MACHINE="v100"

echo "Running new experiments: $EXP_NAME"

if [ "$MACHINE" == "v100" ]; then
	NVPROF="nvprof"
	DEVICE_NUM=2
	PYTHON="python"
elif [ "$MACHINE" == "a100" ]; then
	NVPROF="nvprof"
	DEVICE_NUM=0
	PYTHON="python"
elif [ "$MACHINE" == "TITAN-Xp" ]; then
	NVPROF="/usr/local/cuda-11.1/bin/nvprof"
	DEVICE_NUM=0
elif [ "$MACHINE" == "1080ti" ]; then
	NVPROF="nvprof"
	DEVICE_NUM=1
	PYTHON="python3"
else
	NVPROF="nvprof"
	DEVICE_NUM=0
fi

mkdir -p $OUTPUT_DIR
IN_ARGS=" --task lp --dataset cora --model HGCN --lr 0.01 --dim 16 --num-layers 2 --act relu --bias 1 --dropout 0.5 --weight-decay 0.001 --manifold PoincareBall --log-freq 5 --cuda 0 --c None"

pushd $WORKLOAD_DIR

CUDA_VISIBLE_DEVICES=$DEVICE_NUM "$NVPROF" \
--metrics \
dram_read_transactions,\
dram_write_transactions,\
flop_count_dp,\
flop_count_sp,\
flop_count_hp \
--csv \
--log-file "$NVPROF_OUT" \
"$PYTHON" "$TEST_PY" $IN_ARGS

"$PYTHON" "$TEST_PY" $IN_ARGS --time_file ${OUTPUT_DIR}/time.txt

popd

"${PYTHON}" ${PROJECT_HOME}/common/parser.py --folder ${OUTPUT_DIR} --file ${NVPROF_OUT} --csv ${STATS_CSV}

"${PYTHON}" ${PROJECT_HOME}/common/plot_tradeoffs.py --csv $STATS_CSV --machine $MACHINE --plotdir $OUTPUT_DIR/graphs
