#!/bin/bash

# we create a new short name for each experiment run
BATCH_NAME="gcn"
PYTHON=python3.6
HOME="/home/natasha"
PROJECT_HOME=$HOME/gnns/GNN_Workload_Characterization
LOG_DIR=$PROJECT_HOME/"logs"
EXPERIMENT_HASH=$(date +%s%N | sha256sum | head -c 6)
EXP_NAME=${BATCH_NAME}_${EXPERIMENT_HASH}
OUTPUT_DIR="${LOG_DIR}/${EXP_NAME}"
WORKLOAD_DIR=$HOME/gnns/pygcn/pygcn
TEST_PY=${WORKLOAD_DIR}/test.py
NVPROF_OUT=$OUTPUT_DIR/nvprof.out
STATS_CSV=$OUTPUT_DIR/stats.csv
MACHINE="a100"

echo "Running new experiments: $EXP_NAME"

START_STOP_ARGS="--profile-from-start off"
METRICS="dram_read_transactions,dram_write_transactions,flop_count_dp,flop_count_sp,flop_count_hp"
PARSER_ARGS="--folder ${OUTPUT_DIR} --file ${NVPROF_OUT} --csv ${STATS_CSV}"

if [ "$MACHINE" == "v100" ]; then
	NVPROF="/usr/local/cuda/bin/nvprof"
	DEVICE_NUM=2
	PYTHON="python"
elif [ "$MACHINE" == "a100" ]; then
	NVPROF="/usr/local/cuda/bin/nv-nsight-cu-cli"
	DEVICE_NUM=0
	PYTHON="python3"
	METRICS="dram__sectors_write,dram__sectors_read,smsp__sass_thread_inst_executed_op_fadd_pred_on,smsp__sass_thread_inst_executed_op_fmul_pred_on,smsp__sass_thread_inst_executed_op_hfma_pred_on.sum"
	PARSER_ARGS="--nsys True --folder ${OUTPUT_DIR} --file ${NVPROF_OUT} --csv ${STATS_CSV}"
	#START_STOP_ARGS="-c cudaProfilerApi --profile-from-start off"
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
IN_ARGS=" --pkl_file gcn.pkl"

pushd $WORKLOAD_DIR

CUDA_VISIBLE_DEVICES=$DEVICE_NUM $NVPROF \
$START_STOP_ARGS \
--metrics $METRICS \
--csv \
--log-file "$NVPROF_OUT" \
"$PYTHON" "$TEST_PY" $IN_ARGS

"$PYTHON" "$TEST_PY" $IN_ARGS --time_file ${OUTPUT_DIR}/time.txt

popd

"${PYTHON}" ${PROJECT_HOME}/common/parser.py $PARSER_ARGS

"${PYTHON}" ${PROJECT_HOME}/common/plot_tradeoffs.py --csv $STATS_CSV --machine $MACHINE --plotdir $OUTPUT_DIR/graphs
