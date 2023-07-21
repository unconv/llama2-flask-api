#!/bin/bash

# help command
if [ "$1" == "help" ] || [ "$1" == "--help" ]; then
    echo "Usage: $0 [HOST PORT MODEL MAX_SEQ_LEN MAX_BATCH_SIZE NPROC_PER_NODE]"
    exit
fi

# get arguments
HOST=$1
PORT=$2
MODEL=$3
MAX_SEQ_LEN=$4
MAX_BATCH_SIZE=$5
NPROC_PER_NODE=$6

# set defaults
if [ "$HOST" == "" ]; then
    HOST="127.0.0.1"
fi

if [ "$PORT" == "" ]; then
    PORT="5000"
fi

if [ "$MODEL" == "" ]; then
    MODEL="7b-chat"
fi

if [ "$MAX_SEQ_LEN" == "" ]; then
    MAX_SEQ_LEN=1024
fi

if [ "$MAX_BATCH_SIZE" == "" ]; then
    MAX_BATCH_SIZE=4
fi

if [ "$NPROC_PER_NODE" == "" ]; then
    NPROC_PER_NODE=1
fi

echo "Starting Llama 2 Flask API..."
echo "Model: $MODEL"
echo ""

# run Llama 2
python3 -m torch.distributed.run \
    --nproc_per_node $NPROC_PER_NODE api.py \
    --ckpt_dir llama-2-$MODEL/ \
    --tokenizer_path tokenizer.model \
    --max_seq_len $MAX_SEQ_LEN \
    --max_batch_size $MAX_BATCH_SIZE \
    --host $HOST \
    --port $PORT
