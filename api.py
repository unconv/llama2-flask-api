#!/usr/bin/env python3

# Llama 2 Flask API by Unconventional Coding
# unconventionalcoding@gmail.com

# Llama imports
from llama import Llama
import fire


# API Imports
from torch.multiprocessing import Process, Queue
from flask import Flask, request, jsonify
import torch.distributed as dist
import argparse
import torch
import json
import os

parser = argparse.ArgumentParser(description="Llama 2 Flask API")

parser.add_argument('--model', type=str, default='7b-chat', help='The model name (str)')
parser.add_argument('--host', type=str, default='127.0.0.1', help='API host (str)')
parser.add_argument('--port', type=int, default=5000, help='API port (int)')
parser.add_argument('--max_seq_len', type=int, default=512, help='Maximum sequence length (int)')
parser.add_argument('--backend', type=str, default='nccl', help='Backend (nccl for GPU, gloo for CPU) (str)')
parser.add_argument('--temperature', type=float, default=0.6, help='Temperature for sampling (float)')
parser.add_argument('--top_p', type=float, default=0.9, help='Top p value for nucleus sampling (float)')
parser.add_argument('--world_size', type=int, default=None, help='Number of parallel processes (int)')
parser.add_argument('--max_batch_size', type=int, default=4, help='Maximum batch size (int)')
parser.add_argument('--max_gen_len', type=int, default=None, help='Maximum generation length (int)')
parser.add_argument('--tokenizer_path', type=str, default='tokenizer.model', help='Path to tokenizer model (str)')
parser.add_argument('--ckpt_dir', type=str, default=None, help='The full path to the model directory (str)')
parser.add_argument('--llama_addr', type=str, default='127.0.0.1', help='Llama 2 master address (str)')
parser.add_argument('--llama_port', type=int, default=29500, help='Llama 2 master port (int)')
args = parser.parse_args()

# get default ckpt_dir if not provided
if args.ckpt_dir is None:
    args.ckpt_dir = "llama-2-" + args.model

# guess world size if not provided
if args.world_size is None:
    if "7b" in args.ckpt_dir.lower():
        args.world_size = 1
    elif "13b" in args.ckpt_dir.lower():
        args.world_size = 2
    elif "70b" in args.ckpt_dir.lower():
        args.world_size = 8

notices = False

# check checkpoint dir location
if not os.path.exists(args.ckpt_dir):
    parent_ckpt_dir = os.path.join("..", args.ckpt_dir)
    if os.path.exists(parent_ckpt_dir):
        print(f"NOTICE: Reading model from '{parent_ckpt_dir}'")
        args.ckpt_dir = parent_ckpt_dir
    else:
        print(f"WARNING: Model directory '{args.ckpt_dir}' not found!")
    notices = True

# check tokenizer location
if not os.path.exists(args.tokenizer_path):
    parent_tokenizer_path = os.path.join("..", args.tokenizer_path)
    if os.path.exists(parent_tokenizer_path):
        print(f"NOTICE: Reading tokenizer from '{parent_tokenizer_path}'")
        args.tokenizer_path = parent_tokenizer_path
    else:
        print(f"WARNING: Tokenizer file '{args.tokenizer_path}' not found!")
    notices = True

if notices:
    print()

# initialize queues
request_queues = [Queue() for _ in range(args.world_size)]
response_queues = [Queue() for _ in range(args.world_size)]


def respond_json(response, key="message"):
    return {
        "id": "chatcmpl-123",
        "object": "chat.completion",
        "created": 1677652288,
        "choices": [{
            "index": 0,
            key: response,
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0
        }
    }


def run(rank, size, request_queue, response_queue):
    os.environ['LOCAL_RANK'] = str(rank)

    # initialize Llama 2
    generator = Llama.build(
        ckpt_dir=args.ckpt_dir,
        tokenizer_path=args.tokenizer_path,
        max_seq_len=args.max_seq_len,
        max_batch_size=args.max_batch_size,
    )

    # send initialization signal
    response_queue.put("INITIALIZED")

    while True:
        # load messages from queue
        dialogs = [request_queue.get()]

        # replace Llama 2 default system message
        if dialogs[0][0]["role"] != "system":
            dialogs[0] = [
                {
                    "role": "system",
                    "content": "You are a helpful assistant",
                }
            ] + dialogs[0]

        # send messages to Llama 2
        results = generator.chat_completion(
            dialogs,  # type: ignore
            max_gen_len=args.max_gen_len,
            temperature=args.temperature,
            top_p=args.top_p,
        )

        # get response from Llama 2
        response = results[0]['generation']

        response_queue.put(response)


def init_process(rank, size, fn, request_queue, response_queue, backend=args.backend):
    os.environ['MASTER_ADDR'] = args.llama_addr
    os.environ['MASTER_PORT'] = str(args.llama_port)
    os.environ['WORLD_SIZE'] = str(size)
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size, request_queue, response_queue)


def check_messages(messages):
    if not isinstance(messages, list):
        return jsonify({
            "error": {
                "message": "'messages' must be a list",
                "code": "invalid_message_list"
            }
        }), 400

    for message in messages:
        if not isinstance(message, dict) or 'role' not in message or 'content' not in message:
            return jsonify({
                "error": {
                    "message": "Each message must have a 'role' and a 'content'",
                    "code": "invalid_message"
                }
            }), 400

    return None


# define API route function
def message_route():
    # get messages from request
    messages = request.json.get("messages")

    # validate message format
    errors = check_messages(messages)
    if errors:
        return errors

    # add messages to queue for Llama 2
    for rank in range(args.world_size):
        request_queues[rank].put(messages)

    # wait for response
    for rank in range(args.world_size):
        response = response_queues[rank].get()

    # return mocked stream response
    if request.json.get("stream"):
        maxlen = 128
        rc = response["content"]
        deltas = [rc[i:i+maxlen] for i in range(0, len(rc), maxlen)]
        output = ""
        for delta in deltas:
            delta_response = {
                "role": response["role"],
                "content": delta
            }
            output += "data: " + json.dumps(respond_json(delta_response, "delta")) + "\n"
        return output + "data: [DONE]"

    # return regular JSON response
    return jsonify(respond_json(response))


def main():
    print("Initializing Llama 2...")
    print(f"Model: {args.ckpt_dir}\n")

    processes = []

    # initialize all Llama 2 processes
    for rank in range(args.world_size):
        p = Process(target=init_process, args=(rank, args.world_size, run, request_queues[rank], response_queues[rank]))
        p.start()
        processes.append(p)

    # wait for Llama 2 initialization
    for rank in range(args.world_size):
        response = response_queues[rank].get()

    print("\nStarting Flask API...")

    app = Flask(__name__)
    app.route("/chat", methods=["POST"])(message_route)
    app.run(host=args.host, port=args.port)

    for p in processes:
        p.join()

if __name__ == "__main__":
    main()
