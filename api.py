# Llama 2 Flask API by Unconventional Coding
# unconventionalcoding@gmail.com

# Modified from Llama 2 example_chat_completion.py
# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

# Llama imports
from typing import Optional
from llama import Llama
import fire

# Flask API imports
from flask import Flask, request, jsonify
import json

def main(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_seq_len: int = 512,
    max_batch_size: int = 4,
    max_gen_len: Optional[int] = None,
    host: str = '127.0.0.1',
    port: int = 5000,
):
    # initialize Llama 2
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )

    # initialize Flask
    app = Flask(__name__)

    # add API route
    @app.route("/chat", methods=["POST"])
    def message_route():
        # load messages from JSON request
        dialogs = [request.json.get("messages")]

        # send messages to Llama 2
        results = generator.chat_completion(
            dialogs,  # type: ignore
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
        )

        # get response from Llama 2
        response = results[0]['generation']

        # mock stream response
        if request.json.get("stream"):
            return "data: " + json.dumps({
                "id": "chatcmpl-123",
                "object": "chat.completion",
                "created": 1677652288,
                "choices": [{
                    "index": 0,
                    "delta": response,
                    "finish_reason": "stop"
                }],
                "usage": {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0
                }
            }) + "\ndata: [DONE]"

        # regular JSON response
        return jsonify({
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "created": 1677652288,
            "choices": [{
                "index": 0,
                "message": response,
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0
            }
        })

    # run Flask API
    app.run(host=host, port=port)

if __name__ == "__main__":
    fire.Fire(main)
