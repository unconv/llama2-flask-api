# Llama 2 Flask API

This is a simple HTTP API for the Llama 2 LLM. It is compatible with the ChatGPT API, so you should be able to use it with any application that supports the ChatGPT API, by changing the API URL to `http://localhost:5000/chat`.

## Usage

After installing Llama 2 from the [official repo](https://github.com/facebookresearch/llama), use `app.py` instead of `example_chat_completion.py` with the official example command, or run:

```console
$ ./run_api.sh
```

After that, you will have a Llama 2 API running at http://localhost:5000/chat

To allow access from the public, use the command:

```console
$ ./run_api.sh 0.0.0.0
```

You can also set other command line arguments:

```console
$ ./run_api.sh [HOST PORT MODEL MAX_SEQ_LEN MAX_BATCH_SIZE NPROC_PER_NODE]
```

## Notes

This is a very simple implementation and doesn't support all the same features as the ChatGPT API (token usage calculation, streaming, function calling etc.)

If the API is called with `streaming=true`, it will respond with a single event with the whole response.
