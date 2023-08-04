# Llama 2 Flask API

This is a simple HTTP API for the Llama 2 LLM. It is compatible with the ChatGPT API, so you should be able to use it with any application that supports the ChatGPT API, by changing the API URL to `http://localhost:5000/chat`.

## Usage

After installing Llama 2 from the [official repo](https://github.com/facebookresearch/llama), clone this repository into the Llama directory or just copy `api.py` from this repo to the root of the Llama directory.

Then just run the API:

```console
$ ./api.py
```

After that, you will have a Llama 2 API running at http://localhost:5000/chat

To allow access from the public, use the command:

```console
$ ./api.py --host 0.0.0.0
```

You can also specify a `--port` and other command line arguments (see `--help`)

## Notes

This is a very simple implementation and doesn't support all the same features as the ChatGPT API (token usage calculation, streaming, function calling etc.)

If the API is called with `streaming=true`, it will respond with a single event with the whole response.
