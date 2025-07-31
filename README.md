
# GGUF OpenAI Server

**A lightweight, OpenAI-compatible server for running GGUF models locally or with Docker.**

## THIS IS CPU VERSION!

---

## Features

- **OpenAI API compatibility** “ Works out-of-the-box with tools using OpenAI's APIs.
- **Runs GGUF models** “ Powered by [llama.cpp](https://github.com/ggerganov/llama.cpp).
- **Fast & efficient** “ Minimal dependencies and blazing fast inference.
- **Docker support** “ Fully containerized for easy deployment.
- **Customizable** “ Configure everything via `.env`.

---

## Installation (Without Docker)

```bash
git clone https://github.com/mahdiahmadi87/gguf-openai-server.git
cd gguf-openai-server
pip install -r requirements.txt
```

### Run the server

```bash
python -m llm_server.main
```

Make sure you have a `.env` file with at least:

```env
MODELS_CONFIG='[{"model_id": "gemma-3-4b", "model_path": "models/gemma-3-4b-it-q4_0.gguf", "n_gpu_layers": -1, "is_multimodal": true}]'
```

> **Note:** If you're not using Docker, make sure `model_path` doesn't include `/app/`.

---

## Running with Docker

### Requirements

- Docker (latest version)
- Docker Compose

### Build & Run

```bash
docker compose up --build
```

The API will be available at:  
**http://localhost:8000**

---

## Configuration

- Default configs live in `.env.simple`.
- You can override most options with environment variables.
- Environment variables can be loaded from a `.env` file (see `docker-compose.yml`).

### Example `.env` (Docker)

```env
MODELS_CONFIG='[{"model_id": "gemma-3-4b", "model_path": "/app/models/gemma-3-4b-it-q4_0.gguf", "n_gpu_layers": -1, "is_multimodal": true}]'
```

---

## API Endpoints

| Endpoint        | Method | Description                      |
|----------------|--------|----------------------------------|
| `/v1/chat/completions` | POST   | OpenAI-style chat endpoint         |
| `/v1/completions`      | POST   | OpenAI-style completions endpoint  |
| `/docs`                | GET    | Api Document                       |

> **Pro tip:** You can use this with any library or service that speaks OpenAI, like LangChain, llama-index, or even ChatGPT plugins.

---

## Example Clients

- [OpenAI Python SDK](https://github.com/openai/openai-python)
- [LangChain](https://www.langchain.com/)
- [llama-index](https://www.llamaindex.ai/)

---

## Where to Get GGUF Models?

Check out [HuggingFace](https://huggingface.co/) “ tons of optimized GGUF models for all kinds of tasks.

---

## Contributing

Got ideas? Spotted a bug?  
Feel free to open an issue or pull request!

---

## License

MIT License “ Do whatever you want, just don't sell it as yours ;)

---

Made with caffeine & llama magic by [@mahdiahmadi87](https://github.com/mahdiahmadi87)
