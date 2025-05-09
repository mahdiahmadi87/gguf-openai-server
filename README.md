# GGUF OpenAI Server

This repository contains a server implementation for running OpenAI-compatible APIs using GGUF models. The server is designed to be lightweight, efficient, and easy to deploy.

## Features
- OpenAI API compatibility
- Support for GGUF models
- Scalable and performant
- Easy to configure and extend

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/mahdiahmadi87/gguf-openai-server.git
   ```

2. Navigate to the project directory:
   ```bash
   cd gguf-openai-server
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Start the server:
   ```bash
   python server.py
   ```

2. Access the API at `http://localhost:8000`.

## Configuration

Configuration options can be set in the `config.json` file. Refer to the comments in the file for details on each option.

## Running with Docker

You can run this project using Docker for a consistent and isolated environment.

### Requirements
- Docker (latest version recommended)
- Docker Compose

### Build and Run

1. Build and start the server using Docker Compose:
   ```bash
   docker compose up --build
   ```
   This will build the image using Python 3.11-slim and install all dependencies as specified in `requirements.txt`.

2. The FastAPI server will be available at `http://localhost:8000`.

### Environment Variables
- The server can be configured using environment variables. You may provide a `.env` file in the project root and uncomment the `env_file` line in `docker-compose.yml` to load it automatically.

### Ports
- The API server is exposed on port **8000** by default (`localhost:8000`).

### Special Configuration
- No volumes or persistent storage are required for this setup.
- All GGUF models should be placed in the `models/` directory (already included in the Docker image).
- No additional configuration is needed unless you wish to override settings via environment variables or the `config/` directory.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
