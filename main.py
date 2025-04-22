# main.py
import uvicorn
import logging
import time
import json
from fastapi import FastAPI, Depends, HTTPException, Request, status
from fastapi.responses import StreamingResponse, JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from contextlib import asynccontextmanager
import uuid

# Import local modules
from core.config import settings
from core.security import get_api_key
from core.llm_backend import llm_backend # Eagerly initialized backend
# from core.llm_backend import get_llm_backend # For lazy loading
from models.openai_models import (
    ChatCompletionRequest, ChatCompletionResponse, ChatCompletionStreamResponseDelta, UsageInfo, ChatCompletionChoice, ChatMessage,
    CompletionRequest, CompletionResponse, CompletionStreamResponse, CompletionChoice as LegacyCompletionChoice
)

# --- Logging Setup ---
logging.basicConfig(level=settings.LOG_LEVEL, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Rate Limiting Setup ---
limiter = Limiter(key_func=get_remote_address) # Limit based on IP address

# --- FastAPI App Lifespan (for model loading/unloading) ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Ensure model is loaded (if not using eager loading)
    logger.info("Application startup...")
    # If using lazy loading, you might want to trigger the first load here
    # or just let the first request handle it.
    # get_llm_backend() # Example of triggering lazy load
    logger.info(f"Model '{settings.MODEL_NAME}' ready via backend.")
    yield
    # Shutdown: Clean up resources if needed
    logger.info("Application shutdown.")
    # Add any cleanup logic here if necessary (e.g., explicitly releasing GPU memory)

# --- FastAPI App Initialization ---
app = FastAPI(
    title="Local OpenAI-Compatible LLM API",
    version="1.0.0",
    lifespan=lifespan,
    # Add other FastAPI configurations like description, docs_url, etc.
)

# --- Middleware ---
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Middleware to log incoming requests and outgoing responses."""
    start_time = time.time()
    # Obfuscate sensitive headers like Authorization
    headers = dict(request.headers)
    if "authorization" in headers:
        headers["authorization"] = f"Bearer {headers['authorization'].split()[-1][:5]}..." # Show only prefix

    logger.info(f"Request: {request.method} {request.url.path} - Headers: {headers}")
    try:
        response = await call_next(request)
        process_time = time.time() - start_time
        logger.info(f"Response: Status={response.status_code} - Time={process_time:.4f}s")
        # Note: Logging response body can be verbose, especially for streaming. Omitted here.
    except Exception as e:
        process_time = time.time() - start_time
        logger.error(f"Request failed: {request.method} {request.url.path} - Error: {e} - Time={process_time:.4f}s", exc_info=True)
        # Re-raise the exception to let FastAPI handle it
        raise e
    return response


# --- API Endpoints ---

@app.get("/v1/models", dependencies=[Depends(get_api_key)])
@limiter.limit("60/minute") # Example rate limit
async def list_models(request: Request):
    """
    Emulates OpenAI's /v1/models endpoint.
    Returns a list containing the single loaded model.
    """
    logger.info("Request received for /v1/models")
    return {
        "object": "list",
        "data": [
            {
                "id": settings.MODEL_NAME,
                "object": "model",
                "created": int(time.time()), # Or a fixed timestamp if preferred
                "owned_by": "local",
                "permission": [],
                "root": settings.MODEL_NAME,
                "parent": None,
            }
        ],
    }

@app.post("/v1/chat/completions", dependencies=[Depends(get_api_key)])
@limiter.limit("30/minute") # Example rate limit
async def create_chat_completion(
    request_body: ChatCompletionRequest,
    request: Request # For rate limiting key
    # api_key: str = Depends(get_api_key) # Get validated key if needed for logging/auditing
):
    """
    Handles OpenAI-compatible chat completion requests.
    Supports both streaming and non-streaming responses.
    """
    logger.info(f"Received chat completion request for model: {request_body.model}")
    # Optional: Validate if request_body.model matches settings.MODEL_NAME or is compatible
    # if request_body.model != settings.MODEL_NAME:
    #     logger.warning(f"Requested model '{request_body.model}' does not match loaded model '{settings.MODEL_NAME}'")
        # Decide whether to proceed or return an error

    backend = llm_backend # Use the initialized backend instance
    # backend = get_llm_backend() # If using lazy loading

    # Map request parameters (already done in backend, but can double-check here)
    params = {
        "messages": request_body.messages,
        "temperature": request_body.temperature,
        "top_p": request_body.top_p,
        "max_tokens": request_body.max_tokens,
        "stop": request_body.stop,
        "stream": request_body.stream,
        # Add other params like frequency_penalty, presence_penalty if supported by backend
    }

    try:
        completion_or_generator = backend.get_chat_completion(**params)
    except Exception as e:
        logger.error(f"Error generating chat completion: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"LLM backend error: {e}")


    # --- Handle Streaming Response ---
    if request_body.stream:
        logger.debug("Streaming chat completion response")
        async def stream_generator():
            stream_id = f"chatcmpl-{uuid.uuid4().hex}"
            first_chunk = True
            try:
                for chunk in completion_or_generator:
                    # llama-cpp returns chunks like: {'id': '...', 'object': 'chat.completion.chunk', 'created': ..., 'model': '...', 'choices': [{'index': 0, 'delta': {'role': 'assistant'}, 'finish_reason': None}]}
                    # or {'id': '...', ..., 'choices': [{'index': 0, 'delta': {'content': ' response text'}, 'finish_reason': None}]}
                    # or {'id': '...', ..., 'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}]}

                    if not chunk or not chunk.get("choices"):
                        logger.warning(f"Unexpected chunk format from llama-cpp: {chunk}")
                        continue

                    # Reformat llama-cpp chunk to OpenAI SSE format
                    delta = chunk["choices"][0].get("delta", {})
                    finish_reason = chunk["choices"][0].get("finish_reason")

                    # Create the OpenAI-like delta structure
                    response_delta = {
                        "index": chunk["choices"][0].get("index", 0),
                        "delta": {},
                        "finish_reason": finish_reason
                    }

                    # Add role only in the first chunk if present
                    if first_chunk and "role" in delta:
                         response_delta["delta"]["role"] = delta.get("role")
                         first_chunk = False # Role should only be in the first delta message

                    # Add content if present
                    if "content" in delta:
                         response_delta["delta"]["content"] = delta.get("content")


                    # Create the final stream response object
                    stream_response = ChatCompletionStreamResponseDelta(
                        id=stream_id, # Use consistent ID for the stream
                        model=settings.MODEL_NAME, # Or use model from chunk if available: chunk.get("model", settings.MODEL_NAME)
                        choices=[response_delta]
                        # Usage info is typically not included in stream chunks, only in the final non-stream response
                    )

                    # Yield in SSE format: data: {...}\n\n
                    yield f"data: {stream_response.model_dump_json()}\n\n"
                    await asyncio.sleep(0.01) # Small sleep to prevent overwhelming client

                # Send the final [DONE] marker
                yield "data: [DONE]\n\n"
                logger.debug("Finished streaming chat completion response.")

            except Exception as e:
                 logger.error(f"Error during streaming: {e}", exc_info=True)
                 # You might want to send an error message over the stream if possible,
                 # but often the connection might be closed already.
                 # Example: yield f"error: {json.dumps({'message': str(e)})}\n\n"
                 # Ensure the generator stops cleanly
            finally:
                 # Cleanup if needed
                 pass

        # Need to import asyncio for the sleep
        import asyncio
        return StreamingResponse(stream_generator(), media_type="text/event-stream")

    # --- Handle Non-Streaming Response ---
    else:
        logger.debug("Generating non-streaming chat completion response")
        try:
            # The backend returns the full completion dict directly when stream=False
            full_completion = completion_or_generator

            if not full_completion or not full_completion.get("choices"):
                 logger.error(f"Invalid non-streaming response structure from llama-cpp: {full_completion}")
                 raise HTTPException(status_code=500, detail="Invalid response structure from LLM backend.")

            # Extract necessary data from llama-cpp response
            choice_data = full_completion["choices"][0]
            message_data = choice_data.get("message", {})

            # Format into OpenAI ChatCompletionResponse
            response = ChatCompletionResponse(
                id=full_completion.get("id", f"chatcmpl-{uuid.uuid4().hex}"),
                object="chat.completion",
                created=full_completion.get("created", int(time.time())),
                model=full_completion.get("model", settings.MODEL_NAME),
                choices=[
                    ChatCompletionChoice(
                        index=choice_data.get("index", 0),
                        message=ChatMessage(
                            role=message_data.get("role", "assistant"),
                            content=message_data.get("content", "")
                        ),
                        finish_reason=choice_data.get("finish_reason")
                    )
                ],
                usage=UsageInfo( # Extract usage info from llama-cpp response
                    prompt_tokens=full_completion.get("usage", {}).get("prompt_tokens", 0),
                    completion_tokens=full_completion.get("usage", {}).get("completion_tokens", 0),
                    total_tokens=full_completion.get("usage", {}).get("total_tokens", 0),
                )
            )
            logger.debug(f"Non-streaming response generated: {response.model_dump(exclude_none=True)}")
            return response

        except Exception as e:
            logger.error(f"Error processing non-streaming chat completion: {e}", exc_info=True)
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Error processing LLM response: {e}")


# --- Optional: Legacy Completions Endpoint ---
@app.post("/v1/completions", dependencies=[Depends(get_api_key)], include_in_schema=False) # Hide from default docs
@limiter.limit("30/minute")
async def create_completion(
    request_body: CompletionRequest,
    request: Request
    # api_key: str = Depends(get_api_key)
):
    """
    Handles OpenAI-compatible legacy text completion requests.
    Supports both streaming and non-streaming responses.
    NOTE: This endpoint is generally less preferred than /chat/completions.
    """
    logger.info(f"Received legacy completion request for model: {request_body.model}")

    # Ensure the prompt is a single string if a list is provided (llama-cpp expects string)
    prompt = request_body.prompt
    if isinstance(prompt, list):
        # Decide how to handle list prompts (e.g., concatenate, error out, take first)
        # Concatenating might be a simple approach:
        prompt = "\n".join(prompt)
        logger.warning("Received list prompt for legacy completion, concatenating.")
        # Alternatively, raise an error:
        # raise HTTPException(status_code=400, detail="List prompts not supported for /v1/completions endpoint in this implementation.")

    backend = llm_backend
    params = {
        "prompt": prompt,
        "temperature": request_body.temperature,
        "top_p": request_body.top_p,
        "max_tokens": request_body.max_tokens,
        "stop": request_body.stop,
        "stream": request_body.stream,
        "echo": request_body.echo,
        # Add other params if supported
    }

    try:
        completion_or_generator = backend.get_completion(**params)
    except Exception as e:
        logger.error(f"Error generating legacy completion: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"LLM backend error: {e}")

    # --- Handle Streaming Response ---
    if request_body.stream:
        logger.debug("Streaming legacy completion response")
        async def stream_generator():
            stream_id = f"cmpl-{uuid.uuid4().hex}"
            try:
                for chunk in completion_or_generator:
                    # llama-cpp completion stream chunks look like:
                    # {'id': '...', 'object': 'text_completion', 'created': ..., 'model': '...', 'choices': [{'text': ' response chunk', 'index': 0, 'logprobs': None, 'finish_reason': None}]}
                    # Final chunk might have finish_reason set.

                    if not chunk or not chunk.get("choices"):
                        logger.warning(f"Unexpected legacy chunk format from llama-cpp: {chunk}")
                        continue

                    # Reformat llama-cpp chunk to OpenAI SSE format for completions
                    choice_data = chunk["choices"][0]
                    stream_response = CompletionStreamResponse(
                        id=stream_id, # Use consistent ID
                        object="text_completion", # OpenAI uses this for stream chunks too
                        created=chunk.get("created", int(time.time())),
                        model=chunk.get("model", settings.MODEL_NAME),
                        choices=[{ # Structure matches OpenAI's stream choice format
                            "text": choice_data.get("text", ""),
                            "index": choice_data.get("index", 0),
                            "logprobs": choice_data.get("logprobs"), # Usually None in streaming
                            "finish_reason": choice_data.get("finish_reason")
                        }]
                    )

                    yield f"data: {stream_response.model_dump_json()}\n\n"
                    await asyncio.sleep(0.01)

                yield "data: [DONE]\n\n"
                logger.debug("Finished streaming legacy completion response.")

            except Exception as e:
                 logger.error(f"Error during legacy streaming: {e}", exc_info=True)
            finally:
                 pass # Cleanup if needed

        import asyncio
        return StreamingResponse(stream_generator(), media_type="text/event-stream")

    # --- Handle Non-Streaming Response ---
    else:
        logger.debug("Generating non-streaming legacy completion response")
        try:
            full_completion = completion_or_generator

            if not full_completion or not full_completion.get("choices"):
                 logger.error(f"Invalid non-streaming legacy response structure from llama-cpp: {full_completion}")
                 raise HTTPException(status_code=500, detail="Invalid response structure from LLM backend.")

            # Format into OpenAI CompletionResponse
            response = CompletionResponse(
                id=full_completion.get("id", f"cmpl-{uuid.uuid4().hex}"),
                object="text_completion",
                created=full_completion.get("created", int(time.time())),
                model=full_completion.get("model", settings.MODEL_NAME),
                choices=[
                    LegacyCompletionChoice(
                        index=choice.get("index", 0),
                        text=choice.get("text", ""),
                        logprobs=choice.get("logprobs"),
                        finish_reason=choice.get("finish_reason")
                    ) for choice in full_completion.get("choices", [])
                ],
                 usage=UsageInfo( # Extract usage info
                    prompt_tokens=full_completion.get("usage", {}).get("prompt_tokens", 0),
                    completion_tokens=full_completion.get("usage", {}).get("completion_tokens", 0),
                    total_tokens=full_completion.get("usage", {}).get("total_tokens", 0),
                )
            )
            logger.debug(f"Non-streaming legacy response generated: {response.model_dump(exclude_none=True)}")
            return response

        except Exception as e:
            logger.error(f"Error processing non-streaming legacy completion: {e}", exc_info=True)
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Error processing LLM response: {e}")


# --- Root Endpoint (Optional Health Check) ---
@app.get("/", tags=["Health"])
async def read_root():
    """Simple health check endpoint."""
    logger.debug("Root health check requested.")
    return {"status": "ok", "message": "Local LLM API is running."}

# --- Run the server ---
if __name__ == "__main__":
    logger.info(f"Starting Uvicorn server on {settings.HOST}:{settings.PORT}")
    # Note: When running with `uvicorn main:app --reload`, uvicorn handles the reload.
    # For production, use a process manager like gunicorn or systemd.
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        log_level=settings.LOG_LEVEL.lower(), # Set uvicorn log level
        reload=False # Set reload=True for development ONLY
    )

