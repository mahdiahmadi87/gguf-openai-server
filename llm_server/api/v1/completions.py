# llm_server/api/v1/completions.py
import time
import json
from fastapi import APIRouter, Depends, HTTPException, status, Request
from sse_starlette.sse import EventSourceResponse
from typing import AsyncGenerator, Union

from llm_server.schemas.completion import (
    CompletionRequest, CompletionResponse, CompletionChoice,
    CompletionStreamResponse, CompletionStreamChoice, ChoiceLogProbs
)
from llm_server.schemas.common import UsageInfo
# Updated dependency import
from llm_server.api.deps import get_authenticated_user, validate_model_id, get_model_instance_from_id
from llm_server.core.utils import generate_id
from llama_cpp import Llama, LlamaGrammar

router = APIRouter(tags=["Completions"])

# Functions _map_request_to_llama_params, _parse_completion_output,
# and _completion_stream_generator remain the same...

# --- Add _map_request_to_llama_params here ---
def _map_request_to_llama_params(request: CompletionRequest) -> dict:
    params = {
        "prompt": request.prompt,
        "suffix": request.suffix,
        "max_tokens": request.max_tokens if request.max_tokens is not None else -1,
        "temperature": request.temperature,
        "top_p": request.top_p,
        "logprobs": request.logprobs,
        "echo": request.echo,
        "stop": request.stop,
        "frequency_penalty": request.frequency_penalty,
        "presence_penalty": request.presence_penalty,
        "top_k": request.top_k,
        "repeat_penalty": request.repeat_penalty if request.repeat_penalty is not None else 1.1,
        "logit_bias": request.logit_bias
    }
    return {k: v for k, v in params.items() if v is not None}

# --- Add _parse_completion_output here ---
def _parse_completion_output(output: dict, model_id: str, request: CompletionRequest) -> CompletionResponse:
    choices = []
    for choice_data in output.get("choices", []):
        logprobs = None
        if choice_data.get("logprobs") is not None:
            logprobs = ChoiceLogProbs( # Simplified, adjust as needed
                tokens=choice_data["logprobs"].get("tokens"),
                token_logprobs=choice_data["logprobs"].get("token_logprobs"),
                top_logprobs=choice_data["logprobs"].get("top_logprobs"),
                text_offset=choice_data["logprobs"].get("text_offset")
            )
        choices.append(
            CompletionChoice(
                index=choice_data.get("index", 0),
                text=choice_data.get("text", ""),
                logprobs=logprobs,
                finish_reason=choice_data.get("finish_reason", "stop"),
            )
        )
    usage = None
    if output.get("usage"):
        usage_data = output["usage"]
        usage = UsageInfo(
            prompt_tokens=usage_data.get("prompt_tokens", 0),
            completion_tokens=usage_data.get("completion_tokens", 0),
            total_tokens=usage_data.get("total_tokens", 0),
        )
    return CompletionResponse(
        id=output.get("id", generate_id("cmpl")),
        object="text_completion",
        created=output.get("created", int(time.time())),
        model=output.get("model", model_id),
        choices=choices,
        usage=usage
    )

# --- Add _completion_stream_generator here ---
async def _completion_stream_generator(
    model_id: str,
    llama_instance: Llama,
    llama_params: dict,
    request_id: str
) -> AsyncGenerator[str, None]:
    stream = llama_instance.create_completion(**llama_params, stream=True)
    completion_start_time = int(time.time())
    for output_chunk in stream:
        choices = []
        for choice_data in output_chunk.get("choices", []):
             stream_choice = CompletionStreamChoice(
                index=choice_data.get("index", 0),
                text=choice_data.get("text", ""),
                logprobs=None,
                finish_reason=choice_data.get("finish_reason")
             )
             choices.append(stream_choice)
        chunk = CompletionStreamResponse(
            id=request_id,
            object="text_completion.chunk",
            created=completion_start_time,
            model=model_id,
            choices=choices,
        )
        yield f"data: {chunk.model_dump_json()}\n\n"
    yield "data: [DONE]\n\n"


# --- Modified Endpoint ---
@router.post(
    "/completions",
    response_model=CompletionResponse, # For non-streaming
    dependencies=[Depends(get_authenticated_user)] # Apply authentication dependency
)
async def create_completion(
    request: CompletionRequest, # Request body is parsed here
    # request_object: Request # Optional: Inject raw request if needed
):
    """
    Creates a completion for the provided prompt and parameters.
    Supports both streaming and non-streaming responses.
    """
    # 1. Validate the model ID from the request body
    model_id = validate_model_id(request.model)

    # 2. Get the Llama instance using the validated model_id
    llama = get_model_instance_from_id(model_id) # Handles exceptions

    # --- The rest of the endpoint logic ---

    if request.n is not None and request.n > 1:
         print(f"Warning: Requesting n={request.n} choices. llama-cpp-python might only generate one.")
    if request.best_of is not None and request.best_of > 1:
         print(f"Warning: Requesting best_of={request.best_of}. Not directly supported.")

    try:
        llama_params = _map_request_to_llama_params(request)

        if request.stream:
            request_id = generate_id("cmpl")
            # Pass the retrieved model_id and llama instance
            return EventSourceResponse(
                _completion_stream_generator(model_id, llama, llama_params, request_id),
                media_type="text/event-stream"
            )
        else:
            output = llama.create_completion(**llama_params, stream=False)
            # Pass the validated model_id for parsing
            response = _parse_completion_output(output, model_id, request)
            return response

    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Invalid request parameters: {e}")
    except Exception as e:
        print(f"Error during completion generation for model '{model_id}': {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error during model inference: {str(e)}"
        )