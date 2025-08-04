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
from llm_server.api.deps import get_authenticated_user, validate_model_id
from llm_server.core.utils import generate_id
from llm_server.core.llm_manager import dynamic_model_manager
from asyncio import anext
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

async def _completion_stream_generator(
    model_id: str,
    stream: AsyncGenerator[dict, None],
    request_id: str
) -> AsyncGenerator[str, None]:
    """
    Yields formatted server-sent events from an async stream of chunks.
    """
    completion_start_time = int(time.time())
    async for output_chunk in stream:
        choices = []
        if "choices" in output_chunk and isinstance(output_chunk["choices"], list):
            for choice_data in output_chunk["choices"]:
                stream_choice = CompletionStreamChoice(
                    index=choice_data.get("index", 0),
                    text=choice_data.get("text", ""),
                    logprobs=None,
                    finish_reason=choice_data.get("finish_reason")
                )
                choices.append(stream_choice)

        if not choices:
            continue

        chunk = CompletionStreamResponse(
            id=request_id,
            object="text_completion.chunk",
            created=completion_start_time,
            model=model_id,
            choices=choices,
        )
        yield f"data: {chunk.model_dump_json(exclude_unset=True)}\n\n"
    yield "data: [DONE]\n\n"


@router.post(
    "/completions",
    response_model=CompletionResponse,
    dependencies=[Depends(get_authenticated_user)]
)
async def create_completion(
    request: CompletionRequest,
):
    """
    Creates a completion for the provided prompt and parameters.
    """
    model_id = validate_model_id(request.model)

    try:
        llama_params = _map_request_to_llama_params(request)
        llama_params["stream"] = request.stream

        stream = dynamic_model_manager.handle_inference_request(
            model_id=model_id,
            llama_params=llama_params,
            method_name="create_completion"
        )

        if request.stream:
            request_id = generate_id("cmpl")
            return EventSourceResponse(
                _completion_stream_generator(model_id, stream, request_id),
                media_type="text/event-stream"
            )
        else:
            output = await anext(stream)
            if "error" in output:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Error from model worker: {output['error']}"
                )
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