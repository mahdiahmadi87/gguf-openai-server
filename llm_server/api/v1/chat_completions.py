# llm_server/api/v1/chat_completions.py
import time
import json
from fastapi import APIRouter, Depends, HTTPException, status, Request
from sse_starlette.sse import EventSourceResponse
from typing import AsyncGenerator, List
import logging # <--- Add logging import

from llm_server.schemas.common import UsageInfo
from llm_server.api.deps import get_authenticated_user, validate_model_id, get_model_instance_from_id
from llm_server.core.utils import generate_id
from llama_cpp import Llama

# ... (other imports) ...
from llm_server.schemas.chat import ( # Make sure these are imported
    ChatCompletionRequest, ChatMessage,
    ChatCompletionResponse, ChatCompletionChoice, ChatCompletionMessage,
    ChatCompletionStreamResponse, ChatCompletionStreamChoice, ChatCompletionStreamDelta
)

# --- Add helper functions _map..., _parse... here if they aren't already ---
# (Keep them as they were)
def _map_chat_request_to_llama_params(request: ChatCompletionRequest) -> dict:
    # ... (implementation from previous steps) ...
    params = {
        "messages": [msg.model_dump(exclude_none=True) for msg in request.messages],
        "max_tokens": request.max_tokens if request.max_tokens is not None else -1,
        "temperature": request.temperature,
        "top_p": request.top_p,
        "stop": request.stop,
        "frequency_penalty": request.frequency_penalty,
        "presence_penalty": request.presence_penalty,
        "top_k": request.top_k,
        "repeat_penalty": request.repeat_penalty if request.repeat_penalty is not None else 1.1,
        "logit_bias": request.logit_bias,
    }
    return {k: v for k, v in params.items() if v is not None}

def _parse_chat_completion_output(output: dict, model_id: str, request: ChatCompletionRequest) -> ChatCompletionResponse:
     # ... (implementation from previous steps) ...
    choices = []
    for choice_data in output.get("choices", []):
        message_data = choice_data.get("message", {})
        choices.append(
            ChatCompletionChoice(
                index=choice_data.get("index", 0),
                message=ChatCompletionMessage(
                    role=message_data.get("role", "assistant"),
                    content=message_data.get("content", ""),
                ),
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
    return ChatCompletionResponse(
        id=output.get("id", generate_id("chatcmpl")),
        object="chat.completion",
        created=output.get("created", int(time.time())),
        model=output.get("model", model_id),
        choices=choices,
        usage=usage,
        system_fingerprint=output.get("system_fingerprint")
    )
# ---

router = APIRouter(tags=["Chat"])


import asyncio
import traceback # Ensure traceback is imported


logger = logging.getLogger(__name__)

async def _chat_completion_stream_generator(
    model_id: str,
    llama_instance: Llama,
    llama_params: dict,
    request_id: str
) -> AsyncGenerator[str, None]: # Or AsyncGenerator[Union[str, dict], None] if yielding dicts
    """Generates Server-Sent Events for streaming chat completion."""
    completion_start_time = int(time.time())
    logger.debug(f"[{request_id}] Chat stream generator started.")
    exited_prematurely = False

    # --- PREAMBLE ---
    initial_chunk = ChatCompletionStreamResponse(
        id=request_id, object="chat.completion.chunk", created=completion_start_time, model=model_id,
        choices=[ChatCompletionStreamChoice(index=0, delta={}, finish_reason=None)]
    )
    initial_payload = initial_chunk.model_dump_json(exclude_unset=True)
    logger.info(f"[{request_id}] Yielding PREAMBLE SSE chunk payload: {initial_payload}")
    try:
        # CORRECTED YIELD: Yield only the JSON string payload
        yield initial_payload
    except GeneratorExit:
        logger.warning(f"[{request_id}] GeneratorExit caught after preamble yield...")
        exited_prematurely = True
        return
    except Exception as e:
        logger.error(f"[{request_id}] Error yielding preamble: {e}", exc_info=True)
        exited_prematurely = True
        return
    # --- END PREAMBLE ---

    # --- Core Streaming Logic ---
    stream = None
    try:
        logger.debug(f"[{request_id}] Starting llama_cpp stream loop...")
        stream = llama_instance.create_chat_completion(**llama_params, stream=True)
        i = 0
        for output_chunk in stream:
            try:
                # ... (Logging first chunk etc.) ...
                if i == 0:
                    logger.info(f"[{request_id}] !!! FIRST *REAL* RAW llama_cpp stream chunk: {output_chunk!r}")

                logger.debug(f"[{request_id}] Processing raw chunk {i}: {output_chunk!r}")

                # --- Parse Chunk ---
                # ... (Parsing logic remains the same) ...
                choices = []
                raw_choices = output_chunk.get("choices") if isinstance(output_chunk, dict) else None
                if isinstance(raw_choices, list):
                    for choice_data in raw_choices:
                        # ... parsing ...
                        delta_data = choice_data.get("delta", {})
                        finish_reason = choice_data.get("finish_reason")
                        delta = ChatCompletionStreamDelta(
                            role=delta_data.get("role"), content=delta_data.get("content"),
                        )
                        stream_choice = ChatCompletionStreamChoice(
                            index=choice_data.get("index", 0), delta=delta, finish_reason=finish_reason
                        )
                        choices.append(stream_choice)
                    logger.debug(f"[{request_id}] Parsed choices for chunk {i}: {choices!r}")
                else:
                    logger.warning(f"[{request_id}] Chunk {i} had no valid choices list: {output_chunk!r}")
                    choices = []

                # --- Yield Chunk Payload ---
                if choices:
                    chunk = ChatCompletionStreamResponse(
                        id=request_id, object="chat.completion.chunk", created=completion_start_time,
                        model=model_id, choices=choices, usage=None,
                        system_fingerprint=output_chunk.get("system_fingerprint")
                    )
                    json_payload = chunk.model_dump_json(exclude_unset=True)
                    logger.info(f"[{request_id}] >>> Yielding SSE chunk {i} payload: {json_payload}")
                    # CORRECTED YIELD: Yield only the JSON string payload
                    yield json_payload
                    logger.debug(f"[{request_id}] Yielded SSE chunk {i} successfully.")
                else:
                    logger.warning(f"[{request_id}] Did not yield chunk {i} (no choices parsed): {output_chunk!r}")

                i += 1
            # --- except GeneratorExit / Exception inside loop ---
            except GeneratorExit:
                 logger.warning(f"[{request_id}] GeneratorExit caught mid-stream during chunk {i}...")
                 exited_prematurely = True
                 return
            except Exception as e:
                # ... error logging ...
                exited_prematurely = True
                return
        # --- End of loop ---
        logger.info(f"[{request_id}] Stream loop finished normally after {i} chunks.")

    # --- except GeneratorExit / Exception outside loop ---
    except GeneratorExit:
         logger.warning(f"[{request_id}] GeneratorExit caught outside the main chunk processing loop...")
         exited_prematurely = True
         return
    except Exception as e:
        # ... error logging ...
        exited_prematurely = True

    # --- Finally Block ---
    finally:
        if not exited_prematurely:
            logger.info(f"[{request_id}] >>> Yielding [DONE]")
            try:
                # CORRECTED YIELD: Yield the special string for DONE
                # sse-starlette might have specific handling for None or an object,
                # but yielding the string "[DONE]" is usually interpreted correctly by clients
                # when expecting the OpenAI format. Check sse-starlette docs if needed.
                # Let's try yielding the string directly first.
                yield "[DONE]"
                logger.debug(f"[{request_id}] Yielded [DONE] successfully.")
            except GeneratorExit:
                 logger.warning(f"[{request_id}] GeneratorExit caught while yielding [DONE]...")
            except Exception as e:
                 logger.error(f"[{request_id}] Error yielding [DONE]: {e}", exc_info=True)
        else:
             logger.info(f"[{request_id}] Suppressing [DONE] yield because generator exited prematurely.")
             

# --- Modified Endpoint ---
@router.post(
    "/chat/completions",
    response_model=ChatCompletionResponse, # For non-streaming
    dependencies=[Depends(get_authenticated_user)] # Apply authentication dependency
)
async def create_chat_completion(
    request: ChatCompletionRequest, # Request body is parsed here
):
    """
    Creates a chat completion for the provided messages and parameters.
    Supports both streaming and non-streaming responses.
    """
    model_id = validate_model_id(request.model)
    llama = get_model_instance_from_id(model_id)

    if request.n is not None and request.n > 1:
         print(f"Warning: Requesting n={request.n} choices...")
    if not request.messages or len(request.messages) == 0:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Messages list cannot be empty.")

    try:
        llama_params = _map_chat_request_to_llama_params(request)

        if request.stream:
            request_id = generate_id("chatcmpl")
            return EventSourceResponse(
                _chat_completion_stream_generator(model_id, llama, llama_params, request_id),
                media_type="text/event-stream"
            )
        else:
            output = llama.create_chat_completion(**llama_params, stream=False)
            response = _parse_chat_completion_output(output, model_id, request)
            return response

    except ValueError as e:
         raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Invalid request parameters: {e}")
    except Exception as e:
        logger.error(f"Error creating chat completion for model '{model_id}': {e}", exc_info=True) # Log with traceback
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error during model inference: {str(e)}"
        )       