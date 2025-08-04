# llm_server/api/v1/chat_completions.py
import time
import json
import base64 # <-- Import base64
import re     # <-- Import re
from asyncio import anext
from fastapi import APIRouter, Depends, HTTPException, status, Request
from sse_starlette.sse import EventSourceResponse
from typing import AsyncGenerator, List, Dict, Any, Union, Optional # <-- Add Dict, Any, Union, Optional

from llm_server.schemas.chat import (
    ChatCompletionRequest, ChatMessage, TextContentPart, ImageContentPart, # <-- Import new schemas
    ChatCompletionResponse, ChatCompletionChoice, ChatCompletionMessage,
    ChatCompletionStreamResponse, ChatCompletionStreamChoice, ChatCompletionStreamDelta
)
from llm_server.schemas.common import UsageInfo
from llm_server.api.deps import get_authenticated_user, validate_model_id
from config.settings import get_model_config, ModelInfo
from llm_server.core.utils import generate_id
from llm_server.core.llm_manager import dynamic_model_manager
from llama_cpp import Llama

# --- Add logging ---
import logging
import traceback
logger = logging.getLogger(__name__)
# ---

router = APIRouter(tags=["Chat"])

# --- Helper function to parse data URI ---
def parse_data_uri(uri: str) -> Optional[str]:
    """Extracts base64 data from a data URI if valid, otherwise returns None."""
    match = re.match(r"data:image/(?:jpeg|png|gif|webp);base64,(.*)", uri)
    if match:
        return match.group(1)
    return None

def _map_chat_request_to_llama_params(request: ChatCompletionRequest, model_config: ModelInfo) -> dict:
    """Maps OpenAI Chat request to llama-cpp-python, specifically for Gemma 3 multimodal format."""
    llama_messages = []
    contains_images = False

    for message in request.messages:
        role = message.role
        content = message.content

        # --- Handle string content (convert to list format) ---
        if isinstance(content, str):
            processed_content = [{"type": "text", "text": content}]
        # --- Handle list content (text/image parts) ---
        elif isinstance(content, list):
            processed_content = []
            for part in content:
                part_type = part.type
                if part_type == "text":
                    processed_content.append({"type": "text", "text": part.text})
                elif part_type == "image_url":
                    contains_images = True
                    if not model_config.is_multimodal:
                        raise HTTPException(
                            status_code=status.HTTP_400_BAD_REQUEST,
                            detail=f"Model '{model_config.model_id}' does not support image inputs."
                        )

                    image_url_data = part.image_url
                    url = image_url_data.url

                    if url.startswith("data:image/"):
                        base64_data = parse_data_uri(url)
                        if not base64_data:
                             raise HTTPException(
                                status_code=status.HTTP_400_BAD_REQUEST,
                                detail=f"Invalid image data URI format for role '{role}'.",
                            )
                        # --- >>> GEMMA 3 CHANGE HERE <<< ---
                        # Gemma 3 template expects {'type': 'image', 'image_url': base64_string}
                        # We pass the base64 string directly under 'image_url' key, paired with type 'image'.
                        # llama-cpp-python internal logic should pick this up.
                        processed_content.append({
                            "type": "image",         # <--- Use 'image'
                            "image_url": base64_data # <--- Pass base64 directly
                        })
                        # --- >>> END GEMMA 3 CHANGE <<< ---
                    else: # http/https URLs still not supported
                        raise HTTPException(
                            status_code=status.HTTP_400_BAD_REQUEST,
                            detail=f"Image URLs ('{url[:30]}...') are not supported. Use base64 data URIs.",
                        )
                else:
                     raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Unsupported content part type '{part_type}'")
        else:
             raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Invalid content type for message role '{role}'")

        llama_messages.append({"role": role, "content": processed_content})

    # --- Assemble final parameters ---
    # (The rest of the parameter assembly remains the same)
    params = {
        "messages": llama_messages,
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

# ... (rest of the file: _parse_chat_completion_output, _chat_completion_stream_generator, create_chat_completion endpoint) ...
# Make sure the create_chat_completion endpoint still calls the modified _map_chat_request_to_llama_params

# --- Keep _parse_chat_completion_output the same ---
# (Assuming model output is still text)
def _parse_chat_completion_output(output: dict, model_id: str, request: ChatCompletionRequest) -> ChatCompletionResponse:
    # ... (implementation remains the same as before) ...
    choices = []
    for choice_data in output.get("choices", []):
        message_data = choice_data.get("message", {})
        choices.append(
            ChatCompletionChoice(
                index=choice_data.get("index", 0),
                message=ChatCompletionMessage( # Response message is still text
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


async def _chat_completion_stream_generator(
    model_id: str,
    stream: AsyncGenerator[Dict, None],
    request_id: str
) -> AsyncGenerator[str, None]:
    """
    Yields formatted server-sent events from an async stream of chunks.
    """
    completion_start_time = int(time.time())
    logger.debug(f"[{request_id}] Stream generator started.")

    try:
        i = 0
        async for output_chunk in stream:
            # The worker process now sends parsed dictionaries.
            # We just need to format them into the SSE response schema.
            if i == 0:
                logger.info(f"[{request_id}] First stream chunk received: {output_chunk!r}")

            choices = []
            if "choices" in output_chunk and isinstance(output_chunk["choices"], list):
                for choice_data in output_chunk["choices"]:
                    delta_data = choice_data.get("delta", {})
                    delta = ChatCompletionStreamDelta(
                        role=delta_data.get("role"),
                        content=delta_data.get("content")
                    )
                    stream_choice = ChatCompletionStreamChoice(
                        index=choice_data.get("index", 0),
                        delta=delta,
                        finish_reason=choice_data.get("finish_reason")
                    )
                    choices.append(stream_choice)

            if not choices:
                continue

            chunk_response = ChatCompletionStreamResponse(
                id=request_id,
                object="chat.completion.chunk",
                created=completion_start_time,
                model=model_id,
                choices=choices,
                system_fingerprint=output_chunk.get("system_fingerprint")
            )
            json_payload = chunk_response.model_dump_json(exclude_unset=True)
            yield json_payload
            i += 1

    except Exception as e:
        tb_str = traceback.format_exc()
        logger.error(f"[{request_id}] Error in stream generator: {e}\n{tb_str}")
        # Yield an error message if possible
        error_payload = {"error": f"An error occurred during streaming: {e}"}
        yield json.dumps(error_payload)
    finally:
        # Signal the end of the stream to the client
        yield "[DONE]"
        logger.info(f"[{request_id}] Stream generator finished, sent [DONE].")

@router.post(
    "/chat/completions",
    response_model=ChatCompletionResponse,
    dependencies=[Depends(get_authenticated_user)]
)
async def create_chat_completion(
    request: ChatCompletionRequest,
):
    """
    Creates a chat completion by dynamically spawning a model worker process.
    """
    model_config = get_model_config(request.model)
    if not model_config:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model configuration not found for '{request.model}'."
        )

    model_id = validate_model_id(request.model)

    if not request.messages:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Messages list cannot be empty.")

    try:
        llama_params = _map_chat_request_to_llama_params(request, model_config)
        llama_params["stream"] = request.stream

        # The dynamic manager handles the entire lifecycle of the request
        stream = dynamic_model_manager.handle_inference_request(
            model_id=model_id,
            llama_params=llama_params
        )

        if request.stream:
            request_id = generate_id("chatcmpl")
            return EventSourceResponse(
                _chat_completion_stream_generator(model_id, stream, request_id),
                media_type="text/event-stream"
            )
        else:
            # For non-streaming, get the single result from the async generator
            output = await anext(stream)
            if "error" in output:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Error from model worker: {output['error']}"
                )
            response = _parse_chat_completion_output(output, model_id, request)
            return response

    except HTTPException as e:
        # Re-raise specific HTTP exceptions from mapping function
        raise e
    except ValueError as e:
         # Catch potential value errors from llama-cpp-python itself
         logger.error(f"ValueError during chat completion for '{model_id}': {e}", exc_info=True)
         raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Invalid request parameters: {e}")
    except Exception as e:
        logger.error(f"Error creating chat completion for model '{model_id}': {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error during model inference: {str(e)}"
        )