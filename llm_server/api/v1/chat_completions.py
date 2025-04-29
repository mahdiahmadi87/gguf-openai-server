# llm_server/api/v1/chat_completions.py
import time
import json
import base64 # <-- Import base64
import re     # <-- Import re
from fastapi import APIRouter, Depends, HTTPException, status, Request
from sse_starlette.sse import EventSourceResponse
from typing import AsyncGenerator, List, Dict, Any, Union, Optional # <-- Add Dict, Any, Union, Optional

from llm_server.schemas.chat import (
    ChatCompletionRequest, ChatMessage, TextContentPart, ImageContentPart, # <-- Import new schemas
    ChatCompletionResponse, ChatCompletionChoice, ChatCompletionMessage,
    ChatCompletionStreamResponse, ChatCompletionStreamChoice, ChatCompletionStreamDelta
)
from llm_server.schemas.common import UsageInfo
from llm_server.api.deps import get_authenticated_user, validate_model_id, get_model_instance_from_id
# --- Import get_model_config from settings ---
from config.settings import get_model_config, ModelInfo
from llm_server.core.utils import generate_id
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


# --- Keep _chat_completion_stream_generator the same ---
# (It yields JSON payloads based on llama-cpp-python output, which should still be text deltas)
async def _chat_completion_stream_generator(
    model_id: str,
    llama_instance: Llama,
    llama_params: dict,
    request_id: str
) -> AsyncGenerator[str, None]:
     # ... (implementation remains the same, including preamble and GeneratorExit handling) ...
    completion_start_time = int(time.time())
    logger.debug(f"[{request_id}] Chat stream generator started.")
    exited_prematurely = False # Flag to track if GeneratorExit was caught

    # --- PREAMBLE ---
    initial_chunk = ChatCompletionStreamResponse(
        id=request_id, object="chat.completion.chunk", created=completion_start_time, model=model_id,
        choices=[ChatCompletionStreamChoice(index=0, delta={}, finish_reason=None)]
    )
    initial_payload = initial_chunk.model_dump_json(exclude_unset=True)
    logger.info(f"[{request_id}] Yielding PREAMBLE SSE chunk payload: {initial_payload}")
    try:
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
                choices = []
                raw_choices = output_chunk.get("choices") if isinstance(output_chunk, dict) else None
                if isinstance(raw_choices, list):
                    for choice_data in raw_choices:
                        # Parsing logic remains the same as before (expects text delta)
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
                    yield json_payload
                    logger.debug(f"[{request_id}] Yielded SSE chunk {i} successfully.")
                else:
                    logger.warning(f"[{request_id}] Did not yield chunk {i} (no choices parsed): {output_chunk!r}")


                i += 1
            except GeneratorExit:
                 logger.warning(f"[{request_id}] GeneratorExit caught mid-stream during chunk {i}...")
                 exited_prematurely = True
                 return
            except Exception as e:
                tb_str = traceback.format_exc()
                logger.error(f"[{request_id}] Error processing chunk {i}: {e}\nTRACEBACK:\n{tb_str}")
                exited_prematurely = True
                return
        # --- End of loop ---
        logger.info(f"[{request_id}] Stream loop finished normally after {i} chunks.")

    except GeneratorExit:
         logger.warning(f"[{request_id}] GeneratorExit caught outside the main chunk processing loop...")
         exited_prematurely = True
         return
    except Exception as e:
        tb_str = traceback.format_exc()
        logger.error(f"[{request_id}] Error creating or iterating llama_cpp stream: {e}\nTRACEBACK:\n{tb_str}")
        exited_prematurely = True

    # --- Finally Block ---
    finally:
        if not exited_prematurely:
            logger.info(f"[{request_id}] >>> Yielding [DONE]")
            try:
                yield "[DONE]"
                logger.debug(f"[{request_id}] Yielded [DONE] successfully.")
            except GeneratorExit:
                 logger.warning(f"[{request_id}] GeneratorExit caught while yielding [DONE]...")
            except Exception as e:
                 logger.error(f"[{request_id}] Error yielding [DONE]: {e}", exc_info=True)
        else:
             logger.info(f"[{request_id}] Suppressing [DONE] yield because generator exited prematurely.")


# --- Updated Endpoint ---
@router.post(
    "/chat/completions",
    response_model=ChatCompletionResponse,
    dependencies=[Depends(get_authenticated_user)]
)
async def create_chat_completion(
    request: ChatCompletionRequest, # Uses updated schema
    # request_object: Request # Optional
):
    """Creates a chat completion, handling text and image inputs."""
    # --- Get model config FIRST to check multimodal support ---
    model_config = get_model_config(request.model)
    if not model_config:
         # Should be caught by validate_model_id called implicitly before,
         # but check again for safety before accessing is_multimodal
          raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model configuration not found for '{request.model}'."
        )

    # --- Now validate ID and get instance ---
    model_id = validate_model_id(request.model) # Still useful validation
    llama = get_model_instance_from_id(model_id) # Gets the instance

    # --- Request parameter validation ---
    if request.n is not None and request.n > 1:
         print(f"Warning: Requesting n={request.n} choices...")
    if not request.messages or len(request.messages) == 0:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Messages list cannot be empty.")

    try:
        # --- Map request, passing model_config for multimodal checks ---
        llama_params = _map_chat_request_to_llama_params(request, model_config)

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