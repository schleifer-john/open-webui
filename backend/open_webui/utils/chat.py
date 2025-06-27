import time
import logging
import sys

from aiocache import cached
from typing import Any, Optional
import random
import json
import inspect
import uuid
import asyncio

from fastapi import Request, status
from starlette.responses import Response, StreamingResponse, JSONResponse


from open_webui.models.users import UserModel

from open_webui.socket.main import (
    sio,
    get_event_call,
    get_event_emitter,
)
from open_webui.functions import generate_function_chat_completion

from open_webui.routers.openai import (
    generate_chat_completion as generate_openai_chat_completion,
)

from open_webui.routers.ollama import (
    generate_chat_completion as generate_ollama_chat_completion,
)

from open_webui.routers.myagent import (
    generate_chat_completion as generate_myagent_chat_completion,
)

from open_webui.routers.pipelines import (
    process_pipeline_inlet_filter,
    process_pipeline_outlet_filter,
)

from open_webui.models.functions import Functions
from open_webui.models.models import Models


from open_webui.utils.plugin import load_function_module_by_id
from open_webui.utils.models import get_all_models, check_model_access
from open_webui.utils.payload import convert_payload_openai_to_ollama
from open_webui.utils.response import (
    convert_response_ollama_to_openai,
    convert_streaming_response_ollama_to_openai,
)
from open_webui.utils.filter import (
    get_sorted_filter_ids,
    process_filter_functions,
)

from open_webui.env import SRC_LOG_LEVELS, GLOBAL_LOG_LEVEL, BYPASS_MODEL_ACCESS_CONTROL


logging.basicConfig(stream=sys.stdout, level=GLOBAL_LOG_LEVEL)
log = logging.getLogger(__name__)
log.setLevel(SRC_LOG_LEVELS["MAIN"])


async def generate_direct_chat_completion(
    request: Request,
    form_data: dict,
    user: Any,
    models: dict,
):
    print("generate_direct_chat_completion called")
    print(f"Request object: {request}")
    print(f"Form data: {form_data}")
    print(f"User: {user}")
    print(f"Models: {models}")
    log.info("generate_direct_chat_completion")

    metadata = form_data.pop("metadata", {})
    print(f"Metadata extracted: {metadata}")

    user_id = metadata.get("user_id")
    session_id = metadata.get("session_id")
    request_id = str(uuid.uuid4())  # Generate a unique request ID

    print(f"user_id: {user_id}, session_id: {session_id}, request_id: {request_id}")

    event_caller = get_event_call(metadata)

    channel = f"{user_id}:{session_id}:{request_id}"
    print(f"Constructed channel: {channel}")

    if form_data.get("stream"):
        print("Streaming is enabled")
        q = asyncio.Queue()

        async def message_listener(sid, data):
            """
            Handle received socket messages and push them into the queue.
            """
            print(f"Received data on channel {channel}: {data}")
            await q.put(data)

        # Register the listener
        sio.on(channel, message_listener)

        # Start processing chat completion in background
        res = await event_caller(
            {
                "type": "request:chat:completion",
                "data": {
                    "form_data": form_data,
                    "model": models[form_data["model"]],
                    "channel": channel,
                    "session_id": session_id,
                },
            }
        )
        print(f"Response from event_caller: {res}")
        log.info(f"res: {res}")

        if res.get("status", False):
            # Define a generator to stream responses
            async def event_generator():
                nonlocal q
                try:
                    while True:
                        data = await q.get()  # Wait for new messages
                        print(f"Streaming data: {data}")
                        if isinstance(data, dict):
                            if "done" in data and data["done"]:
                                print("Received 'done' in streaming data")
                                break  # Stop streaming when 'done' is received

                            yield f"data: {json.dumps(data)}\n\n"
                        elif isinstance(data, str):
                            yield data
                except Exception as e:
                    log.debug(f"Error in event generator: {e}")
                    print(f"Error in event generator: {e}")
                    pass

            # Define a background task to run the event generator
            async def background():
                try:
                    del sio.handlers["/"][channel]
                    print(f"Cleaned up socket handler for channel {channel}")
                except Exception as e:
                    print(f"Error while cleaning up handler: {e}")
                    pass

            # Return the streaming response
            response = StreamingResponse(
                event_generator(), media_type="text/event-stream", background=background
            )
            print(f"StreamingResponse: {response}")
            return response
        else:
            print(f"Exception raised with response: {res}")
            raise Exception(str(res))
    else:
        res = await event_caller(
            {
                "type": "request:chat:completion",
                "data": {
                    "form_data": form_data,
                    "model": models[form_data["model"]],
                    "channel": channel,
                    "session_id": session_id,
                },
            }
        )
        print(f"Response from event_caller: {res}")
        if "error" in res and res["error"]:
            print(f"Raising exception with error: {res['error']}")
            raise Exception(res["error"])

        return res


async def generate_chat_completion(
    request: Request,
    form_data: dict,
    user: Any,
    bypass_filter: bool = False,
):
    print("generate_chat_completion called")
    print(f"Request method: {request.method}")
    print(f"Request url: {request.url}")
    print(f"Request headers: {dict(request.headers)}")

    # Read and print the body safely
    body_bytes = await request.body()
    print(f"Request body bytes: {body_bytes}")
    try:
        body_str = body_bytes.decode("utf-8")
        print(f"Request body (decoded): {body_str}")
    except Exception as e:
        print(f"Could not decode request body: {e}")
    
    print(f"Initial form_data: {form_data}")
    print(f"User: {user}")
    print(f"bypass_filter: {bypass_filter}")
    log.debug(f"generate_chat_completion: {form_data}")

    if BYPASS_MODEL_ACCESS_CONTROL:
        bypass_filter = True
        print("Bypass model access control enabled, setting bypass_filter=True")

    if hasattr(request.state, "metadata"):
        print(f"request.state.metadata found: {request.state.metadata}")
        if "metadata" not in form_data:
            form_data["metadata"] = request.state.metadata
            print(f"Added request.state.metadata to form_data.metadata: {form_data['metadata']}")
        else:
            form_data["metadata"] = {
                **form_data["metadata"],
                **request.state.metadata,
            }
            print(f"Merged request.state.metadata into form_data.metadata: {form_data['metadata']}")

    if getattr(request.state, "direct", False) and hasattr(request.state, "model"):
        models = {
            request.state.model["id"]: request.state.model,
        }
        print(f"Direct connection to model(s): {models}")
        log.debug(f"direct connection to model: {models}")
    else:
        models = request.app.state.MODELS
        print(f"Using app models: {list(models.keys())}")

    model_id = form_data["model"]
    print(f"Requested model_id: {model_id}")
    if model_id not in models:
        print(f"Model {model_id} not found in models")
        raise Exception("Model not found")

    model = models[model_id]
    print(f"Selected model: {model}")

    if getattr(request.state, "direct", False):
        print("Calling generate_direct_chat_completion because direct flag is set")
        response = await generate_direct_chat_completion(
            request, form_data, user=user, models=models
        )
        print(f"Response from generate_direct_chat_completion: {response}")
        return response
    else:
        if not bypass_filter and user.role == "user":
            print(f"Checking model access for user with role: {user.role}")
            try:
                check_model_access(user, model)
                print("User access to model verified")
            except Exception as e:
                print(f"Access check failed: {e}")
                raise e

        if model.get("owned_by") == "arena":
            print("Arena model detected")
            model_ids = model.get("info", {}).get("meta", {}).get("model_ids")
            filter_mode = model.get("info", {}).get("meta", {}).get("filter_mode")
            print(f"model_ids: {model_ids}, filter_mode: {filter_mode}")

            if model_ids and filter_mode == "exclude":
                model_ids = [
                    model["id"]
                    for model in list(request.app.state.MODELS.values())
                    if model.get("owned_by") != "arena" and model["id"] not in model_ids
                ]
                print(f"Filtered model_ids due to exclude filter_mode: {model_ids}")

            selected_model_id = None
            if isinstance(model_ids, list) and model_ids:
                selected_model_id = random.choice(model_ids)
                print(f"Selected model_id from filtered list: {selected_model_id}")
            else:
                model_ids = [
                    model["id"]
                    for model in list(request.app.state.MODELS.values())
                    if model.get("owned_by") != "arena"
                ]
                selected_model_id = random.choice(model_ids)
                print(f"Selected model_id from non-arena models: {selected_model_id}")

            form_data["model"] = selected_model_id
            print(f"Updated form_data['model'] to: {selected_model_id}")

            if form_data.get("stream") == True:
                print("Stream is True, defining stream_wrapper and calling recursively with bypass_filter=True")

                async def stream_wrapper(stream):
                    print(f"Sending selected_model_id: {selected_model_id} to client")
                    yield f"data: {json.dumps({'selected_model_id': selected_model_id})}\n\n"
                    async for chunk in stream:
                        print(f"Streaming chunk: {chunk}")
                        yield chunk

                response = await generate_chat_completion(
                    request, form_data, user, bypass_filter=True
                )
                print(f"Response from nested generate_chat_completion: {response}")
                return StreamingResponse(
                    stream_wrapper(response.body_iterator),
                    media_type="text/event-stream",
                    background=response.background,
                )
            else:
                print("Stream is False, calling recursively and returning response with selected_model_id added")
                response = await generate_chat_completion(
                    request, form_data, user, bypass_filter=True
                )
                print(f"Response from nested generate_chat_completion: {response}")
                return {
                    **response,
                    "selected_model_id": selected_model_id,
                }

        if model.get("pipe"):
            print("Model has 'pipe', calling generate_function_chat_completion")
            response = await generate_function_chat_completion(
                request, form_data, user=user, models=models
            )
            print(f"Response from generate_function_chat_completion: {response}")
            return response

        if model.get("owned_by") == "ollama":
            print("Model owned by 'ollama', converting payload and calling generate_ollama_chat_completion")
            form_data = convert_payload_openai_to_ollama(form_data)
            response = await generate_ollama_chat_completion(
                request=request,
                form_data=form_data,
                user=user,
                bypass_filter=bypass_filter,
            )
            print(f"Raw response from generate_ollama_chat_completion: {response}")
            if form_data.get("stream"):
                print("Stream is True for ollama model, returning StreamingResponse")
                response.headers["content-type"] = "text/event-stream"
                return StreamingResponse(
                    convert_streaming_response_ollama_to_openai(response),
                    headers=dict(response.headers),
                    background=response.background,
                )
            else:
                print("Stream is False for ollama model, returning converted response")
                converted = convert_response_ollama_to_openai(response)
                print(f"Converted response: {converted}")
                return converted
        elif model.get("id") == "my-agent-id":
            print("Calling generate_myagent_chat_completion")
            response = await generate_myagent_chat_completion(
                request=request,
                form_data=form_data,
                user=user,
            )
            print(f"Response from generate_myagent_chat_completion: {response}")
            return response
        else:
            print("Calling generate_openai_chat_completion as default")
            response = await generate_openai_chat_completion(
                request=request,
                form_data=form_data,
                user=user,
                bypass_filter=bypass_filter,
            )
            print(f"Response from generate_openai_chat_completion: {response}")
            return response



chat_completion = generate_chat_completion


async def chat_completed(request: Request, form_data: dict, user: Any):
    print(f"chat_completed called")
    print(f"Request object: {request}")
    print(f"Initial form_data: {form_data}")
    print(f"User: {user}")

    if not request.app.state.MODELS:
        print("No models in app state, calling get_all_models()")
        await get_all_models(request, user=user)

    if getattr(request.state, "direct", False) and hasattr(request.state, "model"):
        models = {
            request.state.model["id"]: request.state.model,
        }
        print(f"Using direct model: {models}")
    else:
        models = request.app.state.MODELS
        print(f"Using app state models: {list(models.keys())}")

    data = form_data
    model_id = data["model"]
    print(f"Requested model_id: {model_id}")
    if model_id not in models:
        print(f"Model {model_id} not found")
        raise Exception("Model not found")

    model = models[model_id]
    print(f"Selected model: {model}")

    try:
        data = await process_pipeline_outlet_filter(request, data, user, models)
        print(f"Data after outlet filter: {data}")
    except Exception as e:
        print(f"Error in process_pipeline_outlet_filter: {e}")
        return Exception(f"Error: {e}")

    metadata = {
        "chat_id": data["chat_id"],
        "message_id": data["id"],
        "filter_ids": data.get("filter_ids", []),
        "session_id": data["session_id"],
        "user_id": user.id,
    }
    print(f"Metadata: {metadata}")

    extra_params = {
        "__event_emitter__": get_event_emitter(metadata),
        "__event_call__": get_event_call(metadata),
        "__user__": {
            "id": user.id,
            "email": user.email,
            "name": user.name,
            "role": user.role,
        },
        "__metadata__": metadata,
        "__request__": request,
        "__model__": model,
    }

    try:
        filter_functions = [
            Functions.get_function_by_id(filter_id)
            for filter_id in get_sorted_filter_ids(
                request, model, metadata.get("filter_ids", [])
            )
        ]
        print(f"Filter functions: {filter_functions}")

        result, _ = await process_filter_functions(
            request=request,
            filter_functions=filter_functions,
            filter_type="outlet",
            form_data=data,
            extra_params=extra_params,
        )
        print(f"Result from process_filter_functions: {result}")
        return result
    except Exception as e:
        print(f"Error in process_filter_functions: {e}")
        return Exception(f"Error: {e}")



async def chat_action(request: Request, action_id: str, form_data: dict, user: Any):
    print(f"chat_action called")
    print(f"Request: {request}")
    print(f"Action ID: {action_id}")
    print(f"Form Data: {form_data}")
    print(f"User: {user}")

    if "." in action_id:
        action_id, sub_action_id = action_id.split(".")
    else:
        sub_action_id = None
    print(f"Parsed action_id: {action_id}, sub_action_id: {sub_action_id}")

    action = Functions.get_function_by_id(action_id)
    if not action:
        print(f"Action {action_id} not found")
        raise Exception(f"Action not found: {action_id}")

    if not request.app.state.MODELS:
        print("No models loaded, calling get_all_models()")
        await get_all_models(request, user=user)

    if getattr(request.state, "direct", False) and hasattr(request.state, "model"):
        models = {
            request.state.model["id"]: request.state.model,
        }
        print(f"Using direct model: {models}")
    else:
        models = request.app.state.MODELS
        print(f"Using app state models: {list(models.keys())}")

    data = form_data
    model_id = data["model"]
    print(f"Requested model_id: {model_id}")
    if model_id not in models:
        print(f"Model {model_id} not found")
        raise Exception("Model not found")
    model = models[model_id]
    print(f"Selected model: {model}")

    __event_emitter__ = get_event_emitter(
        {
            "chat_id": data["chat_id"],
            "message_id": data["id"],
            "session_id": data["session_id"],
            "user_id": user.id,
        }
    )
    __event_call__ = get_event_call(
        {
            "chat_id": data["chat_id"],
            "message_id": data["id"],
            "session_id": data["session_id"],
            "user_id": user.id,
        }
    )

    if action_id in request.app.state.FUNCTIONS:
        function_module = request.app.state.FUNCTIONS[action_id]
        print(f"Loaded function_module from cache for action_id {action_id}")
    else:
        function_module, _, _ = load_function_module_by_id(action_id)
        request.app.state.FUNCTIONS[action_id] = function_module
        print(f"Dynamically loaded function_module for action_id {action_id}")

    if hasattr(function_module, "valves") and hasattr(function_module, "Valves"):
        valves = Functions.get_function_valves_by_id(action_id)
        function_module.valves = function_module.Valves(**(valves if valves else {}))
        print(f"Initialized valves: {valves}")

    if hasattr(function_module, "action"):
        try:
            action = function_module.action
            sig = inspect.signature(action)
            print(f"Function signature: {sig}")

            params = {"body": data}
            extra_params = {
                "__model__": model,
                "__id__": sub_action_id if sub_action_id is not None else action_id,
                "__event_emitter__": __event_emitter__,
                "__event_call__": __event_call__,
                "__request__": request,
            }

            for key, value in extra_params.items():
                if key in sig.parameters:
                    params[key] = value

            if "__user__" in sig.parameters:
                __user__ = {
                    "id": user.id,
                    "email": user.email,
                    "name": user.name,
                    "role": user.role,
                }

                try:
                    if hasattr(function_module, "UserValves"):
                        __user__["valves"] = function_module.UserValves(
                            **Functions.get_user_valves_by_id_and_user_id(
                                action_id, user.id
                            )
                        )
                        print(f"Loaded user valves for {user.id}")
                except Exception as e:
                    log.exception(f"Failed to get user valves: {e}")

                params["__user__"] = __user__

            print(f"Final params to action: {params.keys()}")

            if inspect.iscoroutinefunction(action):
                data = await action(**params)
            else:
                data = action(**params)

            print(f"Action result: {data}")

        except Exception as e:
            print(f"Error invoking function action: {e}")
            return Exception(f"Error: {e}")

    return data

