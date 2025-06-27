import json
from uuid import uuid4
from open_webui.utils.misc import (
    openai_chat_chunk_message_template,
    openai_chat_completion_message_template,
)


def convert_ollama_tool_call_to_openai(tool_calls: dict) -> dict:
    print(f"[convert_ollama_tool_call_to_openai] Received tool_calls: {tool_calls}")
    openai_tool_calls = []
    for tool_call in tool_calls:
        print(f"[convert_ollama_tool_call_to_openai] Processing tool_call: {tool_call}")
        openai_tool_call = {
            "index": tool_call.get("index", 0),
            "id": tool_call.get("id", f"call_{str(uuid4())}"),
            "type": "function",
            "function": {
                "name": tool_call.get("function", {}).get("name", ""),
                "arguments": json.dumps(
                    tool_call.get("function", {}).get("arguments", {})
                ),
            },
        }
        print(f"[convert_ollama_tool_call_to_openai] Converted tool_call: {openai_tool_call}")
        openai_tool_calls.append(openai_tool_call)
    print(f"[convert_ollama_tool_call_to_openai] Final openai_tool_calls: {openai_tool_calls}")
    return openai_tool_calls


def convert_ollama_usage_to_openai(data: dict) -> dict:
    print(f"[convert_ollama_usage_to_openai] Received data: {data}")
    result = {
        "response_token/s": (
            round(
                (
                    (
                        data.get("eval_count", 0)
                        / ((data.get("eval_duration", 0) / 10_000_000))
                    )
                    * 100
                ),
                2,
            )
            if data.get("eval_duration", 0) > 0
            else "N/A"
        ),
        "prompt_token/s": (
            round(
                (
                    (
                        data.get("prompt_eval_count", 0)
                        / ((data.get("prompt_eval_duration", 0) / 10_000_000))
                    )
                    * 100
                ),
                2,
            )
            if data.get("prompt_eval_duration", 0) > 0
            else "N/A"
        ),
        "total_duration": data.get("total_duration", 0),
        "load_duration": data.get("load_duration", 0),
        "prompt_eval_count": data.get("prompt_eval_count", 0),
        "prompt_tokens": int(
            data.get("prompt_eval_count", 0)
        ),  # This is the OpenAI compatible key
        "prompt_eval_duration": data.get("prompt_eval_duration", 0),
        "eval_count": data.get("eval_count", 0),
        "completion_tokens": int(
            data.get("eval_count", 0)
        ),  # This is the OpenAI compatible key
        "eval_duration": data.get("eval_duration", 0),
        "approximate_total": (lambda s: f"{s // 3600}h{(s % 3600) // 60}m{s % 60}s")(
            (data.get("total_duration", 0) or 0) // 1_000_000_000
        ),
        "total_tokens": int(  # This is the OpenAI compatible key
            data.get("prompt_eval_count", 0) + data.get("eval_count", 0)
        ),
        "completion_tokens_details": {  # This is the OpenAI compatible key
            "reasoning_tokens": 0,
            "accepted_prediction_tokens": 0,
            "rejected_prediction_tokens": 0,
        },
    }
    print(f"[convert_ollama_usage_to_openai] Converted usage data: {result}")
    return result


def convert_response_ollama_to_openai(ollama_response: dict) -> dict:
    print(f"[convert_response_ollama_to_openai] Received response: {ollama_response}")
    model = ollama_response.get("model", "ollama")
    message_content = ollama_response.get("message", {}).get("content", "")
    tool_calls = ollama_response.get("message", {}).get("tool_calls", None)
    openai_tool_calls = None

    if tool_calls:
        print(f"[convert_response_ollama_to_openai] Found tool_calls: {tool_calls}")
        openai_tool_calls = convert_ollama_tool_call_to_openai(tool_calls)

    data = ollama_response

    usage = convert_ollama_usage_to_openai(data)

    response = openai_chat_completion_message_template(
        model, message_content, openai_tool_calls, usage
    )
    print(f"[convert_response_ollama_to_openai] Final response: {response}")
    return response


async def convert_streaming_response_ollama_to_openai(ollama_streaming_response):
    print("[convert_streaming_response_ollama_to_openai] Starting stream conversion")
    async for data in ollama_streaming_response.body_iterator:
        print(f"[convert_streaming_response_ollama_to_openai] Raw stream data: {data}")
        data = json.loads(data)
        print(f"[convert_streaming_response_ollama_to_openai] Parsed JSON data: {data}")

        model = data.get("model", "ollama")
        message_content = data.get("message", {}).get("content", None)
        tool_calls = data.get("message", {}).get("tool_calls", None)
        openai_tool_calls = None

        if tool_calls:
            print(f"[convert_streaming_response_ollama_to_openai] Found tool_calls: {tool_calls}")
            openai_tool_calls = convert_ollama_tool_call_to_openai(tool_calls)

        done = data.get("done", False)
        print(f"[convert_streaming_response_ollama_to_openai] done: {done}")

        usage = None
        if done:
            print("[convert_streaming_response_ollama_to_openai] Final chunk, converting usage")
            usage = convert_ollama_usage_to_openai(data)

        data = openai_chat_chunk_message_template(
            model, message_content, openai_tool_calls, usage
        )

        line = f"data: {json.dumps(data)}\n\n"
        print(f"[convert_streaming_response_ollama_to_openai] Yielding line: {line.strip()}")
        yield line

    print("[convert_streaming_response_ollama_to_openai] Stream complete, yielding [DONE]")
    yield "data: [DONE]\n\n"
