# Router class to route chat requests to a custom agent
import logging
import json
import aiohttp

from typing import Optional, Union

from fastapi import (
    Depends,
    FastAPI,
    HTTPException,
    Request,
    APIRouter
)

from pydantic import BaseModel, validator

from open_webui.models.models import Models
from open_webui.utils.auth import get_verified_user
from open_webui.env import (
    AIOHTTP_CLIENT_SESSION_SSL,
    AIOHTTP_CLIENT_TIMEOUT,
    SRC_LOG_LEVELS
)
from open_webui.models.users import UserModel

log = logging.getLogger(__name__)
log.setLevel(SRC_LOG_LEVELS["OLLAMA"])

##########################################
# API routes
##########################################
router = APIRouter()

@router.head("/")
@router.get("/")
async def get_status():
    return {"status": True}

@router.post("/api/chat")
async def generate_chat_completion(
    request: Request,
    form_data: dict,
    user=Depends(get_verified_user)
):
    print("\n--- myagent.py generate_chat_completion ---")

    # Debug Information for the Request 
    print(f"Request method: {request.method}")
    print(f"Request url: {request.url}")
    print(f"Request headers: {dict(request.headers)}")
    body_bytes = await request.body()
    try:
        print(f"Request body (decoded): {body_bytes.decode('utf-8')}")
    except Exception as e:
        print(f"Unable to decode request body: {e}")

    # Extract the metadata from the form_data
    print(f"Initial form_data: {form_data}")
    metadata = form_data.pop("metadata", None)
    print(f"Metadata: {metadata}")

    # Convert the form_data object, this can probably be deleted.
    try:
        form_data = GenerateChatCompletionForm(**form_data)
        print(f"Parsed form_data (Pydantic): {form_data}")
    except Exception as e:
        log.exception(e)
        raise HTTPException(
            status_code=400,
            detail=str(e),
        )

    # Clean up the payload
    payload = {**form_data.model_dump(exclude_none=True)}
    if "metadata" in payload:
        del payload["metadata"]

    # Send the Post Request
    print(f"Final payload to POST: {json.dumps(payload)}")
    url = "http://host.docker.internal:8081"

    return await send_post_request(
        url=f"{url}/api/chat",
        payload=json.dumps(payload),
        stream=form_data.stream,
        content_type="application/x-ndjson",
        user=user,
    )

##########################################
# Utility functions
##########################################
class ChatMessage(BaseModel):
    role: str
    content: Optional[str] = None
    tool_calls: Optional[list[dict]] = None
    images: Optional[list[str]] = None

    @validator("content", pre=True)
    @classmethod
    def check_at_least_one_field(cls, field_value, values, **kwargs):
        # Raise an error if both 'content' and 'tool_calls' are None
        if field_value is None and (
            "tool_calls" not in values or values["tool_calls"] is None
        ):
            raise ValueError(
                "At least one of 'content' or 'tool_calls' must be provided"
            )

        return field_value

class GenerateChatCompletionForm(BaseModel):
    model: str
    messages: list[ChatMessage]
    format: Optional[Union[dict, str]] = None
    options: Optional[dict] = None
    template: Optional[str] = None
    stream: Optional[bool] = True
    keep_alive: Optional[Union[int, str]] = None
    tools: Optional[list[dict]] = None

async def send_post_request(
    url: str,
    payload: Union[str, bytes],
    stream: bool = True,
    content_type: Optional[str] = None,
    user: UserModel = None,
):
    print("\n--- send_post_request called ---")
    print(f"URL: {url}")
    try:
        print(f"Payload: {payload.decode('utf-8') if isinstance(payload, bytes) else payload}")
    except Exception as e:
        print(f"Unable to decode payload: {e}")
    print(f"Stream: {stream}")
    print(f"Content-Type: {content_type}")
    print(f"User: {user}")

    r = None
    try:
        session = aiohttp.ClientSession(
            trust_env=True, timeout=aiohttp.ClientTimeout(total=AIOHTTP_CLIENT_TIMEOUT)
        )
        print("ClientSession created")

        r = await session.post(
            url,
            data=payload,
            headers={
                "Content-Type": "application/json",
                **(
                    {
                        "X-OpenWebUI-User-Name": user.name,
                        "X-OpenWebUI-User-Id": user.id,
                        "X-OpenWebUI-User-Email": user.email,
                        "X-OpenWebUI-User-Role": user.role,
                    }
                ),
            },
            ssl=AIOHTTP_CLIENT_SESSION_SSL,
        )
        print(f"Response object: {r}")
        print(f"Response status: {r.status}")

        r.raise_for_status()

        res = await r.json()
        print(f"Response JSON: {res}")

        await cleanup_response(r, session)
        return res

    except Exception as e:
        print(f"Exception occurred: {e}")
        detail = None

        if r is not None:
            try:
                print("Attempting to parse error response JSON")
                res = await r.json()
                print(f"Error response JSON: {res}")
                if "error" in res:
                    detail = f"MyAgent: {res.get('error', 'Unknown error')}"
            except Exception as json_err:
                print(f"Failed to decode error response JSON: {json_err}")
                detail = f"MyAgent: {e}"

        raise HTTPException(
            status_code=r.status if r else 500,
            detail=detail if detail else "Open WebUI: Server Connection Error",
        )

async def cleanup_response(
    response: Optional[aiohttp.ClientResponse],
    session: Optional[aiohttp.ClientSession],
):
    print("\n--- cleanup_response called ---")
    if response:
        print(f"Closing response: {response}")
        response.close()
    if session:
        print("Closing session")
        await session.close()
