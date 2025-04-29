"""
title: n8n Pipe Function with Advanced File Support
author: Cole Medin (Original), Enhanced & Merged Implementation (Benedikt Glaß)
author_url: [https://www.youtube.com/@ColeMedin](https://www.youtube.com/@ColeMedin)
version: 0.4.0

This module defines a robust Pipe class that connects OpenWebUI with n8n workflows,
with advanced file handling capabilities including type detection, size limits,
and multiple file format support.
"""

from typing import Optional, Callable, Awaitable, Dict, List, Tuple, Any, Union
from pydantic import BaseModel, Field
import os
import time
import asyncio
import aiohttp  # Use asynchronous HTTP client
import base64
import json
import mimetypes
import logging

# Setup basic logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# Initialize mimetypes for file type detection
mimetypes.init()


def extract_event_info(event_emitter) -> tuple[Optional[str], Optional[str]]:
    """Extracts chat_id and message_id from the event emitter closure."""
    if not event_emitter or not event_emitter.__closure__:
        return None, None
    for cell in event_emitter.__closure__:
        closure_content = cell.cell_contents
        # Check if it's the request_info dict based on expected keys
        if (
            isinstance(closure_content, dict)
            and "chat_id" in closure_content
            and "message_id" in closure_content
        ):
            chat_id = closure_content.get("chat_id")
            message_id = closure_content.get("message_id")
            log.debug(f"Extracted chat_id: {chat_id}, message_id: {message_id}")
            return chat_id, message_id
    log.warning("Could not extract chat_id/message_id from event emitter.")
    return None, None


class Pipe:
    class Valves(BaseModel):
        # n8n Configuration
        n8n_url: str = Field(
            default="https://n8n.[your domain].com/webhook/[your webhook URL]",
            description="URL of the n8n webhook endpoint.",
        )
        n8n_bearer_token: str = Field(
            default="...", description="Bearer token for n8n webhook authentication."
        )
        input_field: str = Field(
            default="chatInput",
            description="Field name expected by n8n for the chat message content.",
        )
        response_field: str = Field(
            default="output",
            description="Field name in the n8n JSON response containing the text reply.",
        )

        # File Transfer Configuration
        file_transfer_mode: str = Field(
            default="multipart",
            description="How to send files to n8n: 'multipart' (traditional form upload) or 'json_base64' (embedded in JSON).",
        )
        n8n_file_field_base: str = Field(
            default="file",
            description="Base field name for files sent to n8n in multipart mode (e.g., 'file' -> file0, file1...).",
        )
        max_file_size_mb: float = Field(
            default=25.0,
            description="Maximum file size in MB to process (larger files will be skipped).",
        )
        include_file_type: bool = Field(
            default=True,
            description="Include file type information in the payload (JSON mode only).",
        )
        group_files_by_type: bool = Field(
            default=False,
            description="Group files by type in the payload (JSON mode only).",
        )
        prefer_uploaded_file_bytes: bool = Field(
            default=True,
            description="For non-image files: If True, load from uploads directory (raw bytes). If False, use extracted content from body. Images are always taken from the body as base64.",
        )
        openwebui_data_path: str = Field(
            default="/app/backend/data",
            description="Parent directory for OpenWebUI uploads (should contain 'uploads' subdir).",
        )

        # OpenWebUI Configuration (for fetching files)
        openwebui_base_url: str = Field(
            default="[http://host.docker.internal](http://host.docker.internal):8080",  # Adjust if OpenWebUI runs elsewhere
            description="Base URL of the OpenWebUI instance (accessible from this pipe's container) to fetch file content.",
        )
        openwebui_api_key: Optional[str] = Field(
            default=None,
            description="(Optional) API Key to authenticate requests to OpenWebUI for fetching files. Leave empty for unauthenticated.",
        )

        # Status Emission Configuration
        emit_interval: float = Field(
            default=1.0,  # Reduced interval for better feedback during file fetch
            description="Interval in seconds between status emissions.",
        )
        enable_status_indicator: bool = Field(
            default=True, description="Enable or disable status indicator emissions."
        )
        debug_mode: bool = Field(
            default=True,
            description="Enable verbose debug logging.",
        )

    def __init__(self):
        self.type = "pipe"
        self.id = "enhanced_n8n_pipe"
        self.name = "Enhanced N8N Pipe"
        self.valves = self.Valves()
        self.last_emit_time = 0
        log.info(f"Initialized {self.name} (ID: {self.id})")

        # Simple check for placeholder URL
        if (
            "[your domain]" in self.valves.n8n_url
            or "[your webhook URL]" in self.valves.n8n_url
        ):
            log.warning(
                "n8n URL appears to be a placeholder. Please configure it in Valves."
            )
        if self.valves.n8n_bearer_token == "...":
            log.warning(
                "n8n Bearer Token is a placeholder. Please configure it in Valves."
            )

        if self.valves.debug_mode:
            print("============ ENHANCED N8N PIPE INITIALIZED ============")

    def _determine_file_type(
        self, filename: str, content: Optional[Union[str, bytes]] = None
    ) -> Dict[str, str]:
        """
        Determine the file type, category, and MIME type based on filename and optionally content.

        Args:
            filename: The name of the file with extension
            content: Optional file content for further analysis

        Returns:
            Dictionary with file_type, mime_type, and category information
        """
        # Default values
        file_info = {
            "file_type": "unknown",
            "mime_type": "application/octet-stream",
            "category": "other",
        }

        # Try to determine MIME type from filename
        mime_type, _ = mimetypes.guess_type(filename)

        if mime_type:
            file_info["mime_type"] = mime_type

            # Extract general file type from MIME type
            main_type = mime_type.split("/")[0]
            sub_type = mime_type.split("/")[1] if "/" in mime_type else ""

            # Set category based on main MIME type
            if main_type == "image":
                file_info["category"] = "image"
                file_info["file_type"] = sub_type or "image"
            elif main_type == "text":
                file_info["category"] = "document"
                file_info["file_type"] = sub_type or "text"
            elif main_type == "audio":
                file_info["category"] = "audio"
                file_info["file_type"] = sub_type or "audio"
            elif main_type == "video":
                file_info["category"] = "video"
                file_info["file_type"] = sub_type or "video"
            elif main_type == "application":
                # Handle specific application subtypes
                if sub_type in [
                    "pdf",
                    "msword",
                    "vnd.openxmlformats-officedocument.wordprocessingml.document",
                ]:
                    file_info["category"] = "document"
                    file_info["file_type"] = "document"
                elif sub_type in [
                    "vnd.ms-excel",
                    "vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                ]:
                    file_info["category"] = "spreadsheet"
                    file_info["file_type"] = "spreadsheet"
                elif sub_type in [
                    "vnd.ms-powerpoint",
                    "vnd.openxmlformats-officedocument.presentationml.presentation",
                ]:
                    file_info["category"] = "presentation"
                    file_info["file_type"] = "presentation"
                elif sub_type in ["json", "xml"]:
                    file_info["category"] = "data"
                    file_info["file_type"] = sub_type
                else:
                    file_info["category"] = "application"
                    file_info["file_type"] = sub_type or "application"
        else:
            # Try to determine type from file extension if MIME type lookup failed
            ext = os.path.splitext(filename.lower())[1]
            if ext:
                if ext in [".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".svg"]:
                    file_info["category"] = "image"
                    file_info["file_type"] = ext[1:]  # Remove the dot
                    file_info["mime_type"] = f"image/{ext[1:]}"
                elif ext in [".pdf", ".doc", ".docx", ".txt", ".rtf", ".md", ".html"]:
                    file_info["category"] = "document"
                    file_info["file_type"] = "document"
                    if ext == ".pdf":
                        file_info["mime_type"] = "application/pdf"
                elif ext in [".xls", ".xlsx", ".csv"]:
                    file_info["category"] = "spreadsheet"
                    file_info["file_type"] = "spreadsheet"
                elif ext in [".mp3", ".wav", ".ogg", ".flac"]:
                    file_info["category"] = "audio"
                    file_info["file_type"] = "audio"
                elif ext in [".mp4", ".avi", ".mov", ".wmv"]:
                    file_info["category"] = "video"
                    file_info["file_type"] = "video"
                elif ext in [".json", ".xml", ".yaml", ".yml"]:
                    file_info["category"] = "data"
                    file_info["file_type"] = ext[1:]

        return file_info

    def _estimate_content_size_mb(self, content: Union[str, bytes]) -> float:
        """
        Estimate the size of content in megabytes.
        For Base64 content, this is approximate.

        Args:
            content: The content string or bytes

        Returns:
            Estimated size in MB
        """
        if not content:
            return 0.0

        # Calculate size in bytes
        if isinstance(content, str):
            size_bytes = len(content.encode("utf-8"))
        else:
            size_bytes = len(content)

        # Convert to MB
        size_mb = size_bytes / (1024 * 1024)

        return size_mb

    async def emit_status(
        self,
        __event_emitter__: Optional[Callable[[dict], Awaitable[None]]],
        level: str,
        message: str,
        done: bool,
    ):
        """Emits status updates throttled by emit_interval."""
        current_time = time.time()
        if (
            __event_emitter__
            and self.valves.enable_status_indicator
            and (
                current_time - self.last_emit_time >= self.valves.emit_interval or done
            )
        ):
            status_data = {
                "type": "status",
                "data": {
                    "status": "complete" if done else "in_progress",
                    "level": level,
                    "description": message,
                    "done": done,
                },
            }
            log.debug(f"Emitting status: {status_data['data']}")
            try:
                await __event_emitter__(status_data)
                self.last_emit_time = current_time
            except Exception as e:
                log.error(f"Failed to emit status: {e}")

    async def fetch_file_content(
        self, session: aiohttp.ClientSession, file_info: dict, auth_headers: dict
    ) -> Optional[Tuple[str, bytes, Dict[str, Any]]]:
        """
        Fetches the content of a single file from OpenWebUI.

        Returns:
            Tuple of (filename, content_bytes, metadata) or None on failure
        """
        file_id = file_info.get("id")
        original_filename = file_info.get(
            "filename", file_info.get("name", f"unknown_file_{file_id}")
        )
        relative_url = file_info.get("url")

        if not file_id or not relative_url:
            log.warning(f"Skipping file due to incomplete info: {file_info}")
            return None

        # Construct the full URL to fetch file content
        # Ensure no double slashes between base_url and relative_url
        base_url = self.valves.openwebui_base_url.rstrip("/")
        content_url_path = f"{relative_url.strip('/')}/content"
        content_url = f"{base_url}/{content_url_path}"

        log.info(
            f"Attempting to fetch file '{original_filename}' (ID: {file_id}) from: {content_url}"
        )

        try:
            async with session.get(
                content_url,
                headers=auth_headers,
                timeout=aiohttp.ClientTimeout(total=120),
            ) as response:  # Added timeout
                if response.status == 200:
                    content_bytes = await response.read()

                    # Check file size against limit
                    size_mb = len(content_bytes) / (1024 * 1024)
                    if size_mb > self.valves.max_file_size_mb:
                        log.warning(
                            f"File '{original_filename}' exceeds size limit ({size_mb:.2f} MB > {self.valves.max_file_size_mb} MB)"
                        )
                        return None

                    log.info(
                        f"Successfully fetched '{original_filename}' ({len(content_bytes)} bytes, {size_mb:.2f} MB)."
                    )

                    # Get file type information
                    file_type_info = self._determine_file_type(
                        original_filename, content_bytes
                    )

                    return original_filename, content_bytes, file_type_info
                else:
                    error_text = await response.text()
                    log.error(
                        f"Error fetching file '{original_filename}' (ID: {file_id}). Status: {response.status}. Response: {error_text[:500]}"
                    )  # Log first 500 chars
                    return None  # Indicate fetch failure
        except aiohttp.ClientConnectorError as e:
            log.error(
                f"Network connection error fetching '{original_filename}': {e}. Check if OpenWebUI is reachable at {base_url} from the pipe container."
            )
            return None
        except asyncio.TimeoutError:
            log.error(
                f"Timeout error fetching '{original_filename}' from {content_url}."
            )
            return None
        except Exception as e:
            log.exception(
                f"Unexpected error fetching file '{original_filename}': {e}"
            )  # Log full traceback
            return None

    async def pipe(
        self,
        body: dict,
        messages: Optional[List[Dict]] = None,
        model_id: Optional[str] = None,
        user_message: Optional[str] = None,
        __user__: Optional[dict] = None,
        __event_emitter__: Optional[Callable[[dict], Awaitable[None]]] = None,
        __event_call__: Optional[Callable[[dict], Awaitable[dict]]] = None,
        __files__: Optional[List[Dict[str, Any]]] = None,
        **kwargs,
    ) -> Optional[dict]:
        """
        Processes the request, fetches files if present, and sends data
        and files to the configured n8n webhook.

        Supports two file transfer modes:
        1. multipart/form-data (traditional file upload)
        2. json_base64 (files embedded in JSON payload)

        Returns the modified body with n8n response added as an assistant message.
        """
        if self.valves.debug_mode:
            print("\n\n============ ENHANCED N8N PIPE FUNCTION CALLED ============")
            print(f"BODY KEYS: {list(body.keys() if isinstance(body, dict) else [])}")
            #print(f"BODY : {body}")
            print(f"MESSAGES: {len(messages) if messages else 0} messages")
            print(f"__files__: {'Present' if __files__ else 'Not present'}")
            if __files__:
                print(f"  __files__ count: {len(__files__)}")
                if len(__files__) > 0:
                    print(f"  First file keys: {list(__files__[0].keys())}")

        await self.emit_status(
            __event_emitter__, "info", "/ Calling N8N Workflow...", False
        )
        log.debug(f"Received body: {body}")
        if self.valves.debug_mode and __user__:
            log.debug(
                f"Received user context: {__user__}"
            )  # Be careful logging user data

        # Get chat ID for session tracking
        chat_id, _ = extract_event_info(__event_emitter__)
        if not chat_id:
            chat_id = body.get("chat_id", str(int(time.time())))  # Fallback chat_id
            log.warning(f"Using fallback chat_id: {chat_id}")

        # Extract user message from various possible sources
        question = None
        # Try user_message parameter first if provided
        if user_message:
            question = user_message
        # Then try to get from messages parameter if provided
        elif messages and isinstance(messages, list):
            for msg in reversed(messages):
                if isinstance(msg, dict) and msg.get("role") == "user":
                    content = msg.get("content")
                    if isinstance(content, str):
                        question = content
                        break
                    elif isinstance(content, list):
                        # Handle multimodal content
                        text_parts = []
                        for item in content:
                            if isinstance(item, dict) and item.get("type") == "text":
                                text_parts.append(item.get("text", ""))
                        if text_parts:
                            question = " ".join(text_parts)
                            break

        # Finally try to get from body if message not found elsewhere
        if not question:
            body_messages = body.get("messages", [])
            if body_messages and isinstance(body_messages, list):
                for msg in reversed(body_messages):
                    if isinstance(msg, dict) and msg.get("role") == "user":
                        content = msg.get("content")
                        if isinstance(content, str):
                            question = content
                            break
                        elif isinstance(content, list):
                            # Handle multimodal content
                            text_parts = []
                            for item in content:
                                if (
                                    isinstance(item, dict)
                                    and item.get("type") == "text"
                                ):
                                    text_parts.append(item.get("text", ""))
                            if text_parts:
                                question = " ".join(text_parts)
                                break

                # If still no question, try the last message regardless of role
                if not question and body_messages:
                    last_msg = body_messages[-1]
                    if isinstance(last_msg, dict):
                        content = last_msg.get("content")
                        if isinstance(content, str):
                            question = content

        if not question:
            error_msg = "Could not extract user message from request"
            await self.emit_status(__event_emitter__, "error", error_msg, True)
            # Return with error but maintain structure
            if isinstance(body, dict) and "messages" in body:
                body["messages"].append(
                    {"role": "assistant", "content": f"Error: {error_msg}"}
                )
            return body

        # --- File Handling ---
        files_to_send = {}  # For multipart mode
        processed_files = []  # For JSON mode
        file_stats = {"total": 0, "processed": 0, "skipped": 0, "error": 0}
        file_categories = {}  # For grouping by type

        # Get files from __files__ parameter or body["files"]
        uploaded_files_info = __files__ or body.get("files", [])

        # --- Extract base64 images from multimodal messages ---
        multimodal_images = []
        # Check both messages param and body['messages']
        all_messages = []
        if messages and isinstance(messages, list):
            all_messages.extend(messages)
        if isinstance(body, dict) and isinstance(body.get("messages"), list):
            all_messages.extend(body["messages"])
        for msg in all_messages:
            if not (isinstance(msg, dict) and msg.get("role") == "user"):
                continue
            content = msg.get("content")
            if isinstance(content, list):
                for item in content:
                    if (
                        isinstance(item, dict)
                        and item.get("type") == "image_url"
                        and isinstance(item.get("image_url"), dict)
                        and isinstance(item["image_url"].get("url"), str)
                        and item["image_url"]["url"].startswith("data:image/")
                    ):
                        # Extract image format
                        data_url = item["image_url"]["url"]
                        header, base64_data = data_url.split(",", 1)
                        # Try to get extension from mime type
                        mime_type = header.split(":")[1].split(";")[0]  # e.g. image/png
                        ext = mime_type.split("/")[1] if "/" in mime_type else "bin"
                        filename = f"image_from_body_{len(multimodal_images)+1}.{ext}"
                        try:
                            content_bytes = base64.b64decode(base64_data)
                        except Exception as e:
                            log.warning(f"Failed to decode base64 image from body: {e}")
                            continue
                        file_info = {
                            "id": f"body_image_{len(multimodal_images)+1}",
                            "filename": filename,
                            "content": data_url,
                            "from_body_base64": True,
                        }
                        # For multipart mode, add bytes; for json_base64, keep base64 string
                        if self.valves.file_transfer_mode == "multipart":
                            file_info["content_bytes"] = content_bytes
                        multimodal_images.append(file_info)
        if multimodal_images:
            log.info(f"Extracted {len(multimodal_images)} base64 images from body messages.")
            uploaded_files_info = list(uploaded_files_info) + multimodal_images

        # Helper to check if a file is an image (by filename or content)
        def _is_image_file(filename, content=None):
            if filename and any(filename.lower().endswith(ext) for ext in [".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp"]):
                return True
            if content and isinstance(content, str) and content.startswith("data:image/"):
                return True
            return False

        if uploaded_files_info:
            file_stats["total"] = len(uploaded_files_info)
            log.info(f"Found {file_stats['total']} file references in the request.")
            await self.emit_status(
                __event_emitter__,
                "info",
                f"/ Preparing {file_stats['total']} file(s)...",
                False,
            )

            # Prepare headers for fetching files from OpenWebUI API
            ow_auth_headers = {}
            if self.valves.openwebui_api_key:
                ow_auth_headers["Authorization"] = (
                    f"Bearer {self.valves.openwebui_api_key}"
                )
                log.info("Using configured OpenWebUI API key for file fetching.")
            else:
                # Attempt to use user context if available
                user_api_key = __user__.get("api_key") if __user__ else None
                if user_api_key:
                    ow_auth_headers["Authorization"] = f"Bearer {user_api_key}"
                    log.info("Using user's API key from context for file fetching.")
                else:
                    log.warning(
                        "No API key configured or found. Attempting unauthenticated file fetch."
                    )

            # Use a single session for all file fetches
            async with aiohttp.ClientSession(headers=ow_auth_headers) as session:
                for index, file_item in enumerate(uploaded_files_info):
                    try:
                        if self.valves.debug_mode:
                            print(
                                f"Processing file item {index+1}/{file_stats['total']}: {json.dumps(file_item, default=str)[:300]}..."
                            )

                        # Structure seems to be {"file": { ... file info ... }} or direct dict
                        file_info = file_item.get("file", file_item)

                        if self.valves.debug_mode and isinstance(file_info, dict):
                            print(f"File data keys: {list(file_info.keys())}")

                        # Extract file details
                        file_id = file_info.get("id")
                        filename = (
                            file_info.get("filename")
                            or file_info.get("name")
                            or f"file_{index}_{int(time.time())}"
                        )

                        await self.emit_status(
                            __event_emitter__,
                            "info",
                            f"/ Processing file {index + 1}/{file_stats['total']}: {filename}...",
                            False,
                        )

                        # Try to get content directly first
                        content = None
                        content_bytes = None
                        file_type_info = None

                        # Choose strategy for non-image files
                        is_image = _is_image_file(filename, file_info.get("content"))
                        prefer_upload = self.valves.prefer_uploaded_file_bytes
                        # For images, always use content from body (already handled above)
                        # For non-images, branch based on valve
                        if is_image:
                            # Always use content from body (already handled)
                            log.debug(f"Image file '{filename}' - using body/base64 content.")
                            # Existing logic below will process content from file_info
                        elif prefer_upload and file_id:
                            # Try to load from uploads directory
                            try:
                                # Use the correct uploads directory and pattern as per memories
                                uploads_dir = os.path.join(os.environ.get("OPENWEBUI_DATA_PATH", "/app/backend/data"), "uploads")
                                import glob
                                pattern = os.path.join(uploads_dir, f"{file_id}_*")
                                matches = glob.glob(pattern)
                                if matches:
                                    filepath = matches[0]
                                    with open(filepath, "rb") as f:
                                        content_bytes = f.read()
                                    log.info(f"Loaded file '{filename}' from uploads directory as raw bytes.")
                                    # For json_base64 mode, encode to base64
                                    if self.valves.file_transfer_mode == "json_base64":
                                        content = base64.b64encode(content_bytes).decode("utf-8")
                                else:
                                    log.warning(f"File ID {file_id} not found in uploads directory. Falling back to body content if available.")
                                    # fallback to body content below
                            except Exception as e:
                                log.warning(f"Failed to load file '{filename}' from uploads: {e}. Falling back to body content if available.")
                                # fallback to body content below
                        # If not using uploads, or fallback
                        if content_bytes is None and "content" in file_info:
                            content = file_info.get("content")
                            if isinstance(content, str):
                                # Convert string content to bytes if needed
                                try:
                                    # Check if it's base64 encoded
                                    if content.startswith("data:"):
                                        # Extract base64 content from data URI
                                        _, base64_content = content.split(",", 1)
                                        content_bytes = base64.b64decode(base64_content)
                                    else:
                                        # Try to decode as base64 directly
                                        content_bytes = base64.b64decode(content)
                                except Exception:
                                    # If not base64, treat as UTF-8 text
                                    content_bytes = content.encode("utf-8")
                            elif isinstance(content, bytes):
                                content_bytes = content

                            # If we got content, check size and determine type
                            if content_bytes:
                                size_mb = len(content_bytes) / (1024 * 1024)
                                if size_mb > self.valves.max_file_size_mb:
                                    log.warning(
                                        f"File '{filename}' exceeds size limit ({size_mb:.2f} MB > {self.valves.max_file_size_mb} MB)"
                                    )
                                    file_stats["skipped"] += 1
                                    continue

                                file_type_info = self._determine_file_type(
                                    filename, content_bytes
                                )

                        # Check if file_info has nested data with content
                        elif (
                            not content_bytes
                            and "data" in file_info
                            and isinstance(file_info["data"], dict)
                        ):
                            nested_content = file_info["data"].get("content")
                            if nested_content:
                                try:
                                    if isinstance(nested_content, str):
                                        content_bytes = nested_content.encode("utf-8")
                                    elif isinstance(nested_content, bytes):
                                        content_bytes = nested_content

                                    size_mb = len(content_bytes) / (1024 * 1024)
                                    if size_mb > self.valves.max_file_size_mb:
                                        log.warning(
                                            f"File '{filename}' exceeds size limit ({size_mb:.2f} MB > {self.valves.max_file_size_mb} MB)"
                                        )
                                        file_stats["skipped"] += 1
                                        continue

                                    file_type_info = self._determine_file_type(
                                        filename, content_bytes
                                    )
                                except Exception as e:
                                    log.warning(
                                        f"Error processing nested content for '{filename}': {e}"
                                    )

                        # If no content yet, try to fetch using URL
                        if not content_bytes:
                            url = file_info.get("url")
                            if url:
                                fetch_result = await self.fetch_file_content(
                                    session, file_info, {}
                                )
                                if fetch_result:
                                    filename, content_bytes, file_type_info = (
                                        fetch_result
                                    )
                                else:
                                    file_stats["error"] += 1
                                    await self.emit_status(
                                        __event_emitter__,
                                        "warning",
                                        f"/ Failed to fetch file: {filename}",
                                        False,
                                    )
                                    continue
                            else:
                                log.warning(f"No content or URL for file '{filename}'")
                                file_stats["error"] += 1
                                continue

                        # Now we should have content_bytes, file_type_info, and filename
                        if content_bytes:
                            # For multipart mode, prepare the tuple (filename, bytes)
                            if self.valves.file_transfer_mode == "multipart":
                                n8n_field_name = (
                                    f"{self.valves.n8n_file_field_base}{index}"
                                )
                                files_to_send[n8n_field_name] = (
                                    filename,
                                    content_bytes,
                                )

                            # For JSON mode, prepare the processed file object
                            else:  # json_base64 mode
                                # Convert bytes to base64 string for JSON
                                base64_content = base64.b64encode(content_bytes).decode(
                                    "utf-8"
                                )

                                # Create processed file data
                                processed_file = {
                                    "id": file_id,
                                    "name": filename,
                                    "content": base64_content,
                                    "size": len(content_bytes)
                                    / (1024 * 1024),  # size in MB
                                }

                                # Add file type information if enabled
                                if self.valves.include_file_type and file_type_info:
                                    processed_file.update(file_type_info)

                                    # Track file categories for grouping
                                    if self.valves.group_files_by_type:
                                        category = file_type_info.get(
                                            "category", "other"
                                        )
                                        if category not in file_categories:
                                            file_categories[category] = []
                                        file_categories[category].append(processed_file)

                                processed_files.append(processed_file)

                            file_stats["processed"] += 1

                            await self.emit_status(
                                __event_emitter__,
                                "info",
                                f"/ Processed file: {filename} ({file_type_info.get('category', 'unknown')})",
                                False,
                            )
                        else:
                            file_stats["error"] += 1
                            await self.emit_status(
                                __event_emitter__,
                                "warning",
                                f"/ Could not process file: {filename}",
                                False,
                            )
                    except Exception as e:
                        file_stats["error"] += 1
                        log.exception(f"Error processing file {index}: {e}")
                        await self.emit_status(
                            __event_emitter__,
                            "error",
                            f"/ Error processing file: {str(e)}",
                            False,
                        )

            if file_stats["processed"] == 0 and file_stats["total"] > 0:
                log.warning("Failed to process any of the files.")
                await self.emit_status(
                    __event_emitter__,
                    "warning",
                    f"/ Failed to process any files. {file_stats['error']} errors, {file_stats['skipped']} skipped.",
                    False,
                )
            elif file_stats["processed"] < file_stats["total"]:
                log.info(
                    f"Processed {file_stats['processed']}/{file_stats['total']} files successfully."
                )
                await self.emit_status(
                    __event_emitter__,
                    "info",
                    f"/ Processed {file_stats['processed']}/{file_stats['total']} files successfully.",
                    False,
                )
            else:
                log.info(f"Successfully processed all {file_stats['processed']} files.")
                log.info(f"PROCESSED FILES: {processed_files}")
                await self.emit_status(
                    __event_emitter__,
                    "info",
                    f"/ All {file_stats['processed']} files ready.",
                    False,
                )

        # --- Create n8n Payload ---
        n8n_response_content = "Error: Did not receive a valid response from n8n."

        try:
            # Prepare headers for n8n request
            n8n_headers = {
                "Authorization": f"Bearer {self.valves.n8n_bearer_token}",
            }

            # Create the payload with the chat message
            payload_data = {"sessionId": chat_id}
            payload_data[self.valves.input_field] = question

            # Always send pure JSON to n8n (files as base64 in 'files' array)
            await self.emit_status(
                __event_emitter__,
                "info",
                f"/ Sending data {f'and {len(processed_files)} file(s)' if processed_files else ''} to n8n via JSON...",
                False,
            )

            # Create the complete payload including files
            if processed_files:
                # If grouping by type is enabled and we have categories
                if (
                    self.valves.group_files_by_type
                    and file_categories
                    and len(file_categories) > 1
                ):
                    payload_data["files_by_type"] = file_categories
                    # Add summary counts
                    payload_data["file_summary"] = {
                        "total": file_stats["total"],
                        "categories": {
                            category: len(files)
                            for category, files in file_categories.items()
                        },
                    }
                else:
                    # Otherwise, add flat list of files
                    payload_data["files"] = processed_files
                # Add file statistics
                payload_data["file_stats"] = file_stats

            if self.valves.debug_mode:
                log.debug(
                    f"Sending payload to n8n: {json.dumps(payload_data, default=str)[:1000]}..."
                )

            # Ensure processed files are included in the payload
            if processed_files:
                payload_data["files"] = processed_files

            # Log the final payload for debugging
            log.info(f"FINAL PAYLOAD: {json.dumps(payload_data)[:1000]}")

            # Make the POST request to n8n
            async with aiohttp.ClientSession() as session:
                log.info(f"Sending JSON POST request to n8n: {self.valves.n8n_url}")
                async with session.post(
                    self.valves.n8n_url,
                    json=payload_data,
                    headers=n8n_headers,
                    timeout=aiohttp.ClientTimeout(total=300),  # Longer timeout
                ) as response:
                    await self._process_n8n_response(response, __event_emitter__)
                    if response.status == 200:
                        try:
                            response_json = await response.json()
                            n8n_response_content = response_json.get(
                                self.valves.response_field,
                                "Received success from n8n but missing expected response field.",
                            )
                        except aiohttp.ContentTypeError:
                            log.warning(
                                "n8n response was not valid JSON. Using text response."
                            )
                            n8n_response_content = await response.text()
                    else:
                        response_text = await response.text()
                        error_msg = f"Error from n8n: Status {response.status} - {response_text[:200]}"
                        log.error(error_msg)
                        n8n_response_content = error_msg

        except aiohttp.ClientConnectorError as e:
            log.error(f"Network connection error calling n8n: {e}")
            await self.emit_status(
                __event_emitter__, "error", f"/ Error: Could not connect to n8n.", True
            )
            n8n_response_content = f"Error: Could not connect to n8n server: {str(e)}"
        except asyncio.TimeoutError:
            log.error(f"Timeout error calling n8n at {self.valves.n8n_url}")
            await self.emit_status(
                __event_emitter__, "error", f"/ Error: Timeout connecting to n8n.", True
            )
            n8n_response_content = "Error: Connection to n8n timed out. The operation may have been too complex or the server is busy."
        except Exception as e:
            log.exception(f"Unexpected error during n8n communication: {e}")
            await self.emit_status(
                __event_emitter__, "error", f"/ Error: {str(e)}", True
            )
            n8n_response_content = (
                f"An unexpected error occurred when communicating with n8n: {str(e)}"
            )

        # --- Format and Return Response ---
        log.info(
            f"n8n interaction complete. Response content (truncated): {n8n_response_content[:100]}..."
        )

        # Ensure the response content is a string
        if not isinstance(n8n_response_content, str):
            log.warning(
                f"n8n response content was not a string, converting: {type(n8n_response_content)}"
            )
            n8n_response_content = str(n8n_response_content)

        # Append the response from n8n as an assistant message
        if isinstance(body, dict) and "messages" in body:
            body["messages"].append(
                {"role": "assistant", "content": n8n_response_content}
            )

        await self.emit_status(
            __event_emitter__, "info", "/ N8N Workflow Complete", True
        )

        # Return the modified body to be passed to the next stage
        return n8n_response_content

    async def _process_n8n_response(self, response, __event_emitter__):
        """Process the response from n8n and emit appropriate status updates."""
        status = response.status
        if status == 200:
            await self.emit_status(
                __event_emitter__,
                "info",
                f"/ n8n workflow executed successfully.",
                False,
            )
        elif status >= 400 and status < 500:
            await self.emit_status(
                __event_emitter__,
                "error",
                f"/ Error: Client error when calling n8n (Status {status}).",
                False,
            )
        elif status >= 500:
            await self.emit_status(
                __event_emitter__,
                "error",
                f"/ Error: Server error in n8n workflow (Status {status}).",
                False,
            )
        else:
            await self.emit_status(
                __event_emitter__,
                "warning",
                f"/ Unexpected response from n8n (Status {status}).",
                False,
            )
