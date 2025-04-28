# n8n Pipe with File Support

## Overview

This project provides a robust Python pipe for integrating OpenWebUI with n8n workflows, supporting advanced file handling:
- **Text and binary file uploads** (including images, PDFs, etc.)
- **Multiple transfer modes:** `json_base64` (files as base64 in JSON) and `multipart` (files as multipart/form-data)
- **File size limits, type detection, and grouping**
- **Async, robust, and ready for community use**

---

## Features

- Send user messages and files from OpenWebUI to n8n endpoints.
- Receive and display LLM or workflow responses in chat.
- Handles files uploaded via OpenWebUI, including large and binary files.
- Configurable via the `Valves` class.
- Includes a comprehensive test suite.

---

## Installation

1. **Clone or copy the files:**
    - `n8n_pipe_v4.py`
    - (Optional) `test_n8n_pipe_v4.py` for testing

2. **Install dependencies:**

    On Ubuntu/Debian:
    ```bash
    apt install python3-pydantic python3-aiohttp python3-pytest python3-pytest-asyncio
    ```

    Or, using pip (if allowed):
    ```bash
    pip3 install pydantic aiohttp pytest pytest-asyncio
    ```

---

## Configuration

All configuration is handled via the `Pipe.Valves` class. Key options include:

- `n8n_url`: Your n8n webhook endpoint.
- `n8n_bearer_token`: Bearer token for n8n authentication.
- `FILE_TRANSFER_MODE`: `'json_base64'` (default, best for OpenWebUI Docker) or `'multipart'`.
- `OPENWEBUI_DATA_PATH`: Path to OpenWebUI's data directory (parent of `uploads/`).
- `max_file_size_mb`: Maximum file size to process (default: 25MB).
- `debug_mode`: Enable verbose logging for debugging.

You can override these by editing the `Valves` class or setting them at runtime.

---

## How File Uploads Work

- **json_base64 mode:**  
  Files are loaded from disk (using the pattern `<OPENWEBUI_DATA_PATH>/uploads/<file_id>_<original_filename>`) and base64-encoded into the outgoing JSON.
- **multipart mode:**  
  Files are fetched via HTTP (using the `download_url`) or taken directly from the provided base64 content, and sent as multipart form fields.

---

## Usage in OpenWebUI

1. **Paste the `Pipe` class (from `n8n_pipe_v4.py`) into OpenWebUI's custom pipe interface.**
2. **Configure your valves as needed.**
3. **Upload files and send messages as usual.**
4. **The response from your n8n workflow will be displayed in chat.**

---

## Testing

A full test suite is provided in `test_n8n_pipe_v4.py`.

### Run the tests:

```bash
pytest test_n8n_pipe_v4.py
```

- Tests cover both file modes, error scenarios, and edge cases.
- No need for a running OpenWebUI or n8n instance; all external calls are mocked.

---

## Troubleshooting

- **File not found?**  
  Ensure `OPENWEBUI_DATA_PATH` is correct and files are in the expected `uploads/` directory.
- **File too large?**  
  Adjust `max_file_size_mb` in your valves.
- **Async test warnings?**  
  Make sure `pytest-asyncio` is installed.
- **Pydantic deprecation warnings?**  
  These are safe to ignore for now but may require updating field definitions in the future.

---

## Example Valves Configuration

```python
class Valves(BaseModel):
    n8n_url: str = "https://n8n.example.com/webhook/your-webhook"
    n8n_bearer_token: str = "your-token"
    FILE_TRANSFER_MODE: str = "json_base64"
    OPENWEBUI_DATA_PATH: str = "/app/backend/data"
    max_file_size_mb: float = 25.0
    debug_mode: bool = True
    # ...other options...
```

---

## Community & Contribution

- Issues and PRs welcome!
- Please include test cases for new features.
- For questions, open an issue or join the discussion in the OpenWebUI or n8n communities.

---

## License

MIT License (or your preferred license)

---

Let me know if you want to include any extra sections, usage diagrams, or want this README as a file!
