# n8n Pipe for OpenWebUI – Community Guide

---

## What is this?

This repository provides a **plug-and-play Python function** for OpenWebUI that enables robust, flexible file transfer from OpenWebUI uploads (images, documents, etc.) directly to your n8n automations. It supports advanced features like type detection, size limits, grouping, and seamless integration—**no standalone Python execution required**.

---

## How to Use with OpenWebUI

1. **Create Your Function in OpenWebUI**
   - Go to the **Functions** panel in OpenWebUI.
   - Click **Add Function** and give it a name (e.g., `n8n_pipe_files_v4`).

2. **Copy & Paste the Pipe Code**
   - Open the `n8n_pipe_v4.py` file from this repository.
   - Copy all its contents and paste them into your new OpenWebUI function.

3. **Configure the Valves**
   - At the top of the pasted code, locate the `Valves` class.
   - Set the following fields to match your environment:
     - `n8n_url`: Your n8n webhook endpoint.
     - `openwebui_data_path`: Parent directory for uploads (e.g., `/app/backend/data`).
     - `file_transfer_mode`: `'json_base64'` or `'multipart'`.
     - Adjust any other options as needed (see table below).

4. **Save and Use the Function**
   - Save your function.
   - In OpenWebUI, **select this function as a model** in your chat, workflow, or automation trigger.
   - When you upload files via OpenWebUI, they will be automatically sent to your n8n workflow according to your configuration.

5. **Test Your Setup**
   - Upload a file in OpenWebUI and verify it reaches your n8n workflow as expected.

---

## Configuration Options (`Valves`)

All behavior is controlled via the `Valves` class at the top of your function.

| Valve                      | Type    | Default                  | Description                                  |
|----------------------------|---------|--------------------------|----------------------------------------------|
| n8n_url                    | str     | —                        | n8n webhook endpoint                         |
| n8n_bearer_token           | str     | —                        | Auth token for n8n (if needed)               |
| file_transfer_mode         | str     | 'json_base64'            | 'json_base64' or 'multipart'                 |
| prefer_uploaded_file_bytes | bool    | True                     | For non-images, load from disk or body       |
| openwebui_data_path        | str     | '/app/backend/data'      | Parent dir for uploads                       |
| openwebui_api_key          | str     | None                     | API key for OpenWebUI fetch (multipart)      |
| max_file_size_mb           | float   | 25.0                     | Skip files over this size                    |
| include_file_type          | bool    | True                     | Include type info in payload                 |
| group_files_by_type        | bool    | False                    | Group files by type                          |

---

## How it Works

- **No need to run as a script:** OpenWebUI executes the function automatically when selected as a model.
- **Handles all file types:** Images, PDFs, docs, and more.
- **Smart logic:** Skips oversize files, groups by type if configured, and handles missing files gracefully.
- **Modes:**
  - `'json_base64'`: Files are base64-encoded and sent in JSON.
  - `'multipart'`: Files are sent as multipart form-data (optionally fetched with API key).

---

## Testing

To run the test suite locally (for development or troubleshooting):

1. Make sure you have Python 3.12+.
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the tests:
   ```bash
   pytest test_n8n_pipe_v4.py
   ```

**requirements.txt:**
```txt
aiohttp>=3.9.0,<4.0.0
pydantic>=2.0.0,<3.0.0
pytest>=8.0.0,<9.0.0
```

---

## Troubleshooting

| Symptom                        | Solution/Check                                                      |
|--------------------------------|---------------------------------------------------------------------|
| File not found on disk         | Confirm `openwebui_data_path` and uploads dir structure.            |
| Images not processed correctly | Ensure images are sent as base64 data URLs.                         |
| Oversize files skipped         | Increase `max_file_size_mb` if needed.                              |
| API key not sent in multipart  | Set `openwebui_api_key` in the valves.                              |
| Test failures                  | Double-check field names and config (case-sensitive).               |
| n8n webhook not called         | Confirm `n8n_url` and authentication.                               |

---

## Contributing

- **Enhancements welcome!**
  - If you improve file handling, add new features, or fix bugs, please share your updated function with the community.
- **How to contribute:**
  - Fork this repo or copy the function.
  - Add your improvements.
  - Open a pull request or share your changes on the OpenWebUI Discord/community.
- **Ideas for contribution:**
  - Support more file types or add custom logic.
  - Improve error handling or logging.
  - Add more tests or documentation.

---

**Maintained by the OpenWebUI & n8n community.**
_If you improve this pipe, please share back!_

---
