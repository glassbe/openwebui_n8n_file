import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from n8n_pipe_v4 import Pipe
import base64
import os

@pytest.fixture
def pipe_instance():
    pipe = Pipe()
    # Use a test config
    pipe.valves.OPENWEBUI_DATA_PATH = '/tmp/openwebui_data'  # Use a temp dir
    pipe.valves.max_file_size_mb = 1.0  # 1MB for testing
    pipe.valves.debug_mode = False
    return pipe

@pytest.mark.asyncio
async def test_pipe_no_files_json_base64(pipe_instance):
    pipe_instance.valves.FILE_TRANSFER_MODE = 'json_base64'
    body = {'messages': [{'role': 'user', 'content': 'Hello!'}]}
    # Patch emit_status to avoid side effects
    with patch.object(pipe_instance, 'emit_status', new=AsyncMock()):
        # Patch requests to n8n (simulate n8n response)
        with patch('aiohttp.ClientSession.post', new_callable=AsyncMock) as mock_post:
            mock_response = MagicMock()
            mock_response.status = 200
            mock_response.json = AsyncMock(return_value={pipe_instance.valves.response_field: 'n8n says hi'})
            mock_post.return_value.__aenter__.return_value = mock_response
            result = await pipe_instance.pipe(body, __files__=[], __event_emitter__=None)
            assert result is not None

@pytest.mark.asyncio
async def test_pipe_file_missing_on_disk(pipe_instance, tmp_path):
    pipe_instance.valves.FILE_TRANSFER_MODE = 'json_base64'
    pipe_instance.valves.OPENWEBUI_DATA_PATH = str(tmp_path)
    file_id = 'notfoundid'
    file_info = {'id': file_id, 'name': 'nofile.txt'}
    body = {'messages': [{'role': 'user', 'content': 'Test'}]}
    with patch.object(pipe_instance, 'emit_status', new=AsyncMock()):
        result = await pipe_instance.pipe(body, __files__=[file_info], __event_emitter__=None)
        # Should not crash, should return something reasonable
        assert result is not None

@pytest.mark.asyncio
async def test_pipe_file_oversize_json_base64(pipe_instance, tmp_path):
    pipe_instance.valves.FILE_TRANSFER_MODE = 'json_base64'
    pipe_instance.valves.OPENWEBUI_DATA_PATH = str(tmp_path)
    uploads_dir = tmp_path / 'uploads'
    uploads_dir.mkdir()
    file_id = 'oversizeid'
    file_path = uploads_dir / f'{file_id}_file.txt'
    # Write a 2MB file (over the 1MB limit)
    with open(file_path, 'wb') as f:
        f.write(os.urandom(2 * 1024 * 1024))
    file_info = {'id': file_id, 'name': 'file.txt'}
    body = {'messages': [{'role': 'user', 'content': 'Test'}]}
    with patch.object(pipe_instance, 'emit_status', new=AsyncMock()):
        result = await pipe_instance.pipe(body, __files__=[file_info], __event_emitter__=None)
        assert result is not None

@pytest.mark.asyncio
async def test_pipe_multipart_direct_content(pipe_instance):
    pipe_instance.valves.FILE_TRANSFER_MODE = 'multipart'
    content = base64.b64encode(b'hello world').decode('utf-8')
    file_info = {'id': 'directid', 'name': 'direct.txt', 'content': content, 'content_type': 'text/plain'}
    body = {'messages': [{'role': 'user', 'content': 'Test'}]}
    with patch.object(pipe_instance, 'emit_status', new=AsyncMock()):
        with patch('aiohttp.ClientSession.post', new_callable=AsyncMock) as mock_post:
            mock_response = MagicMock()
            mock_response.status = 200
            mock_response.json = AsyncMock(return_value={pipe_instance.valves.response_field: 'n8n says hi'})
            mock_post.return_value.__aenter__.return_value = mock_response
            result = await pipe_instance.pipe(body, __files__=[file_info], __event_emitter__=None)
            assert result is not None

@pytest.mark.asyncio
async def test_pipe_multipart_http_fetch_failure(pipe_instance):
    pipe_instance.valves.FILE_TRANSFER_MODE = 'multipart'
    file_info = {'id': 'httpid', 'name': 'http.txt', 'download_url': 'http://fakeurl'}
    body = {'messages': [{'role': 'user', 'content': 'Test'}]}
    with patch.object(pipe_instance, 'emit_status', new=AsyncMock()):
        with patch.object(pipe_instance, 'fetch_file_content', new=AsyncMock(return_value=None)):
            with patch('aiohttp.ClientSession.post', new_callable=AsyncMock) as mock_post:
                mock_response = MagicMock()
                mock_response.status = 200
                mock_response.json = AsyncMock(return_value={pipe_instance.valves.response_field: 'n8n says hi'})
                mock_post.return_value.__aenter__.return_value = mock_response
                result = await pipe_instance.pipe(body, __files__=[file_info], __event_emitter__=None)
                assert result is not None
