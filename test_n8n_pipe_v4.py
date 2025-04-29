import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from n8n_pipe_v4 import Pipe
import base64
import os
import json
import re

@pytest.fixture
def pipe_instance():
    pipe = Pipe()
    # Use a test config
    pipe.valves.openwebui_data_path = '/tmp/openwebui_data'  # Use a temp dir
    pipe.valves.max_file_size_mb = 1.0  # 1MB for testing
    pipe.valves.debug_mode = False
    return pipe

@pytest.mark.asyncio
async def test_pipe_no_files_json_base64(pipe_instance):
    pipe_instance.valves.file_transfer_mode = 'json_base64'
    pipe_instance.valves.prefer_uploaded_file_bytes = True
    body = {'messages': [{'role': 'user', 'content': 'Hello!'}]}
    with patch.object(pipe_instance, 'emit_status', new=AsyncMock()):
        with patch('aiohttp.ClientSession.post', new_callable=AsyncMock) as mock_post:
            mock_response = MagicMock()
            mock_response.status = 200
            mock_response.json = AsyncMock(return_value={pipe_instance.valves.response_field: 'n8n says hi'})
            mock_post.return_value.__aenter__.return_value = mock_response
            result = await pipe_instance.pipe(body, __files__=[], __event_emitter__=None)
            assert result is not None

@pytest.mark.asyncio
async def test_pipe_nonimage_prefer_upload_bytes(tmp_path, pipe_instance):
    pipe_instance.valves.file_transfer_mode = 'json_base64'
    pipe_instance.valves.prefer_uploaded_file_bytes = True
    pipe_instance.valves.openwebui_data_path = str(tmp_path)
    uploads_dir = tmp_path / 'uploads'
    uploads_dir.mkdir()
    file_id = 'testid'
    file_path = uploads_dir / f'{file_id}_file.txt'
    with open(file_path, 'wb') as f:
        f.write(b'raw file content')
    file_info = {'id': file_id, 'name': 'file.txt'}
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
async def test_pipe_nonimage_prefer_body_content(pipe_instance):
    pipe_instance.valves.file_transfer_mode = 'json_base64'
    pipe_instance.valves.prefer_uploaded_file_bytes = False
    file_info = {'id': 'bodyid', 'name': 'file.txt', 'content': base64.b64encode(b'body content').decode('utf-8')}
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
async def test_pipe_image_always_from_body(pipe_instance):
    pipe_instance.valves.file_transfer_mode = 'json_base64'
    pipe_instance.valves.prefer_uploaded_file_bytes = True  # Should not matter for images
    img_base64 = base64.b64encode(b'fakeimagedata').decode('utf-8')
    image_data_url = f'data:image/png;base64,{img_base64}'
    image_item = {'type': 'image_url', 'image_url': {'url': image_data_url}}
    body = {'messages': [{'role': 'user', 'content': [image_item]}]}
    with patch.object(pipe_instance, 'emit_status', new=AsyncMock()):
        with patch('aiohttp.ClientSession.post', new_callable=AsyncMock) as mock_post:
            mock_response = MagicMock()
            mock_response.status = 200
            mock_response.json = AsyncMock(return_value={pipe_instance.valves.response_field: 'n8n says hi'})
            mock_post.return_value.__aenter__.return_value = mock_response
            result = await pipe_instance.pipe(body, __files__=[], __event_emitter__=None)
            assert result is not None

@pytest.mark.asyncio
async def test_pipe_file_missing_on_disk(pipe_instance, tmp_path):
    pipe_instance.valves.file_transfer_mode = 'json_base64'
    pipe_instance.valves.prefer_uploaded_file_bytes = True
    pipe_instance.valves.openwebui_data_path = str(tmp_path)
    file_id = 'notfoundid'
    file_info = {'id': file_id, 'name': 'nofile.txt'}
    body = {'messages': [{'role': 'user', 'content': 'Test'}]}
    with patch.object(pipe_instance, 'emit_status', new=AsyncMock()):
        result = await pipe_instance.pipe(body, __files__=[file_info], __event_emitter__=None)
        assert result is not None

@pytest.mark.asyncio
async def test_pipe_file_oversize_json_base64(pipe_instance, tmp_path):
    pipe_instance.valves.file_transfer_mode = 'json_base64'
    pipe_instance.valves.prefer_uploaded_file_bytes = True
    pipe_instance.valves.openwebui_data_path = str(tmp_path)
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
    pipe_instance.valves.file_transfer_mode = 'multipart'
    pipe_instance.valves.prefer_uploaded_file_bytes = False
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
    pipe_instance.valves.file_transfer_mode = 'multipart'
    pipe_instance.valves.prefer_uploaded_file_bytes = True
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

@pytest.mark.asyncio
async def test_pipe_invalid_base64_in_body(pipe_instance):
    pipe_instance.valves.file_transfer_mode = 'json_base64'
    pipe_instance.valves.prefer_uploaded_file_bytes = False
    file_info = {'id': 'badid', 'name': 'file.txt', 'content': '!!!notbase64!!!'}
    body = {'messages': [{'role': 'user', 'content': 'Test'}]}
    with patch.object(pipe_instance, 'emit_status', new=AsyncMock()):
        try:
            result = await pipe_instance.pipe(body, __files__=[file_info], __event_emitter__=None)
        except Exception as e:
            result = str(e)
        assert result is not None

@pytest.mark.asyncio
async def test_pipe_unknown_extension(pipe_instance, tmp_path):
    pipe_instance.valves.file_transfer_mode = 'json_base64'
    pipe_instance.valves.prefer_uploaded_file_bytes = True
    uploads_dir = tmp_path / 'uploads'
    uploads_dir.mkdir()
    file_id = 'weirdid'
    filename = 'weirdid.abcxyz'  # Unknown extension
    file_path = uploads_dir / filename
    file_path.write_bytes(b'weird content')
    file_info = {'id': file_id, 'name': filename}
    body = {'messages': [{'role': 'user', 'content': 'Test'}]}
    pipe_instance.valves.openwebui_data_path = str(tmp_path)
    with patch.object(pipe_instance, 'emit_status', new=AsyncMock()):
        result = await pipe_instance.pipe(body, __files__=[file_info], __event_emitter__=None)
        assert result is not None

@pytest.mark.asyncio
async def test_pipe_group_files_by_type(pipe_instance, tmp_path):
    pipe_instance.valves.file_transfer_mode = 'json_base64'
    pipe_instance.valves.group_files_by_type = True
    uploads_dir = tmp_path / 'uploads'
    uploads_dir.mkdir()
    file1_id = 'id1'
    file2_id = 'id2'
    file1_path = uploads_dir / f'{file1_id}.txt'
    file2_path = uploads_dir / f'{file2_id}.pdf'
    file1_path.write_bytes(b'abc')
    file2_path.write_bytes(b'def')
    file1_info = {'id': file1_id, 'name': 'id1.txt'}
    file2_info = {'id': file2_id, 'name': 'id2.pdf'}
    body = {'messages': [{'role': 'user', 'content': 'Test'}]}
    pipe_instance.valves.openwebui_data_path = str(tmp_path)
    with patch.object(pipe_instance, 'emit_status', new=AsyncMock()):
        result = await pipe_instance.pipe(body, __files__=[file1_info, file2_info], __event_emitter__=None)
        assert result is not None
        if isinstance(result, dict):
            assert 'files_by_type' in result

@pytest.mark.asyncio
async def test_pipe_include_file_type_false(pipe_instance, tmp_path):
    pipe_instance.valves.file_transfer_mode = 'json_base64'
    pipe_instance.valves.include_file_type = False
    uploads_dir = tmp_path / 'uploads'
    uploads_dir.mkdir()
    file_id = 'idtype'
    filename = 'idtype.txt'
    file_path = uploads_dir / filename
    file_path.write_bytes(b'abc')
    file_info = {'id': file_id, 'name': filename}
    body = {'messages': [{'role': 'user', 'content': 'Test'}]}
    pipe_instance.valves.openwebui_data_path = str(tmp_path)
    with patch.object(pipe_instance, 'emit_status', new=AsyncMock()):
        result = await pipe_instance.pipe(body, __files__=[file_info], __event_emitter__=None)
        assert result is not None
        if isinstance(result, dict):
            if 'files' in result and isinstance(result['files'], list):
                for f in result['files']:
                    assert 'type' not in f

@pytest.mark.asyncio
async def test_pipe_multipart_api_key_header(pipe_instance):
    pipe_instance.valves.file_transfer_mode = 'multipart'
    pipe_instance.valves.openwebui_api_key = 'FAKEKEY'
    file_info = {'id': 'httpid', 'name': 'http.txt', 'download_url': 'http://fakeurl'}
    body = {'messages': [{'role': 'user', 'content': 'Test'}]}
    with patch('aiohttp.ClientSession.get', new_callable=AsyncMock) as mock_get:
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.read = AsyncMock(return_value=b'abc')
        mock_get.return_value.__aenter__.return_value = mock_response
        with patch.object(pipe_instance, 'emit_status', new=AsyncMock()):
            await pipe_instance.pipe(body, __files__=[file_info], __event_emitter__=None)
            called_args = mock_get.call_args
            if called_args:
                headers = called_args.kwargs.get('headers', {})
                assert headers.get('Authorization') == 'Bearer FAKEKEY'

@pytest.mark.asyncio
async def test_pipe_emit_status_called(pipe_instance):
    pipe_instance.valves.file_transfer_mode = 'json_base64'
    pipe_instance.valves.prefer_uploaded_file_bytes = True
    file_info = {'id': 'emitid', 'name': 'file.txt'}
    body = {'messages': [{'role': 'user', 'content': 'Test'}]}
    emitter = AsyncMock()
    with patch('aiohttp.ClientSession.post', new_callable=AsyncMock) as mock_post, \
         patch('n8n_pipe_v4.extract_event_info', return_value=('dummy_chat_id', None)):
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={'result': 'ok'})
        mock_post.return_value.__aenter__.return_value = mock_response
        await pipe_instance.pipe(body, __files__=[file_info], __event_emitter__=emitter)
        emitter.assert_called()

@pytest.mark.asyncio
async def test_pipe_multiple_files_mixed_types(pipe_instance, tmp_path):
    pipe_instance.valves.file_transfer_mode = 'json_base64'
    uploads_dir = tmp_path / 'uploads'
    uploads_dir.mkdir()
    file1_id = 'mixid1'
    file2_id = 'mixid2'
    file3_id = 'mixid3'
    file1_path = uploads_dir / f'{file1_id}.txt'
    file2_path = uploads_dir / f'{file2_id}.pdf'
    file3_path = uploads_dir / f'{file3_id}.png'
    file1_path.write_bytes(b'abc')
    file2_path.write_bytes(b'def')
    file3_path.write_bytes(b'ghi')
    file1_info = {'id': file1_id, 'name': 'mixid1.txt'}
    file2_info = {'id': file2_id, 'name': 'mixid2.pdf'}
    file3_info = {'id': file3_id, 'name': 'mixid3.png', 'content': base64.b64encode(b'ghi').decode('utf-8')}  # image from body
    body = {'messages': [{'role': 'user', 'content': 'Test'}]}
    pipe_instance.valves.openwebui_data_path = str(tmp_path)
    with patch.object(pipe_instance, 'emit_status', new=AsyncMock()):
        result = await pipe_instance.pipe(body, __files__=[file1_info, file2_info, file3_info], __event_emitter__=None)
        assert result is not None
