"""Experiment contracts: retain exact inputs and do not mislabel verification."""
import argparse
import json
from types import SimpleNamespace

import pytest

from cascade.agent.llm import LLMResponse, ToolCall
from scripts.rebot_astra.experiment import cleanup, task_contract, verdict
from scripts.rebot_astra.model_log import RecordingLLM


def test_logging_preserves_tool_history_and_image_bytes_without_mutation(tmp_path):
    calls = []
    reply = LLMResponse(tool_calls=[ToolCall('task_done', {'success': True}, 'done')])
    def chat(system, messages, tools, max_tokens):
        calls.append((system, messages, tools, max_tokens))
        return reply
    client = SimpleNamespace(chat=chat, supports_vision=True, model='test')
    recorder = RecordingLLM(client, tmp_path)
    tc = ToolCall('get_observation', {}, 'observe')
    messages = [{'role': 'assistant', 'tool_calls': [tc]},
                {'role': 'user', 'content': 'current view', 'images': [b'exact JPEG bytes']}]
    tools = [{'name': 'get_observation'}]
    assert recorder.chat('system', messages, tools) is reply
    assert client.chat is chat
    assert calls[0][1] is messages
    assert messages[0]['tool_calls'][0] is tc
    saved = json.loads((tmp_path / '01/input.json').read_text())
    assert saved['messages'][0]['tool_calls'][0]['id'] == 'observe'
    image = saved['messages'][1]['images'][0]
    assert (tmp_path / '01' / image).read_bytes() == b'exact JPEG bytes'
    assert json.loads((tmp_path / '01/response.json').read_text())['tool_calls'][0]['name'] == 'task_done'


def test_failed_request_gets_its_own_directory(tmp_path):
    def fail(*args, **kwargs):
        raise RuntimeError('provider failure')
    recorder = RecordingLLM(SimpleNamespace(chat=fail), tmp_path)
    for _ in range(2):
        with pytest.raises(RuntimeError):
            recorder.chat('system', [])
    assert (tmp_path / '01/error.json').exists()
    assert (tmp_path / '02/input.json').exists()


def test_custom_task_requires_explicit_verifier():
    args = SimpleNamespace(task='Observe the scene', verify=None, plate=False, inspect_only=False)
    with pytest.raises(SystemExit):
        task_contract(args, argparse.ArgumentParser())
    args.verify = 'none'
    assert task_contract(args, argparse.ArgumentParser()) == ('Observe the scene', 'none')
    assert verdict('none', True, 0, None) == {
        'mode': 'none', 'independent_success': None, 'success': True}
    assert not verdict('lift', True, 0, None)['success']
    assert not verdict('plate', True, .1, {'passed': False})['success']
    assert not verdict('plate', False, .1, {'passed': True})['success']


def test_recorder_failure_does_not_skip_runtime_or_llm_cleanup(tmp_path, monkeypatch):
    from cascade.apps import demo
    calls = []
    def stop():
        calls.append('recorder')
        raise RuntimeError('encoder failed')
    monkeypatch.setattr(demo, 'shutdown_runtime', lambda *_: calls.append('runtime'))
    llm = SimpleNamespace(close=lambda: calls.append('llm'))
    output = tmp_path / 'not-created-before-failure'
    receipt = cleanup(output, recorder=SimpleNamespace(stop=stop),
                      runtime=object(), arm=object(), llm=llm)
    assert calls == ['recorder', 'runtime', 'llm']
    assert not receipt['ok']
    assert json.loads((output / 'teardown.json').read_text()) == receipt
