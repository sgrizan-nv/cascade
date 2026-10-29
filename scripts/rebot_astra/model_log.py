"""Record adapter inputs and JPEGs without changing the client's chat method."""
from dataclasses import asdict
import json
import time


class RecordingLLM:
    def __init__(self, client, output, min_output_tokens=4096):
        self.client = client
        self.output = output
        self.min_output_tokens = min_output_tokens
        self.requests = []
        self.attempts = 0

    def __getattr__(self, name):
        return getattr(self.client, name)

    def close(self):
        self.client.close()

    def chat(self, system, messages, tools=None, max_tokens=1024):
        self.attempts += 1
        directory = self.output / f'{self.attempts:02d}'
        directory.mkdir(parents=True)
        saved_messages = []
        for i, message in enumerate(messages):
            saved = {k: v for k, v in message.items() if k != 'images'}
            if message.get('tool_calls'):
                saved['tool_calls'] = [asdict(call) for call in message['tool_calls']]
            if message.get('images'):
                saved['images'] = []
                for j, jpeg in enumerate(message['images']):
                    name = f'message-{i:02d}-image-{j:02d}.jpg'
                    (directory / name).write_bytes(jpeg)
                    saved['images'].append(name)
            saved_messages.append(saved)
        max_tokens = max(self.min_output_tokens, max_tokens)
        (directory / 'input.json').write_text(json.dumps({
            'system': system, 'messages': saved_messages, 'tools': tools,
            'max_tokens': max_tokens,
            'format': 'LLM adapter inputs; images are relative JPEG paths, not HTTP data URLs',
        }, indent=2))
        start = time.monotonic()
        try:
            response = self.client.chat(system, messages, tools, max_tokens=max_tokens)
        except Exception as exc:
            # Provider exception strings may contain request details.
            (directory / 'error.json').write_text(json.dumps({
                'type': type(exc).__name__, 'status': getattr(exc, 'status_code', None),
            }))
            raise
        (directory / 'response.json').write_text(json.dumps(asdict(response), indent=2))
        row = {'seconds': round(time.monotonic() - start, 2),
               'images': sum(len(m.get('images') or []) for m in messages),
               'calls': [{'name': c.name, 'arguments': c.arguments} for c in response.tool_calls]}
        self.requests.append(row)
        print(json.dumps(row), flush=True)
        return response
