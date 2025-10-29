import numpy as np
import queue as _q
from txori.audio import StreamAudioSource

class _Q:
    def __init__(self, maxsize=1):
        self._d = []
        self._max = maxsize
    def put(self, v, block=False):
        if len(self._d) >= self._max:
            raise _q.Full
        self._d.append(v)
    def put_nowait(self, v):
        return self.put(v, block=False)
    def get(self, block=True):
        return self._d.pop(0)
    def get_nowait(self):
        return self.get(block=False)


def test_stream_queue_full_branch(monkeypatch):
    src = StreamAudioSource(sample_rate=8000, channels=1, blocksize=16)

    def _input_stream(**kwargs):
        class _S:
            def __enter__(self):
                for _ in range(5):
                    data = np.random.rand(16, 1).astype(np.float32)
                    kwargs["callback"](data, 16, None, None)
                return self
            def __exit__(self, *a):
                return False
        return _S()

    monkeypatch.setattr("txori.audio.queue.Queue", lambda maxsize=10: _Q(maxsize=1))
    monkeypatch.setattr("txori.audio.sd.InputStream", _input_stream)

    it = src.blocks()
    # Consumir algunos bloques sin excepciones
    for _ in range(3):
        b = next(it)
        assert b.shape == (16,)
