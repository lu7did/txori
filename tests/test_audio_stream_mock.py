import numpy as np
from txori.audio import StreamAudioSource


class _Q:
    def __init__(self):
        self._d = []

    def put(self, v, block: bool = False):
        self._d.append(v)

    def get(self):
        return self._d.pop(0)


def test_stream_blocks_iterator(monkeypatch):
    src = StreamAudioSource(sample_rate=8000, channels=1, blocksize=16)

    def _input_stream(**kwargs):
        class _S:
            def __enter__(self):
                # Emula 5 callbacks entregando bloques
                for _ in range(5):
                    data = np.random.rand(16, 1).astype(np.float32)
                    kwargs["callback"](data, 16, None, None)
                return self

            def __exit__(self, *a):  # noqa: D401
                return False

        return _S()

    # Parchar cola para control determinista
    monkeypatch.setattr("txori.audio.queue.Queue", lambda maxsize=10: _Q())
    monkeypatch.setattr("txori.audio.sd.InputStream", _input_stream)

    it = src.blocks()
    b1 = next(it)
    b2 = next(it)
    assert b1.shape == (16,)
    assert b2.shape == (16,)
