from __future__ import annotations

import queue
import subprocess
import threading
from collections.abc import Callable, Iterator


def terminate_process(process: subprocess.Popen) -> None:
    """Stop a model child predictably, escalating only after a grace period."""
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=8)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


def controlled_lines(process: subprocess.Popen, checkpoint: Callable[[], None],
                     interval: float = .25) -> Iterator[str]:
    """Drain merged model output without letting a silent process hide controls.

    Reading ``stdout`` directly can block for minutes while a model loads.  A
    dedicated drain thread prevents pipe back-pressure while this iterator calls
    the job checkpoint at a fixed cadence.  Any pause/cancel exception terminates
    the child before it can continue writing partial outputs.
    """
    if process.stdout is None:
        raise RuntimeError("Controlled child process has no output stream")
    pending: queue.Queue[str | None] = queue.Queue()

    def drain() -> None:
        try:
            assert process.stdout is not None
            for value in process.stdout:
                pending.put(value)
        finally:
            pending.put(None)

    reader = threading.Thread(target=drain, name="model-output-drain", daemon=True)
    reader.start()
    reached_eof = False
    try:
        while True:
            checkpoint()
            try:
                value = pending.get(timeout=interval)
            except queue.Empty:
                if process.poll() is not None and not reader.is_alive():
                    reached_eof = True
                    break
                continue
            if value is None:
                reached_eof = True
                break
            yield value
    except BaseException:
        terminate_process(process)
        raise
    finally:
        if not reached_eof:
            terminate_process(process)
        reader.join(timeout=2)
        if process.poll() is not None and process.stdout:
            process.stdout.close()
