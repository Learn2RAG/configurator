'''
Utilities which do not depend on Learn2RAG.
'''
import logging
import platform
import os
import subprocess
from pathlib import Path
from time import sleep
from typing import Callable, Optional

import xdg.BaseDirectory


def is_windows() -> bool:
    return platform.system() == 'Windows'


def normalize_path(path: Path) -> Path:
    'Expand ~ and ~user constructs; make the path absolute, resolving symlinks'
    return Path(path).expanduser().resolve()


def open_web_browser(url: str) -> None:
    'Tries to open the specified URL in a web browser'
    try:
        if not is_windows():
            subprocess.Popen(['xdg-open', url])
        else:
            subprocess.Popen(['explorer', url])
    except FileNotFoundError:
        pass
    except Exception:
        logging.error('Unable to open the web browser', exc_info=True)


def save_data_path(*resource: str) -> str:
    'Returns the application data path for the specified resource'
    if not is_windows():
        return xdg.BaseDirectory.save_data_path(*resource)
    else:
        windows_app_data = os.getenv('LOCALAPPDATA')
        assert windows_app_data is not None
        return os.path.join(windows_app_data, *resource)


# adapted from pytestqt
def waitUntil(
    callback: Callable[[], Optional[bool]], *, timeout: int = 5000
) -> None:
    """
    .. versionadded:: 2.0

    Wait in a busy loop, calling the given callback periodically until timeout is reached.

    ``callback()`` should raise ``AssertionError`` to indicate that the desired condition
    has not yet been reached, or just return ``None`` when it does. Useful to ``assert`` until
    some condition is satisfied:

    .. code-block:: python

        def view_updated():
            assert view_model.count() > 10


        qtbot.waitUntil(view_updated)

    Another possibility is for ``callback()`` to return ``True`` when the desired condition
    is met, ``False`` otherwise. Useful specially with ``lambda`` for terser code, but keep
    in mind that the error message in those cases is usually not very useful because it is
    not using an ``assert`` expression.

    .. code-block:: python

        qtbot.waitUntil(lambda: view_model.count() > 10)

    Note that this usage only accepts returning actual ``True`` and ``False`` values,
    so returning an empty list to express "falseness" raises a ``ValueError``.

    :param callback: callable that will be called periodically.
    :param timeout: timeout value in ms.
    :raises ValueError: if the return value from the callback is anything other than ``None``,
        ``True`` or ``False``.

    .. note:: This method is also available as ``wait_until`` (pep-8 alias)
    """
    __tracebackhide__ = True
    import time

    start = time.time()

    def timed_out() -> bool:
        elapsed = time.time() - start
        elapsed_ms = elapsed * 1000
        return elapsed_ms > timeout

    timeout_msg = f"waitUntil timed out in {timeout} milliseconds"

    while True:
        try:
            result = callback()
        except AssertionError as e:
            if timed_out():
                raise TimeoutError(timeout_msg) from e
        else:
            if result not in (None, True, False):
                msg = "waitUntil() callback must return None, True or False, returned %r"
                raise ValueError(msg % result)

            # 'assert' form
            if result is None:
                return

            # 'True/False' form
            if result:
                return
            if timed_out():
                raise TimeoutError(timeout_msg)
        sleep(10)
