'''
Utilities which do not depend on Learn2RAG.
'''
import logging
import platform
import os
import subprocess
from pathlib import Path

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
