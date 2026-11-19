"""Thread-safety of the compose database.

The admin UI serves requests from a thread pool, so Project methods are called
from several threads at once. These tests reproduce that.
"""
import sqlite3
import sys
import threading
import time
from collections.abc import Callable, Iterator
from pathlib import Path

import psutil
import pytest
import yaml

from .. import Project, ProjectNotRunning, active_popens


def sleep_command(seconds: float) -> list[str]:
    return [sys.executable, '-c', f'import time; time.sleep({seconds})']


def write_project(path: Path, services: dict[str, list[str]]) -> Path:
    path.write_text(yaml.safe_dump(
        {'services': {name: {'command': command} for name, command in services.items()}},
        sort_keys=False,
    ))
    return path


def wait_for(condition: Callable[[], bool], timeout: float = 10.0) -> bool:
    # Sleeping in the main thread also lets the SIGCHLD handler reap exited services.
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(0.05)
    return condition()


def run_threads(targets: list[Callable[[], None]], timeout: float = 60.0) -> list[BaseException]:
    errors: list[BaseException] = []

    def wrap(target: Callable[[], None]) -> Callable[[], None]:
        def run() -> None:
            try:
                target()
            except BaseException as e:
                errors.append(e)
        return run

    threads = [threading.Thread(target=wrap(target), daemon=True) for target in targets]
    for thread in threads:
        thread.start()
    assert wait_for(lambda: not any(thread.is_alive() for thread in threads), timeout), 'threads did not finish'
    return errors


@pytest.fixture
def long_project(tmp_path: Path) -> Path:
    return write_project(tmp_path / 'long.yml', {'main': sleep_command(60)})


@pytest.fixture
def short_project(tmp_path: Path) -> Path:
    # 'quick' exits on its own, which makes check() stop the whole project.
    return write_project(tmp_path / 'short.yml', {'quick': sleep_command(0.3), 'main': sleep_command(60)})


@pytest.fixture(autouse=True)
def cleanup() -> Iterator[None]:
    yield
    for project in Project.get_all(check=False).values():
        if project.running:
            project.stop()
        project.remove()


def test_concurrent_reads(long_project: Path) -> None:
    """Many threads reading at once used to fail with 'IndexError: tuple index out of range'."""
    for i in range(3):
        project = Project.create(long_project, f'long{i}')
        assert project is not None
        project.start()

    def read() -> None:
        for _ in range(100):
            assert set(Project.get_all()) == {'long0', 'long1', 'long2'}
            project = Project.get('long1')
            assert project is not None and project.running

    assert run_threads([read] * 16) == []


def test_concurrent_reads_and_writes(long_project: Path, short_project: Path) -> None:
    for i in range(3):
        project = Project.create(long_project, f'long{i}')
        assert project is not None
        project.start()

    done = threading.Event()

    def read() -> None:
        while not done.is_set():
            Project.get_all()
            Project.get('long1')

    def write(worker: int) -> Callable[[], None]:
        def run() -> None:
            for i in range(5):
                name = f'w{worker}_{i}'
                project = Project.create(short_project, name)
                assert project is not None
                project.start()
                # The readers race each other to notice 'quick' exiting and stop the project.
                def is_stopped() -> bool:
                    current = Project.get(name)
                    assert current is not None
                    return not current.running
                for _ in range(100):
                    if is_stopped():
                        break
                    time.sleep(0.05)
                assert is_stopped(), f'{name} was never stopped'
                project.remove()
        return run

    def writers() -> None:
        try:
            assert run_threads([write(0), write(1)]) == []
        finally:
            done.set()

    assert run_threads([writers] + [read] * 8) == []
    assert set(Project.get_all()) == {'long0', 'long1', 'long2'}


def test_stop_already_stopped_project(short_project: Path) -> None:
    project = Project.create(short_project, 'race')
    assert project is not None
    project.start()

    # Two requests loaded the project while it was still running.
    first = Project.get('race', check=False)
    second = Project.get('race', check=False)
    assert first is not None and second is not None

    first.stop()
    with pytest.raises(ProjectNotRunning):
        second.stop()


def test_check_tolerates_concurrent_stop(short_project: Path) -> None:
    """Two threads noticing the same exited service used to raise 'Could not mark the project as stopped'."""
    project = Project.create(short_project, 'race')
    assert project is not None
    project.start()
    quick_pid = project.services[0].pid

    first = Project.get('race', check=False)
    second = Project.get('race', check=False)
    assert first is not None and second is not None
    assert wait_for(lambda: not psutil.pid_exists(quick_pid))  # exited and reaped

    first.check()
    assert not first.running
    second.check()  # must not raise
    assert not second.running

    stored = Project.get('race')
    assert stored is not None and not stored.running


def test_failed_start_cleans_up(tmp_path: Path) -> None:
    project_file = write_project(tmp_path / 'broken.yml', {
        'main': sleep_command(60),
        'broken': [str(tmp_path / 'does-not-exist')],
    })
    project = Project.create(project_file, 'broken')
    assert project is not None

    before = set(active_popens)
    with pytest.raises(OSError):
        project.start()

    # The service that did launch was stopped again.
    launched = set(active_popens) - before
    assert len(launched) == 1
    for pid in launched:
        assert wait_for(lambda: not psutil.pid_exists(pid) or psutil.Process(pid).status() == psutil.STATUS_ZOMBIE)

    # The project is not left marked as running, so it can be removed or started again.
    stored = Project.get('broken')
    assert stored is not None and not stored.running
    stored.remove()
    assert Project.get('broken') is None


def test_failed_write_does_not_leave_transaction_open(long_project: Path) -> None:
    assert Project.create(long_project, 'dup') is not None
    with pytest.raises(sqlite3.IntegrityError):
        Project.create(long_project, 'dup')

    # The same thread's connection must still accept new transactions.
    assert Project.create(long_project, 'other') is not None
    assert set(Project.get_all()) == {'dup', 'other'}


def test_uses_isolated_database(compose_db: Path) -> None:
    from .. import get_connection
    (path,) = [row['file'] for row in get_connection().execute('PRAGMA database_list') if row['name'] == 'main']
    assert Path(path) == compose_db

