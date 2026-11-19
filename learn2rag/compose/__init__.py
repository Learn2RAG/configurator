from pathlib import Path
import contextlib
import json
import logging
import os
import signal
import sqlite3
import ssl
import subprocess
import threading
import urllib.request
from typing import Any, Iterator, Optional

import jinja2
import psutil
import yaml

logger = logging.getLogger(__name__)
active_popens: dict[int, subprocess.Popen[Any]] = {}
# FIXME
# remove child processes immediately when they exit
import platform
import signal


exit_statuses = {}
# Windows does not support SIGCHLD
if hasattr(signal, "SIGCHLD"):
    def handle_SIGCHLD(signum: int, frame: Optional[Any]) -> None:
        with contextlib.suppress(ChildProcessError):
            while True:
                pid, status = os.waitpid(-1, os.WNOHANG)
                if pid == 0: break
                exit_statuses[pid] = status
    signal.signal(signal.SIGCHLD, handle_SIGCHLD)


init_sql = ['''
CREATE TABLE IF NOT EXISTS projects (
  name TEXT UNIQUE NOT NULL,
  content TEXT NOT NULL,
  running BOOLEAN NOT NULL DEFAULT FALSE
);
''', '''
ALTER TABLE projects ADD succeeded BOOLEAN NOT NULL DEFAULT FALSE;
''', '''
ALTER TABLE projects ADD failed BOOLEAN NOT NULL DEFAULT FALSE;
''', '''
CREATE TABLE IF NOT EXISTS services (
  project TEXT NOT NULL,
  name TEXT NOT NULL,
  pid INTEGER NOT NULL
);
''']


def kill_process(pid: int) -> None:
    if hasattr(os, 'killpg'):
        logger.debug('Using os.killpg for process group %s', pid)
        os.killpg(os.getpgid(pid), signal.SIGTERM)
    else:
        try:
            parent = psutil.Process(pid)
            procs = parent.children(recursive=True)
            procs.append(parent)

            logger.debug('Found process tree for %s with %d processes', pid, len(procs))

            for p in procs:
                try:
                    logger.debug('Terminating process %s', p.pid)
                    p.terminate()
                except psutil.NoSuchProcess:
                    pass

            logger.debug('Waiting up to 5 seconds for processes to terminate and release ports...')
            gone, alive = psutil.wait_procs(procs, timeout=5)

            if alive:
                logger.warning('%d processes refused to terminate gracefully. Forcing kill.', len(alive))
                for p in alive:
                    try:
                        logger.debug('Force killing process %s', p.pid)
                        p.kill()
                    except psutil.NoSuchProcess:
                        pass
            else:
                logger.debug('All processes in tree %s terminated gracefully.', pid)

        except psutil.NoSuchProcess:
            logger.debug('Process %s already stopped, nothing to kill.', pid)


def init_db(con: sqlite3.Connection) -> None:
    cur = con.cursor()
    for sql in init_sql:
        try:
            cur.execute(sql)
        except sqlite3.OperationalError as e:
            logger.debug('Database initialization: %s', e)


def process_running(pid: int) -> bool:
    # I comment it because it seems for windows kill(pid,0) does not work
    # try:
    #     process = psutil.Process(pid)
    #     os.kill(pid, 0)
    # except psutil.NoSuchProcess:
    #     return False
    # else:
    #     return process.is_running()
    try:
        process = psutil.Process(pid)
    except psutil.NoSuchProcess:
        return False
    else:
        return process.is_running()

def healthy(value: list[str]) -> bool:
    assert len(value) == 4
    assert value[0:3] == ['CMD', 'curl' ,'-f']
    url = value[3]
    try:
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        with urllib.request.urlopen(url, timeout=1, context=ctx):
            return True
    except Exception:
        return False


# Resolved once so every thread opens the same file even if the CWD changes later.
DB_PATH = Path(os.environ.get('COMPOSE_DB', 'compose.db')).absolute()
# How long a connection waits for another thread/process holding a write lock.
DB_TIMEOUT = 30.0

_local = threading.local()


def _connect() -> sqlite3.Connection:
    con = sqlite3.connect(DB_PATH, timeout=DB_TIMEOUT)
    con.row_factory = sqlite3.Row
    return con


def get_connection() -> sqlite3.Connection:
    """Return this thread's own connection to the compose database.

    sqlite3 connections must not be used by several threads at once, and the UI
    serves requests from a thread pool, so each thread gets its own connection.
    SQLite's file locking (plus the busy timeout) coordinates between them, and
    between separate processes using the same database.
    """
    con: sqlite3.Connection | None = getattr(_local, 'con', None)
    if con is None:
        con = _local.con = _connect()
    return con


@contextlib.contextmanager
def transaction() -> Iterator[sqlite3.Cursor]:
    """Run a write transaction on this thread's connection; commit, or roll back on error.

    BEGIN IMMEDIATE takes the write lock up front, so concurrent writers wait
    (up to DB_TIMEOUT) instead of failing halfway through.
    """
    con = get_connection()
    cur = con.cursor()
    cur.execute('BEGIN IMMEDIATE')
    try:
        yield cur
    except BaseException:
        con.rollback()
        raise
    else:
        con.commit()
    finally:
        cur.close()


class ProjectNotRunning(AssertionError):
    """The project was already stopped, e.g. by another thread."""


def _setup_db() -> None:
    with contextlib.closing(_connect()) as con:
        # WAL lets readers proceed while another connection is writing.
        con.execute('PRAGMA journal_mode=WAL')
        init_db(con)
        con.commit()


_setup_db()


class Project():
    name: str
    content: dict[str, Any]

    @staticmethod
    def create(
            compose_file: str | Path,
            name: str,
            *,
            template: bool = False,
            template_context: dict[str, Any] = {},
    ) -> 'Project | None':
        if template:
            content = yaml.safe_load(jinja2.Template(Path(compose_file).read_text()).render(template_context))
        else:
            with open(compose_file) as f:
                content = yaml.safe_load(f)
        assert len(content['services']) > 0
        with transaction() as cur:
            cur.execute('INSERT INTO projects (name, content) VALUES (:name, :content)', {'name': name, 'content': json.dumps(content)})
        return Project.get(name, check=False)

    @staticmethod
    def _from_row(row: sqlite3.Row, *, check: bool) -> 'Project':
        project = Project()
        project.name = row['name']
        project.content = json.loads(row['content'])
        project.running = row['running']
        project.health = False
        project.succeeded = bool(row['succeeded'])
        project.failed = bool(row['failed'])

        if check and project.running:
            project.check()

        if project.running:
            project.healthcheck()

        return project
    # TODO : check for memory leak
    # If a process stops on its own, check() runs and cleans up the dictionary. But if a user manually stops a project (triggering Project.stop() directly), check() never runs. The processes are killed, but their handles stay in active_popens forever, causing a memory leak
    def check(self) -> None:
        con = get_connection()
        cur = con.cursor()
        cur.execute('SELECT * FROM services WHERE project = :project', {'project': self.name})
        rows = cur.fetchall()
        cur.close()

        if not hasattr(signal, "SIGCHLD"):
            for row in rows:
                pid = row['pid']
                if pid in active_popens:
                    ret = active_popens[pid].poll()
                    if ret is not None:
                        exit_statuses[pid] = ret

        if stopped := list(filter(lambda row: not process_running(row['pid']), rows)):
            logger.info('Stopping project %s due to stopped services: %s', self.name, [row['name'] for row in stopped])
            try:
                self.stop(stopped=stopped)
            except ProjectNotRunning:
                # Another thread noticed the same stopped services and stopped it first.
                logger.debug('Project %s was already stopped', self.name)
                self.running = False

            for process in stopped:
                pid = process['pid']
                if pid in active_popens:
                    del active_popens[pid]

    def healthcheck(self) -> None:
        # what it should be when there are no healthchecks defined?
        self.health = all(map(healthy, filter(None, (service.get('healthcheck', {}).get('test') for service in self.content['services'].values()))))

    @staticmethod
    def get(name: str, *, check: bool=True) -> 'Project | None':
        con = get_connection()
        cur = con.cursor()
        cur.execute('SELECT * FROM projects WHERE name = ?', (name,))
        row = cur.fetchone()
        cur.close()
        return Project._from_row(row, check=check) if row else None

    @staticmethod
    def get_all(*, check: bool=True) -> dict[str, 'Project']:
        con = get_connection()
        cur = con.cursor()
        cur.execute('SELECT * FROM projects')
        rows = cur.fetchall()
        cur.close()
        return {project.name: project for project in map(lambda row: Project._from_row(row, check=check), rows)}

    def remove(self) -> None:
        with transaction() as cur:
            cur.execute('DELETE FROM projects WHERE name = :name AND running = FALSE', {'name': self.name})
            if cur.rowcount != 1:
                raise AssertionError('Could not delete the project')

    def start(self) -> None:
        # Claim the project first, so two threads cannot start it at the same time.
        with transaction() as cur:
            cur.execute('UPDATE projects SET running = TRUE WHERE name = :name AND running = FALSE', {'name': self.name})
            if cur.rowcount != 1:
                raise AssertionError('Could not mark the project as running')

        # Writing files and launching processes can be slow, so it happens outside any transaction.
        self.services = []
        launched: list[tuple[str, subprocess.Popen[Any]]] = []
        try:
            # files
            for file in self.content.get('files', []):
                file_path = Path(file['path']).expanduser().absolute()
                if not file_path.exists() or file.get('force', True):
                    file_path.parent.mkdir(parents=True, exist_ok=True)
                    if 'content' in file:
                        content = file['content']
                    elif 'src' in file:
                        content = Path(file['src']).read_text()
                    else:
                        raise NotImplementedError(file)
                    file_path.write_text(content)

            # services
            for name, service in self.content['services'].items():
                # working_dir
                if working_dir := service.get('working_dir'):
                    Path(working_dir).mkdir(parents=True, exist_ok=True)

                # command
                popen_args = dict(
                    cwd=working_dir,
                    env=os.environ | service.get('environment', {}),
                )
                if hasattr(subprocess, 'CREATE_NEW_PROCESS_GROUP'):
                    popen_args |= dict(
                        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
                    )
                else:
                    popen_args |= dict(
                        start_new_session=True,
                    )
                proc = subprocess.Popen(service['command'], **popen_args)
                # we dont delete because for windows we need to retrive the exit code
                active_popens[proc.pid] = proc
                launched.append((name, proc))
                self.services.append(proc)

            with transaction() as cur:
                cur.executemany('INSERT INTO services (project, name, pid) VALUES (:project, :name, :pid)', [
                    {'project': self.name, 'name': name, 'pid': proc.pid} for name, proc in launched
                ])
        except BaseException:
            # Undo: stop whatever was launched and release the claim on the project.
            for _, proc in launched:
                with contextlib.suppress(Exception):
                    kill_process(proc.pid)
            with transaction() as cur:
                cur.execute('DELETE FROM services WHERE project = :project', {'project': self.name})
                cur.execute('UPDATE projects SET running = FALSE WHERE name = :name', {'name': self.name})
            raise
        self.running = True

    def stop(self, stopped: list[dict[str, Any]]=[]) -> None:
        # FIXME: consider only main process exiting with exit code 0 as success
        self.succeeded = len(stopped) != 0 and all(process['pid'] in exit_statuses for process in stopped) and all(exit_statuses[process['pid']] == 0 for process in stopped)
        self.failed = any(process['pid'] in exit_statuses and exit_statuses[process['pid']] != 0 for process in stopped)
        with transaction() as cur:
            cur.execute('UPDATE projects SET running = FALSE, succeeded = :succeeded, failed = :failed WHERE name = :name AND running = TRUE', {
                'name': self.name,
                'succeeded': self.succeeded,
                'failed': self.failed,
            })
            if cur.rowcount != 1:
                raise ProjectNotRunning(f'Could not mark the project {self.name} as stopped ({cur.rowcount} != 1)')
            cur.execute('SELECT * FROM services WHERE project = :project', {'project': self.name})
            services = cur.fetchall()
            cur.execute('DELETE FROM services WHERE project = :project', {'project': self.name})

        # Killing can take a while (up to 5 s on Windows), so it happens after the commit.
        stopped_names = {process['name'] for process in stopped}
        for row in services:
            if row['name'] not in stopped_names:
                try:
                    kill_process(row['pid'])
                    # TODO wait for services to actually terminate
                except (ProcessLookupError, OSError):
                    logger.debug('Attempted to stop a process which does not exist: %s', dict(row))
                except Exception as e:
                    logger.debug('Attempted to stop a process but got exception: %s, %s', dict(row), e)
        self.running = False
        logger.debug('Stopped project: %s', self.name)
