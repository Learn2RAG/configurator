from pathlib import Path
import contextlib
import json
import logging
import os
import signal
import sqlite3
import subprocess
import urllib.request
from typing import Any

import psutil
import yaml

logger = logging.getLogger(__name__)

# FIXME
# remove child processes immediately when they exit
import platform
import signal


exit_statuses = {}

if hasattr(signal, "SIGCHLD"):
    def handle_SIGCHLD(*args):
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
        os.killpg(os.getpgid(pid), signal.SIGTERM)
    else:
        os.kill(pid, signal.SIGTERM)


def init_db(con: sqlite3.Connection) -> None:
    cur = con.cursor()
    for sql in init_sql:
        try:
            cur.execute(sql)
        except sqlite3.OperationalError as e:
            logger.warning('Database initialization: %s', e)


def process_running(pid: int) -> bool:
    # if process := popens.get(pid):
    #     returncode = process.poll()
    #     print(returncode)
    #     return returncode == None
    try:
        process = psutil.Process(pid)
        os.kill(pid, 0)
    except psutil.NoSuchProcess:
        return False
    else:
        return process.is_running()


def healthy(value: list[str]) -> bool:
    assert len(value) == 4
    assert value[0:3] == ['CMD', 'curl' ,'-f']
    url = value[3]
    try:
        with urllib.request.urlopen(url, timeout=1):
            return True
    except Exception:
        return False


con = sqlite3.connect(
    os.environ.get('COMPOSE_DB', 'compose.db'),
    # FIXME
    check_same_thread=False,
)
init_db(con)


class Project():
    name: str
    content: dict[str, Any]

    @staticmethod
    def create(project_file: str | Path, name: str) -> 'Project | None':
        with open(project_file) as f:
            content = yaml.safe_load(f)
        assert len(content['services']) > 0
        cur = con.cursor()
        cur.execute('BEGIN EXCLUSIVE')
        try:
            cur.execute('INSERT INTO projects (name, content) VALUES (:name, :content)', {'name': name, 'content': json.dumps(content)})
        except sqlite3.IntegrityError as e:
            con.rollback()
            raise e
        con.commit()
        cur.close()
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

    def check(self) -> None:
        cur = con.cursor()
        cur.row_factory = sqlite3.Row  # type: ignore[assignment]
        cur.execute('SELECT * FROM services WHERE project = :project', {'project': self.name})
        rows = cur.fetchall()
        cur.close()
        if stopped := list(filter(lambda row: not process_running(row['pid']), rows)):
            logger.info('Stopping project %s due to stopped services: %s', self.name, [row['name'] for row in stopped])
            self.stop(stopped=stopped)

    def healthcheck(self) -> None:
        # what it should be when there are no healthchecks defined?
        self.health = all(map(healthy, filter(None, (service.get('healthcheck', {}).get('test') for service in self.content['services'].values()))))

    @staticmethod
    def get(name: str, *, check: bool=True) -> 'Project | None':
        cur = con.cursor()
        cur.row_factory = sqlite3.Row  # type: ignore[assignment]
        cur.execute('SELECT * FROM projects WHERE name = ?', (name,))
        row = cur.fetchone()
        cur.close()
        return Project._from_row(row, check=check) if row else None

    @staticmethod
    def get_all(*, check: bool=True) -> dict[str, 'Project']:
        cur = con.cursor()
        cur.row_factory = sqlite3.Row  # type: ignore[assignment]
        cur.execute('SELECT * FROM projects')
        rows = cur.fetchall()
        cur.close()
        return {project.name: project for project in map(lambda row: Project._from_row(row, check=check), rows)}

    def remove(self) -> None:
        cur = con.cursor()
        cur.execute('BEGIN EXCLUSIVE')
        cur.execute('DELETE FROM projects WHERE name = :name AND running = FALSE', {'name': self.name})
        if cur.rowcount != 1:
            con.rollback()
            raise AssertionError('Could not delete the project')
        con.commit()
        cur.close()

    def start(self) -> None:
        cur = con.cursor()
        cur.execute('BEGIN EXCLUSIVE')
        cur.execute('UPDATE projects SET running = TRUE WHERE name = :name AND running = FALSE', {'name': self.name})
        if cur.rowcount != 1:
            con.rollback()
            raise AssertionError('Could not mark the project as running')

        # files
        try:
            for file in self.content.get('files', []):
                file_path = Path(file['path']).expanduser().absolute()
                file_path.parent.mkdir(parents=True, exist_ok=True)
                file_path.write_text(file['content'])
        except Exception as e:
            con.rollback()
            raise e

        # services
        self.services = []
        for name, service in self.content['services'].items():
            # working_dir
            if working_dir := service.get('working_dir'):
                Path(working_dir).mkdir(parents=True, exist_ok=True)

            # command
            try:
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
            except Exception as e:
                # TODO stop already started processes immediately
                con.rollback()
                raise e
            self.services.append(proc)
            cur.execute('INSERT INTO services (project, name, pid) VALUES (:project, :name, :pid)', {
                'project': self.name,
                'name': name,
                'pid': proc.pid,
            })
            del proc
            # what if insert failed?
        con.commit()
        cur.close()
        self.running = True

    def stop(self, stopped: list[dict[str, Any]]=[]) -> None:
        # FIXME: consider only main process exiting with exit code 0 as success
        self.succeeded = len(stopped) != 0 and all(process['pid'] in exit_statuses for process in stopped) and all(exit_statuses[process['pid']] == 0 for process in stopped)
        self.failed = any(process['pid'] in exit_statuses and exit_statuses[process['pid']] != 0 for process in stopped)
        cur = con.cursor()
        cur.row_factory = sqlite3.Row  # type: ignore[assignment]
        cur.execute('BEGIN EXCLUSIVE')
        cur.execute('UPDATE projects SET running = FALSE, succeeded = :succeeded, failed = :failed WHERE name = :name AND running = TRUE', {
            'name': self.name,
            'succeeded': self.succeeded,
            'failed': self.failed,
        })
        if cur.rowcount != 1:
            con.rollback()
            raise AssertionError(f'Could not mark the project {self.name} as stopped ({cur.rowcount} != 1)')
        cur.execute('SELECT * FROM services WHERE project = :project', {'project': self.name})
        for row in cur.fetchall():
            if row['name'] not in {process['name'] for process in stopped}:
                try:
                    kill_process(row['pid'])
                    # TODO wait for services to actually terminate
                except ProcessLookupError:
                    logger.debug('Attempted to stop a process which does not exist: %s', dict(row))
                except Exception as e:
                    logger.debug('Attempted to stop a process but got exception: %s, %s', dict(row), e)
        cur.execute('DELETE FROM services WHERE project = :project', {'project': self.name})
        con.commit()
        cur.close()
        self.running = False
        logger.debug('Stopped project: %s', self.name)
