from pathlib import Path
import json
import logging
import os
import signal
import sqlite3
import subprocess

import psutil
import yaml

logger = logging.getLogger(__name__)

# FIXME
# remove child processes immediately when they exit
import platform
import signal

if hasattr(signal, "SIGCHLD"):
    signal.signal(signal.SIGCHLD, signal.SIG_IGN)


init_sql = ['''
CREATE TABLE IF NOT EXISTS projects (
  name TEXT UNIQUE NOT NULL,
  content TEXT NOT NULL,
  running BOOLEAN NOT NULL DEFAULT FALSE
);
''', '''
CREATE TABLE IF NOT EXISTS services (
  project TEXT NOT NULL,
  name TEXT NOT NULL,
  pid INTEGER NOT NULL
);
''']


def init_db(con):
    cur = con.cursor()
    for sql in init_sql:
        cur.execute(sql)


def process_running(pid):
    try:
        process = psutil.Process(pid)
        os.kill(pid, 0)
    except psutil.NoSuchProcess:
        return False
    else:
        return process.is_running()


con = sqlite3.connect(
    os.environ.get('COMPOSE_DB', 'compose.db'),
    # FIXME
    check_same_thread=False,
)
init_db(con)


class Project():
    def create(project_file, name):
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

    def _from_row(row, *, check):
        project = Project()
        project.name = row['name']
        project.content = json.loads(row['content'])
        project.running = row['running']

        if check and project.running:
            project.check()

        return project

    def check(self):
        cur = con.cursor()
        cur.row_factory = sqlite3.Row
        cur.execute('SELECT * FROM services WHERE project = :project', {'project': self.name})
        rows = cur.fetchall()
        cur.close()
        if stopped := list(filter(lambda row: not process_running(row['pid']), rows)):
            logger.info('Stopping project %s due to stopped services: %s', self.name, [row['name'] for row in stopped])
            self.stop(stopped=stopped)

    def get(name, *, check=True):
        cur = con.cursor()
        cur.row_factory = sqlite3.Row
        cur.execute('SELECT * FROM projects WHERE name = ?', (name,))
        row = cur.fetchone()
        cur.close()
        return Project._from_row(row, check=check) if row else None

    def get_all(*, check=True):
        cur = con.cursor()
        cur.row_factory = sqlite3.Row
        cur.execute('SELECT * FROM projects')
        rows = cur.fetchall()
        cur.close()
        return {project.name: project for project in map(lambda row: Project._from_row(row, check=check), rows)}

    def remove(self):
        cur = con.cursor()
        cur.execute('BEGIN EXCLUSIVE')
        cur.execute('DELETE FROM projects WHERE name = :name AND running = FALSE', {'name': self.name})
        if cur.rowcount != 1:
            con.rollback()
            raise AssertionError('Could not delete the project')
        con.commit()
        cur.close()

    def start(self):
        cur = con.cursor()
        cur.execute('BEGIN EXCLUSIVE')
        cur.execute('UPDATE projects SET running = TRUE WHERE name = :name AND running = FALSE', {'name': self.name})
        if cur.rowcount != 1:
            con.rollback()
            raise AssertionError('Could not mark the project as running')

        # files
        try:
            for file in self.content.get('files', []):
                file_path = Path(file['path'])
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
                proc = subprocess.Popen(
                    service['command'],
                    cwd=working_dir,
                    env=os.environ | service.get('environment', {}),
                    start_new_session=True,
                )
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

    def stop(self, stopped=[]):
        cur = con.cursor()
        cur.row_factory = sqlite3.Row
        cur.execute('BEGIN EXCLUSIVE')
        cur.execute('UPDATE projects SET running = FALSE WHERE name = :name AND running = TRUE', {'name': self.name})
        if cur.rowcount != 1:
            con.rollback()
            raise AssertionError(f'Could not mark the project {self.name} as stopped ({cur.rowcount} != 1)')
        cur.execute('SELECT * FROM services WHERE project = :project', {'project': self.name})
        for row in cur.fetchall():
            if row['name'] not in {process['name'] for process in stopped}:
                try:
                    os.killpg(os.getpgid(row['pid']), signal.SIGTERM)
                    # TODO wait for services to actually terminate
                except ProcessLookupError:
                    logger.debug('Attempted to stop a process which does not exist: %s', dict(row))
        cur.execute('DELETE FROM services WHERE project = :project', {'project': self.name})
        con.commit()
        cur.close()
        self.running = False
        logger.debug('Stopped project: %s', self.name)
