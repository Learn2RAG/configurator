import json
import os
import signal
import sqlite3
import subprocess

import yaml

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


# FIXME
con = sqlite3.connect('compose.db', check_same_thread=False)
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
        return Project.get(name)

    def get(name):
        cur = con.cursor()
        cur.row_factory = sqlite3.Row
        cur.execute('SELECT * FROM projects WHERE name = ?', (name,))
        project = None
        if row := cur.fetchone():
            project = Project()
            project.name = row['name']
            project.content = json.loads(row['content'])
            project.running = row['running']
        cur.close()
        return project

    def get_all():
        cur = con.cursor()
        cur.row_factory = sqlite3.Row
        cur.execute('SELECT * FROM projects')
        projects = {}
        for row in cur.fetchall():
            project = Project()
            project.name = row['name']
            project.content = json.loads(row['content'])
            project.running = row['running']
            projects[project.name] = project
        cur.close()
        return projects

    def remove(self):
        cur = con.cursor()
        cur.execute('BEGIN EXCLUSIVE')
        cur.execute('DELETE FROM projects WHERE name = :name AND running = FALSE', {'name': self.name})
        if cur.rowcount != 1:
            con.rollback()
            raise AssertionError('Could not delete the project')
        con.commit()

    def start(self):
        cur = con.cursor()
        cur.execute('BEGIN EXCLUSIVE')
        cur.execute('UPDATE projects SET running = TRUE WHERE name = :name AND running = FALSE', {'name': self.name})
        if cur.rowcount != 1:
            con.rollback()
            raise AssertionError('Could not mark the project as running')
        self.services = []
        for name, service in self.content['services'].items():
            proc = subprocess.Popen(
                service['command'],
                cwd=service.get('working_dir'),
                env=os.environ | service.get('environment', {}),
                start_new_session=True,
            )
            self.services.append(proc)
            cur.execute('INSERT INTO services (project, name, pid) VALUES (:project, :name, :pid)', {
                'project': self.name,
                'name': name,
                'pid': proc.pid,
            })
            # what if insert failed?
        con.commit()

    def stop(self):
        cur = con.cursor()
        cur.row_factory = sqlite3.Row
        cur.execute('BEGIN EXCLUSIVE')
        cur.execute('UPDATE projects SET running = FALSE WHERE name = :name AND running = TRUE', {'name': self.name})
        if cur.rowcount != 1:
            con.rollback()
            raise AssertionError('Could not mark the project as stopped')
        cur.execute('SELECT * FROM services WHERE project = :project', {'project': self.name})
        for row in cur.fetchall():
            try:
                os.killpg(os.getpgid(row['pid']), signal.SIGTERM)
            except ProcessLookupError:
                print('Process does not exist: ', dict(row))
            # TODO wait for services to actually terminate
        cur.execute('DELETE FROM services WHERE project = :project', {'project': self.name})
        con.commit()
