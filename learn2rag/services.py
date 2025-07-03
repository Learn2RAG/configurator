import os
import signal
import subprocess

import yaml


class Project():
    def __init__(self, project_file):
        with open(project_file) as f:
            project = yaml.safe_load(f)
            self.services = project['services']
        self.processes = []

    def start(self):
        for name, service in self.services.items():
            self.processes.append(subprocess.Popen(
                service['command'],
                cwd=service['working_dir'],
                env=os.environ | service.get('environment', {}),
                start_new_session=True,
            ))

    def stop(self):
        for process in self.processes:
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
            process.wait()
        self.processes = []
