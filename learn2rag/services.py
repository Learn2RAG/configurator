import os
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
                env=os.environ | service['environment'],
            ))

    def stop(self):
        for process in self.processes:
            # TODO needs killing all child processes
            process.terminate()
            process.wait()
        self.processes = []
