import argparse
from pprint import pprint

from learn2rag.compose import Project

parser = argparse.ArgumentParser()
parser.add_argument('command')
parser.add_argument('--name')
parser.add_argument('--project-file')

args = parser.parse_args()
if args.command == 'create':
    assert args.name is not None
    assert args.project_file is not None
    proj = Project.create(args.project_file, args.name)
    if proj is not None:
        print(proj)
    else:
        print('Failed')
elif args.command == 'start':
    assert args.name is not None
    project = Project.get(args.name)
    assert project is not None
    project.start()
elif args.command == 'stop':
    assert args.name is not None
    project = Project.get(args.name)
    assert project is not None
    project.stop()
elif args.command == 'remove':
    assert args.name is not None
    project = Project.get(args.name)
    assert project is not None
    project.remove()
elif args.command == 'list':
    pprint(Project.get_all())
else:
    raise AssertionError('Unknown command')
