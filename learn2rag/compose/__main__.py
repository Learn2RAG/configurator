import argparse

from learn2rag.compose import Project

parser = argparse.ArgumentParser()
parser.add_argument('command')
parser.add_argument('--name', required=True)
parser.add_argument('--project-file')

args = parser.parse_args()
if args.command == 'create':
    assert args.project_file is not None
    Project.create(args.project_file, args.name)
elif args.command == 'start':
    Project.get(args.name).start()
elif args.command == 'stop':
    Project.get(args.name).stop()
elif args.command == 'remove':
    Project.get(args.name).remove()
else:
    raise AssertionError('Unknown command')
