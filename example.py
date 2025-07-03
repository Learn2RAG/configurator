import atexit

from learn2rag.services import Project


p = Project('example.yml')
atexit.register(p.stop)
p.start()
input()
