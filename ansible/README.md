Install required dependencies.

Start with the example inventory file:
```
cp hosts.example.yml myhosts.yml
```

Configure your server hostname:
```
sed -i -e 's|myserver.example|actualserver.example|' myhosts.yml
```

Configure the working directory:
```
sed -i -e 's|#learn2rag_dir: .*|learn2rag_dir: /data/learn2rag|' myhosts.yml
```

Install and start:
```
ansible-playbook --inventory myhosts.yml learn2rag.yml
```

Next steps:
- set up config.yml
