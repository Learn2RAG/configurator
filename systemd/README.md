```
cp learn2rag.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable learn2rag
systemctl --user restart learn2rag
```
