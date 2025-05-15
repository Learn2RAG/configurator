.PHONY: install test clean
INSTALL_CHECK = .venv/lib/python3.*/site-packages

install: $(INSTALL_CHECK)

test: $(INSTALL_CHECK)
	.venv/bin/python3 -m unittest discover tests

$(INSTALL_CHECK): .venv
	.venv/bin/pip install -r requirements.txt

.venv:
	python3 -m venv .venv

clean:
	rm -rf __pycache__ .venv
