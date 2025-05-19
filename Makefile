.PHONY: install test clean
INSTALL_CHECK = .venv/lib/python3.*/site-packages

install: $(INSTALL_CHECK)

test: $(INSTALL_CHECK) tests/data/wikibooks/pages-articles.xml.bz2
	.venv/bin/mypy .
	#.venv/bin/python3 -m unittest discover tests

$(INSTALL_CHECK): .venv
	.venv/bin/pip install -r requirements.txt

tests/data/wikibooks/pages-articles.xml.bz2:
	mkdir --parents $$(dirname $@)
	wget --output-document $@ "https://files.dice-research.org/datasets/Wikibooks/20250501/dewikibooks-20250501-pages-articles.xml.bz2"

.venv:
	python3 -m venv .venv

clean:
	rm -rf __pycache__ .venv
