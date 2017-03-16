# Run flake8 and pytest

PEP8ARGS=--show-source --ignore=W503,E226,E241,E731

help:
	@echo "Options:"
	@echo ""
	@echo "    style         check for style consistency"
	@echo "    test          run the test suite"
	@echo ""

style:
	flake8

test:
	python -m pytest --doctest-modules --cov=code --cov-report term-missing
