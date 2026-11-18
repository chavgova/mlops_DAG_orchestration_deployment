.PHONY: test
test:
	coverage run -m pytest -v -p no:warnings . && coverage report --rcfile=.coveragerc

.PHONY: format
format:
	ruff format . --line-length 120

.PHONY: build
build:
	docker build . -t nova -f infra/build/Dockerfile

