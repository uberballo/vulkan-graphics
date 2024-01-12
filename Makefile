.PHONY: run
run:
	cargo run

.PHONY: compile
compile:
	compile.bat

.PHONY: comp-run
comp-run: compile run