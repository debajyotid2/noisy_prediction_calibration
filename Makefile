all: run

run: main.py
	python $^

clean:
	$(RM) -r outputs __pycache__ logs
