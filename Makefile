all: run

run: main.py
	TF_CPP_MIN_LOG_LEVEL=3 python $^

clean:
	$(RM) -r outputs __pycache__ logs
