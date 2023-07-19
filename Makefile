all: run_mnist

run_mnist: main.py
	TF_CPP_MIN_LOG_LEVEL=3 python $^ dataset=mnist

run_cifar: main.py
	TF_CPP_MIN_LOG_LEVEL=3 python $^ dataset=cifar10

run_fmnist: main.py
	TF_CPP_MIN_LOG_LEVEL=3 python $^ dataset=fashion_mnist

clean_cache:
	$(RM) -r cache

clean:
	$(RM) -r outputs __pycache__ logs
