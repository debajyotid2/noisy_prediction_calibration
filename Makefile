all: run_mnist

run_mnist: main.py
	TF_CPP_MIN_LOG_LEVEL=3 python $^ dataset=mnist npc.noise_mode=symmetric npc.noise_rate=0
	TF_CPP_MIN_LOG_LEVEL=3 python $^ dataset=mnist npc.noise_mode=idn npc.noise_rate=40

run_cifar: main.py
	TF_CPP_MIN_LOG_LEVEL=3 python $^ dataset=cifar10 npc.noise_mode=sym npc.noise_rate=0
	TF_CPP_MIN_LOG_LEVEL=3 python $^ dataset=cifar10 npc.noise_mode=sym npc.noise_rate=20
	TF_CPP_MIN_LOG_LEVEL=3 python $^ dataset=cifar10 npc.noise_mode=sym npc.noise_rate=80
	TF_CPP_MIN_LOG_LEVEL=3 python $^ dataset=cifar10 npc.noise_mode=asym npc.noise_rate=20
	TF_CPP_MIN_LOG_LEVEL=3 python $^ dataset=cifar10 npc.noise_mode=asym npc.noise_rate=40
	TF_CPP_MIN_LOG_LEVEL=3 python $^ dataset=cifar10 npc.noise_mode=idn npc.noise_rate=20
	TF_CPP_MIN_LOG_LEVEL=3 python $^ dataset=cifar10 npc.noise_mode=idn npc.noise_rate=40
	TF_CPP_MIN_LOG_LEVEL=3 python $^ dataset=cifar10 npc.noise_mode=sridn npc.noise_rate=20
	TF_CPP_MIN_LOG_LEVEL=3 python $^ dataset=cifar10 npc.noise_mode=sridn npc.noise_rate=40

run_fmnist: main.py
	TF_CPP_MIN_LOG_LEVEL=3 python $^ dataset=fashion_mnist npc.noise_mode=sym npc.noise_rate=0
	TF_CPP_MIN_LOG_LEVEL=3 python $^ dataset=fashion_mnist npc.noise_mode=sym npc.noise_rate=20
	TF_CPP_MIN_LOG_LEVEL=3 python $^ dataset=fashion_mnist npc.noise_mode=sym npc.noise_rate=80
	TF_CPP_MIN_LOG_LEVEL=3 python $^ dataset=fashion_mnist npc.noise_mode=asym npc.noise_rate=20
	TF_CPP_MIN_LOG_LEVEL=3 python $^ dataset=fashion_mnist npc.noise_mode=asym npc.noise_rate=40
	TF_CPP_MIN_LOG_LEVEL=3 python $^ dataset=fashion_mnist npc.noise_mode=idn npc.noise_rate=20
	TF_CPP_MIN_LOG_LEVEL=3 python $^ dataset=fashion_mnist npc.noise_mode=idn npc.noise_rate=40
	TF_CPP_MIN_LOG_LEVEL=3 python $^ dataset=fashion_mnist npc.noise_mode=sridn npc.noise_rate=20
	TF_CPP_MIN_LOG_LEVEL=3 python $^ dataset=fashion_mnist npc.noise_mode=sridn npc.noise_rate=40

clean_cache:
	$(RM) -r cache

clean:
	$(RM) -r outputs __pycache__ logs
