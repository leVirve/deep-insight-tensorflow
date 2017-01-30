all: clean train tsb

train:
	CUDA_VISIBLE_DEVICES=1 python main.py train

tsb:
	tensorboard --logdir=logs/

clean:
	rm -rf logs/


# Fast access temporarily

tf:
	python tf_main.py train
tfr:
	CUDA_VISIBLE_DEVICES=1 python tfr_main.py train
