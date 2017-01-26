all: clean train tsb

train:
	CUDA_VISIBLE_DEVICES=1 python main.py train

tsb:
	tensorboard --logdir=logs/

clean:
	rm -rf logs/
