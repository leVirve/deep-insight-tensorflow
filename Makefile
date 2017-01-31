all: clean keras tsb

keras:
	python keras_main.py train
tf:
	-python tf_main.py train
tfr:
	-python tfr_main.py train

tsb:
	tensorboard --logdir=./logs/train/

clean:
	rm -rf logs/
