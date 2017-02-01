all: clean keras tsb

keras:
	-python main.py keras train
tf:
	-python main.py tf train
tfr:
	-python main.py tfr train

tsb:
	tensorboard --logdir=./logs/train/

clean:
	rm -rf logs/
