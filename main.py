from datasets.mnist import MNist
from models.network import CNet

mnist = MNist()
train_set, test_set = mnist.load()
train = (train_set.x, train_set.y)
test = (test_set.x, test_set.y)

TRAIN = 0
EVALUATE = 1

if TRAIN:
    model = CNet(train=True, image_shape=mnist.image_size, epoch=10, batch_size=512)
    model.run(train, test, save=True)
elif EVALUATE:
    model = CNet(image_shape=mnist.image_size, train=True)
    model.load_weights('data/weights/CNet_weights.h5')
    print(model.evaluate(*train))
    print(model.evaluate(*test))
else:
    model = CNet(image_shape=mnist.image_size)
    model.load_weights('data/weights/CNet_weights.h5')
    print(model.predict(test[0]))
