from datasets.mnist import MNist
from models.network import CNet

mnist = MNist()
train_set, test_set = mnist.load()
train = (train_set.x, train_set.y)
test = (test_set.x, test_set.y)

TRAIN = 0
EVALUATE = 1

cnet = CNet(train=True, image_shape=mnist.image_size)
model = cnet.model

if TRAIN:

    model.fit(*train, validation_data=test, nb_epoch=1, batch_size=512)
    cnet.save()

elif EVALUATE:

    cnet.load()
    _, accuracy = model.evaluate(*test, batch_size=512)
    print('== %s ==\nTest accuracy: %.2f%%' % (cnet.NAME, accuracy * 100))

else:

    cnet.load()
    print(model.predict(test[0]))
