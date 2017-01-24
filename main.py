from datasets.mnist import MNist
from models.network import CNet

mnist = MNist()
train_set, test_set = mnist.load()
train = (train_set.x, train_set.y)
test = (test_set.x, test_set.y)

TRAIN = 0
EVALUATE = 0

if TRAIN:

    model = CNet(train=True, image_shape=mnist.image_size)
    model.fit(*train, validation_data=test, nb_epoch=1, batch_size=512)
    _, accuracy = model.evaluate(*test, batch_size=512)
    model.save()
    print('== %s ==\nTest accuracy: %.2f%%' % (model.NAME, accuracy * 100))

elif EVALUATE:

    model = CNet(image_shape=mnist.image_size, train=True)
    model.load_weights('data/weights/CNet_weights.h5')
    print(model.evaluate(*test))

else:

    model = CNet(image_shape=mnist.image_size)
    model.load_weights('data/weights/CNet_weights.h5')
    print(model.predict(test[0]))
