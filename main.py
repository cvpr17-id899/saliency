from __future__ import division
from keras.optimizers import RMSprop
from keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler
from keras.layers import Input
from keras.models import Model
import os, cv2, sys
import numpy as np
from config import *
from utilities import preprocess_images, preprocess_maps, postprocess_predictions
from model import sam, kl_divergence, schedule


def generator(b_s, phase_gen='train'):
    if phase_gen == 'train':
        images = [imgs_train_path + f for f in os.listdir(imgs_train_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
        maps = [maps_train_path + f for f in os.listdir(maps_train_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
    elif phase_gen == 'val':
        images = [imgs_val_path + f for f in os.listdir(imgs_val_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
        maps = [maps_val_path + f for f in os.listdir(maps_val_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
    else:
        raise NotImplementedError

    images.sort()
    maps.sort()

    gaussian = np.zeros((b_s, nb_gaussian, shape_r_gt, shape_c_gt))

    counter = 0
    while True:
        yield [preprocess_images(images[counter:counter+b_s], shape_r, shape_c), gaussian], preprocess_maps(maps[counter:counter+b_s], shape_r_gt, shape_c_gt)
        counter = (counter + b_s) % len(images)


def generator_test(b_s, imgs_test_path):
    images = [imgs_test_path + f for f in os.listdir(imgs_test_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
    images.sort()

    gaussian = np.zeros((b_s, nb_gaussian, shape_r_gt, shape_c_gt))

    counter = 0
    while True:
        yield [preprocess_images(images[counter:counter + b_s], shape_r, shape_c), gaussian]
        counter = (counter + b_s) % len(images)


if __name__ == '__main__':
    phase = sys.argv[1]

    x = Input((3, shape_r, shape_c))
    x_maps = Input((nb_gaussian, shape_r_gt, shape_c_gt))
    m = Model(input=[x, x_maps], output=sam([x, x_maps]))

    print("Compiling SAM")
    m.compile(RMSprop(lr=1e-4, rho=0.9, epsilon=1e-08, decay=0.0), kl_divergence)

    if phase == 'train':
        print("Training SAM")
        m.fit_generator(generator(b_s=b_s), nb_imgs_train, nb_epoch=nb_epoch,
                        validation_data=generator(b_s=b_s, phase_gen='val'), nb_val_samples=nb_imgs_val,
                        callbacks=[EarlyStopping(patience=5),
                                   ModelCheckpoint('weights.sam.{epoch:02d}-{val_loss:.4f}.pkl', save_best_only=True),
                                   LearningRateScheduler(schedule=schedule)])

    elif phase == "test":
        # path of output folder
        output_folder = ''

        if len(sys.argv) < 2:
            raise SyntaxError
        imgs_test_path = sys.argv[2]

        file_names = [f for f in os.listdir(imgs_test_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
        file_names.sort()
        nb_imgs_test = len(file_names)

        print("Loading SAM weights")
        m.load_weights('sam_salicon_weights.pkl')

        print("Predicting saliency maps for " + imgs_test_path)
        predictions = m.predict_generator(generator_test(b_s=b_s, imgs_test_path=imgs_test_path), nb_imgs_test)

        for pred, name in zip(predictions, file_names):
            original_image = cv2.imread(imgs_test_path + name, 0)
            res = postprocess_predictions(pred[0], original_image.shape[0], original_image.shape[1])
            name = name.split('.')[0]
            cv2.imwrite(output_folder + '%s_sam.jpg' % name, res.astype(int))
    else:
        raise NotImplementedError