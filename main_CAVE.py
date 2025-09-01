import MyLib as ML
import CAVE_dataReader as Crd
from SMF2Net import mymodel
import argparse
from tensorflow.keras.optimizers import Adam
from tensorflow import keras
import tensorflow as tf
import random
import numpy as np
import os
import scipy.io as sio


parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--mode', type=str, default='train', help='train/test')
parser.add_argument('--sf', type=int, default=32)
# output channel number
parser.add_argument('--outDim', type=int, default=31)
parser.add_argument('--epoch', type=int, default=200)
# the size of training samples
parser.add_argument('--image_size', type=int, default=96)
parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--num_gpus', type=int, default=1)

args = parser.parse_args()
print(args)

def set_random_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
set_random_seed(seed=55)

def train():
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    model = mymodel()

    # CAVE DATA READER
    allX, allY = Crd.all_train_data_in()
    batch_X, batch_Y, batch_Z = Crd.train_data_in(allX, allY, args.image_size, patchNum = 2240)

    checkpoint_filepath = r"temp_train\CAVE\best_model.h5"
    checkpoint_callback = keras.callbacks.ModelCheckpoint(
        checkpoint_filepath,
        monitor='val_loss',
        verbose=1,
        save_best_only=True,
        save_weights_only=False
    )

    # lr set
    step = tf.Variable(0, trainable=False)
    batch_per_epoch = len(batch_X) // args.batch_size
    boundaries = [50 * batch_per_epoch]
    values = [0.001, 0.0001]
    learning_rate_fn = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
    boundaries,
    values
    )

    class UpdateStepCallback(tf.keras.callbacks.Callback):
        def on_batch_end(self, epoch, logs=None):
            step.assign_add(1)

    class PrintLrCallback(tf.keras.callbacks.Callback):
        def on_epoch_begin(self, epoch, logs=None):
            current_lr = learning_rate_fn(step).numpy()
            print(f"\nEpoch {epoch + 1} lr: {current_lr:.6f}")

    model.compile(optimizer=Adam(learning_rate=learning_rate_fn(step)),
                    loss="mean_squared_error")
    model.run_eagerly = True

    input = [batch_Y, batch_Z]
    history = model.fit(input, batch_X, epochs = args.epoch, batch_size = args.batch_size,
                validation_split=0.2, callbacks=[checkpoint_callback,UpdateStepCallback(), PrintLrCallback()])


    print('Complete training')
    return  history


def test():
    model = mymodel()
    model.run_eagerly = True

    false_Y = tf.random.normal((1,512, 512, 3))
    false_Z = tf.random.normal((1,16, 16, 31))
    model([false_Y, false_Z])

    checkpoint_filepath = r".\temp_train\CAVE\best_model.h5"
    model.load_weights(checkpoint_filepath)
    print('model load succeed')

    for root, dirs, files in os.walk(r'.\CAVEdata\X'):
        List = sio.loadmat(r'.\CAVEdata\myList')
        Ind = List['list'][0][20:]
        for i in Ind:
            print(files[i - 1])
            data = sio.loadmat(r".\CAVEdata\X/" + files[i - 1])
            gtX = data['msi']
            gtX = np.expand_dims(gtX, axis=0)

            data = sio.loadmat(r".\CAVEdata\Y/" + files[i - 1])
            inY = data['RGB']
            inY = np.expand_dims(inY, axis=0)
            inY = tf.convert_to_tensor(inY)

            data = sio.loadmat(r".\CAVEdata\Z/" + files[i - 1])
            inZ = data['Zmsi']
            inZ = np.expand_dims(inZ, axis=0)
            inZ = tf.convert_to_tensor(inZ)

            input_data = [inY, inZ]
            pred_X = model.predict(input_data, batch_size=1)

            sio.savemat(r'TestResult\CAVE/' + files[i - 1] + '.mat', {'outX': pred_X})

            showX = ML.get3band_of_tensor(pred_X, nbanch=0, nframe=[9,19,29])
            maxS = np.max(showX)
            minS = np.min(showX)
            toshow = ML.setRange(ML.get3band_of_tensor(pred_X, nbanch=0, nframe=[9,19,29]), maxS, minS)
            ML.imwrite(toshow, ('TestResult\CAVE\%s.png' % (files[i - 1])))
            
            pred_X = np.squeeze(np.array(pred_X))
            gtX = np.squeeze(np.array(gtX))
            pred_X = ML.normalized(pred_X)
            gtX = ML.normalized(gtX)
            
            psnr_val = tf.image.psnr(pred_X, gtX, max_val=255.0)
            print(files[i - 1] + ' done!')
            print("PSNR:", psnr_val)


if __name__ == '__main__':
    if args.num_gpus == 0:
        dev = '/cpu:0'
    elif args.num_gpus == 1:
        dev = '/gpu:0'
    else:
        raise ValueError('Only support 0 or 1 gpu.')

    with tf.device(dev):
        if args.mode == 'test':  # test
            test()
        elif args.mode == 'train':  # train
            print('Start training')
            train()




