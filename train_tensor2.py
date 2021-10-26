import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152

import argparse
import random
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras import Model
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.losses import BinaryCrossentropy, MSE
from tensorflow.keras.metrics import BinaryAccuracy, SensitivityAtSpecificity, SpecificityAtSensitivity
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from gen_tensor2 import get_data, generator_training_data, data_augmentation, generator_test_data, ModelCheckpoint
from model_tensor2 import build_model

def parse_args():
    parser = argparse.ArgumentParser(description='Training Falling Detection')
    parser.add_argument('--clip_len', type=int, default=16, help='clip length')
    parser.add_argument('--crop_size', type=int, default=224, help='crop size')
    parser.add_argument('--alpha', type=float, default=0.1, help='alpha')
    parser.add_argument('--gpu', type=str, default='1', help='GPU id')
    parser.add_argument('--use_mse', type=int, default=1, help='use mse')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--drop_rate', type=float, default=0.5, help='drop rate')
    parser.add_argument('--reg_factor', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--epochs', type=int, default=300, help='number of total epochs to run')
    parser.add_argument('--batch_size', type=int, default=64, help='mini-batch size')

    args = parser.parse_args()
    return args

class Self_KD(Model):
    def __init__(self, model, use_mse):
        super(Self_KD, self).__init__()
        self.model = model
        self.use_mse = use_mse

    def compile(self, optimizer, ce_loss, alpha=0.1):
        super(Self_KD, self).compile()
        self.optimizer = optimizer
        self.ce_loss = ce_loss
        self.alpha = alpha

        self.accuracy1 = BinaryAccuracy()
        self.accuracy2 = BinaryAccuracy()
        self.val_accuracy = BinaryAccuracy()
        self.Sensitivity = SensitivityAtSpecificity(0.5)
        self.Specificity = SpecificityAtSensitivity(0.5)

    @tf.function
    def train_step(self, data):
        # Unpack data
        data_1, data_2, y = data

        # ---------------------------------- Training--------------------------------
        with tf.GradientTape() as tape:
            # Forward pass
            [predict_1, z1] = self.model(data_1, training=True)
            [predict_2, z2] = self.model(data_2, training=True)

            # Compute ce loss
            ce_loss_1 = self.ce_loss(y, predict_1)
            ce_loss_2 = self.ce_loss(y, predict_2)

            # Compute KLD loss
            mse1 = MSE(tf.stop_gradient(z2),z1)
            mse2 = MSE(tf.stop_gradient(z1),z2)

            if self.use_mse:
                loss = ce_loss_1 + ce_loss_2 + self.alpha * mse1 + self.alpha * mse2
            else:
                loss = ce_loss_1 + ce_loss_2
            loss += sum(self.model.losses)

        # Compute gradients
        trainable_vars = (self.model.trainable_variables)
        gradients = tape.gradient(loss, trainable_vars)
        # Update weights for model
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Update the metrics configured in `compile()
        self.accuracy1.update_state(y, predict_1)
        self.accuracy2.update_state(y, predict_2)


        # Update the metrics configured in `compile()`.
        # self.compiled_metrics.update_state(y, predict_1)

        # Return a dict of performance
        results = {}
        # results = {m.name: m.result() for m in self.metrics}
        results.update({"loss": loss, "accuracy_1": self.accuracy1.result(), "accuracy_2": self.accuracy2.result()})
        return results

    @tf.function
    def test_step(self, data):
        # Unpack the data
        x, y = data

        # Compute predictions
        [predict, _] = self.model(x, training=False)

        # Calculate the loss for student 1
        celoss = self.ce_loss(y, predict)
        loss = celoss + sum(self.model.losses)

        # Update the metrics.
        self.val_accuracy.update_state(y, predict)
        self.Sensitivity.update_state(y, predict)
        self.Specificity.update_state(y, predict)

        # Update the metrics.
        # self.compiled_metrics.update_state(y, predict)

        # Return a dict of performance
        results = {}
        # results = {m.name: m.result() for m in self.metrics}
        results.update({"loss": loss, "accuracy": self.val_accuracy.result(),
                        'sensitivity': self.Sensitivity.result(), 'specificity': self.Specificity.result()})
        return results

def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))

def build_callbacks(save_path):
    jsonPath = os.path.join(save_path, "output")
    jsonName = 'log_results.json'
    earlyStopping = EarlyStopping(monitor='val_loss', patience=30, verbose=1)
    Checkpoint = ModelCheckpoint(folderpath=save_path, jsonPath=jsonPath, jsonName=jsonName, monitor='val_accuracy', mode='max', verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, verbose=1, mode='min', min_lr=1e-8)
    return [earlyStopping, reduce_lr, Checkpoint]

def training(train_dataset, test_dataset, input_shape, lr_init=0.001,  use_mse_loss=True, batch_size=32, reg_factor=5e-4, drop_rate=0.5, alpha=0.1,
             epochs=200, save_path='save_model'):
    # Prepare data for training phase
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    ds_train = tf.data.Dataset.from_generator(generator_training_data,
                                                (tf.float32, tf.float32, tf.float32),
                                                (tf.TensorShape(input_shape), tf.TensorShape(input_shape), tf.TensorShape([])),
                                                args=[train_dataset, input_shape[0],input_shape[1]])
    ds_train = ds_train.map(data_augmentation, num_parallel_calls=AUTOTUNE).batch(batch_size).prefetch(AUTOTUNE)

    ds_test = tf.data.Dataset.from_generator(generator_test_data, (tf.float32, tf.float32),
                                             (tf.TensorShape(input_shape), tf.TensorShape([])),
                                             args=[test_dataset, input_shape[0], input_shape[1]])
    ds_test = ds_test.batch(batch_size).prefetch(AUTOTUNE)

    # Build model
    base_model = build_model(input_shape=input_shape, num_classes=1, kernel_initializer="he_normal", reg_factor=reg_factor, drop_rate=drop_rate)
    # base_model.summary(line_length=150)

    # Build callbacks
    callback = build_callbacks(save_path)

    # Build optimizer and loss function
    optimizer = SGD(learning_rate=lr_init, momentum=0.9, nesterov=True)
    ce_loss = BinaryCrossentropy()

    # Build model
    Model_KD = Self_KD(base_model, use_mse=use_mse_loss)
    Model_KD.compile(
        optimizer,
        ce_loss,
        alpha=alpha,
    )

    # Training
    Model_KD.fit(ds_train, epochs=epochs, verbose=1,
                      steps_per_epoch=(len(train_dataset) // batch_size),
                      # steps_per_epoch=10,
                      validation_data=ds_test,
                      validation_steps=len(test_dataset) // (batch_size),
                      callbacks=callback
                      )
    

def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)  # Choose GPU for training

    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.compat.v1.InteractiveSession(config=config)

    input_shape = (args.clip_len, args.crop_size, args.crop_size, 3)
    reg_factor = args.reg_factor
    batch_size = args.batch_size
    epochs = args.epochs
    lr_init = args.lr
    alpha = args.alpha
    drop_rate = args.drop_rate
    temp = args.use_mse
    if temp>0:
        use_mse_loss=True
    else:
        use_mse_loss=False

    # Read dataset
    if use_mse_loss:
        save_path = './save_model/'
    else:
        save_path = './save_normal_model/'
    train_dataset = get_data('train.csv')
    test_dataset = get_data('test.csv')
    random.shuffle(test_dataset)
    random.shuffle(train_dataset)
    print('Train set:', len(train_dataset))
    print('Test set:', len(test_dataset))
    
    # Create folders for callback
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    if not os.path.exists(os.path.join(save_path, "output")):
        os.mkdir(os.path.join(save_path, "output"))
    if not os.path.exists(os.path.join(save_path, "checkpoints")):
        os.mkdir(os.path.join(save_path, "checkpoints"))

    # Write all config to file
    f = open(os.path.join(save_path, 'config.txt'), "w")
    f.write('input shape: ' + str(input_shape) + '\n')
    f.write('reg factor: ' + str(reg_factor) + '\n')
    f.write('batch size: ' + str(batch_size) + '\n')
    f.write('numbers of epochs: ' + str(epochs) + '\n')
    f.write('lr init: ' + str(lr_init) + '\n')
    f.write('alpha: ' + str(alpha) + '\n')
    f.write('Drop rate: ' + str(drop_rate) + '\n')
    f.close()

    # --------------------------------------Training ----------------------------------------
    training(train_dataset, test_dataset, input_shape, lr_init, use_mse_loss=use_mse_loss,
               batch_size=batch_size, reg_factor=reg_factor, drop_rate=drop_rate, alpha=alpha, 
               epochs=epochs, save_path=save_path)

if __name__ == '__main__':
    print(tf.__version__)
    main()

