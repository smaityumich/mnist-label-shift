from data_load import mixture
import tensorflow as tf
import numpy as np
from sklearn import metrics
import itertools
import sys

def label_shift(n_target = 1000, n_source = 20000, pi_target = 0.1,\
     pi_source = 0.9, epochs = 5, reg = 0, batch_size = 400):

    x_source, y_source = mixture(n = n_source, pi = pi_source, flatten=False)
    x_target, y_target = mixture(n = n_target, pi = pi_target, flatten=False)
    xt, yt = mixture(n = 20000, pi = pi_target, flatten=False)
    x_val, y_val = mixture(n = 1000, pi = pi_target, flatten=False)

    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    ### Pilot
    model_pilot = tf.keras.models.Sequential([
            tf.keras.layers.Conv1D(3, 2, input_shape=(28, 28)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(20),
            tf.keras.layers.Dense(2)
        ])
    
    model_pilot.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy'])
    model_pilot.fit(x_source, y_source, epochs=epochs, batch_size=batch_size)
    accuracy_train_pilot = model_pilot.evaluate(x_source, y_source, verbose=2, batch_size=n_source)[1]
    accuracy_pilot = model_pilot.evaluate(x_target, y_target, verbose=2, batch_size=n_target)[1]


    
    ### Reweighting
    confusion_matrix = metrics.confusion_matrix(model_pilot.predict_classes(x_source),y_source, labels=[0,1])/n_source
    prop_target = np.mean(model_pilot.predict_classes(x_target))
    
    xi = np.array([1-prop_target,prop_target])
    w = np.matmul(np.linalg.inv(confusion_matrix + reg*np.diag([1, 1])),xi)
    prop_targets = w*np.array([1-pi_source, pi_source])
    if prop_targets[0] <0:
        prop_targets[0] = 0
    if prop_targets[1] <0:
        prop_targets[1] = 0
    prop_targets = prop_targets/np.sum(prop_targets)

    sample_weight_0 = prop_targets[0]/(1-pi_source)
    sample_weight_1 = prop_targets[1]/(pi_source)
    sample_weight = sample_weight_0 * (1-y_source) + sample_weight_1 * y_source

    model_reweighted = tf.keras.models.Sequential([
            tf.keras.layers.Conv1D(3, 2, input_shape=(28, 28)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(20),
            tf.keras.layers.Dense(2)
        ])
    model_reweighted.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy'])
    model_reweighted.fit(x_source, y_source, epochs=epochs, sample_weight = sample_weight, batch_size=batch_size)
    accuracy_train_reweighted = model_reweighted.evaluate(x_source, y_source, verbose=2, batch_size=n_source)[1]
    accuracy_reweighted = model_reweighted.evaluate(x_target, y_target, verbose=2, batch_size=n_target)[1]


    bias_correction = np.log(prop_targets[1]/prop_targets[0]) - np.log(pi_source/(1-pi_source))
    w = model_pilot.get_weights()
    w[-1][0], w[-1][1] = w[-1][0] - bias_correction, w[-1][1] + bias_correction
    model_pilot.set_weights(w)
    accuracy_train_ls = model_pilot.evaluate(x_source, y_source, verbose=2, batch_size=n_source)[1]
    accuracy_label_shift = model_pilot.evaluate(x_target, y_target, verbose=2, batch_size=n_target)[1]

    bias_oracle = np.log(pi_target/pi_target) - np.log(pi_source/(1-pi_source))
    w = model_pilot.get_weights()
    w[-1][0], w[-1][1] = w[-1][0] + bias_correction - bias_oracle, w[-1][1] - bias_correction + bias_oracle
    model_pilot.set_weights(w)
    accuracy_oracle = model_pilot.evaluate(x_target, y_target, verbose=2, batch_size=n_target)[1]


    model_gs = tf.keras.models.Sequential([
            tf.keras.layers.Conv1D(3, 2, input_shape=(28, 28)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(20),
            tf.keras.layers.Dense(2)
            ])
    model_gs.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy'])
    model_gs.fit(xt, yt, epochs=epochs, batch_size=batch_size)
    accuracy_gs = model_gs.evaluate(x_val, y_val, verbose=2, batch_size=1000)[1]

    return accuracy_pilot, accuracy_reweighted, accuracy_label_shift, accuracy_train_pilot,\
        accuracy_train_reweighted, accuracy_train_ls, accuracy_gs, accuracy_oracle


n_targets = [500, 1000, 5000, 10000, 20000]
pi_targets = [0.1, 0.2, 0.3, 0.4, 0.5]
iters = range(20)
l = list(itertools.product(n_targets, pi_targets, iters))
i = int(float(sys.argv[1]))
n_target, pi_target, iteration = l[i]
pi_source = 1-pi_target
accuracy_pilot, accuracy_reweighted, accuracy_label_shift, accuracy_train_pilot,\
        accuracy_train_reweighted, accuracy_train_ls, accuracy_gs, accuracy_oracle\
             = label_shift(n_target=n_target,\
         pi_source=pi_source, pi_target=pi_target, epochs= 25, batch_size=None)
filename = f'nt_{n_target}_pt_{pi_target}_i_{iteration}'
dict_output = dict()
dict_output['n_target'] = n_target
dict_output['pi_target'] = pi_target
dict_output['iter'] = iteration
dict_output['acc-pilot-test'] = accuracy_pilot
dict_output['acc-pilot-train'] = accuracy_train_pilot
dict_output['acc-rw-test'] = accuracy_reweighted
dict_output['acc-rw-train'] = accuracy_train_reweighted
dict_output['acc-ls-test'] = accuracy_label_shift
dict_output['acc-ls-train'] = accuracy_train_ls
dict_output['acc-oracle'] = accuracy_oracle
dict_output['gold-standard'] = accuracy_gs
with open('temp/'+filename, 'w+') as f:
    f.writelines(str(dict_output) + '\n')
print(str(dict_output) + '\n')
