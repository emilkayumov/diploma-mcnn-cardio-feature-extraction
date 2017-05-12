import numpy as np

from scipy.signal import resample
from scipy.fftpack import fft

from sklearn.metrics import accuracy_score, roc_auc_score

from keras.layers import Convolution1D, Dense, Dropout, Input, concatenate, GlobalMaxPooling1D
from keras.models import Model
from keras.optimizers import Adam


# for working with all signal
window = 1000
step   = 500

# different dataset size
prep_size = 150
fft_size  = 500
wv_size   = 1010

# slice of signal
slice_len = 1000

EPS = 1e-10



def preprocess_signal(signal):
    """ Normalize only slice """
    signal = np.array(signal, dtype=np.float32)
    signal -= np.mean(signal)
    signal = signal / (np.max(signal) + EPS)
    return signal


def preprocess_signal_all(data):
    data = np.array(data, dtype=np.float32)
    data -= np.mean(data, axis=1).reshape((-1, 1))
    data = data / (np.max(data, axis=1).reshape((-1, 1)) + EPS)
    return data


def generate_slice(slice_len, data, labels, coef=0.8):
    i = np.random.randint(0, len(data))
    signal_i = np.random.randint(0, len(data[i]))
    X = data[i][signal_i].reshape((-1, 1))
    rand_slice_len = np.random.randint(int(slice_len*coef), int(slice_len/coef)+1)
    slice_start = np.random.randint(0, len(X)-rand_slice_len)
    slice_x = resample(X[slice_start:slice_start+rand_slice_len], slice_len)
    slice_x = preprocess_signal(slice_x)
    return slice_x, labels[i]


def generator(batch_size, slice_len, data, labels):
    while True:
        batch_x = []
        batch_y = []
        for i in range(0, batch_size):
            x, y = generate_slice(slice_len, data, labels)
            batch_x.append(x)
            batch_y.append(y)
        y = np.array(batch_y)
        x_250 = np.array([resample(i, 250) for i in batch_x])
        x_500 = np.array([resample(i, 500) for i in batch_x])
        x = np.array([i for i in batch_x])
        yield ([x_250, x_500, x], y)


def get_base_model(input_len, fsize, nb_filters):
    input_seq = Input(shape=(input_len, 1))
    convolved = Convolution1D(nb_filters, fsize, padding='same', activation='tanh')(input_seq)
    processed = GlobalMaxPooling1D()(convolved)
    compressed = Dense(150, activation='tanh')(processed)
    compressed = Dropout(0.1)(compressed)
    compressed = Dense(150, activation='tanh')(compressed)
    compressed = Dropout(0.1)(compressed)
    model = Model(inputs=input_seq, outputs=compressed)            
    return model


def make_network():
    input250_seq = Input(shape=(250, 1))
    input500_seq = Input(shape=(500, 1))
    input1000_seq = Input(shape=(1000, 1))

    base_network250  = get_base_model(250, 50, 50) 
    base_network500  = get_base_model(500, 50, 50) 
    base_network1000 = get_base_model(1000, 50, 50) 
    
    embedding_250  = base_network250(input250_seq)
    embedding_500  = base_network500(input500_seq)
    embedding_1000 = base_network1000(input1000_seq)

    merged = concatenate([embedding_250, embedding_500, embedding_1000])
    merged = Dense(150, activation="tanh")(merged)
    merged = Dropout(0.2)(merged)
    out = Dense(1, activation='sigmoid')(merged)
    
    model = Model(inputs=[input250_seq, input500_seq, input1000_seq], outputs=out)
    opt = Adam(lr=0.0001)
    model.compile(loss="binary_crossentropy", optimizer=opt, metrics=['accuracy'])
    return model


def base_model_preprocess(input_len, nb_filters, fsize, weights):
    input_seq = Input(shape=(input_len, 1))
    convolved = Convolution1D(nb_filters, fsize, padding='same', activation='tanh', 
                              weights=[weights[0], weights[1]])(input_seq)
    processed = GlobalMaxPooling1D()(convolved)
    compressed = Dense(150, activation='tanh', weights=[weights[2], weights[3]])(processed)
    compressed = Dropout(0.1)(compressed)
    compressed = Dense(150, activation='tanh', weights=[weights[4], weights[5]])(compressed)
    compressed = Dropout(0.1)(compressed)
    model = Model(inputs=[input_seq], outputs=[compressed]) 
    return model


def make_preprocess(weights):
    input250_seq = Input(shape=(250, 1))
    input500_seq = Input(shape=(500, 1))
    input1000_seq = Input(shape=(1000, 1))

    model_250  = base_model_preprocess(250, 50, 50, weights[0:6])
    model_500  = base_model_preprocess(500, 50, 50, weights[6:12])
    model_1000 = base_model_preprocess(1000, 50, 50, weights[12:18])

    merged = concatenate([model_250(input250_seq), model_500(input500_seq), model_1000(input1000_seq)])
    merged = Dense(150, activation='tanh', weights=[weights[18], weights[19]])(merged)
    preprocess = Model(inputs=[input250_seq, input500_seq, input1000_seq], outputs=merged)
    return preprocess


def preprocess_data(size, data, labels):
    prep = np.empty((size, prep_size))
    prep_y = np.empty(size)
    for j in xrange(size):
        slice_x, y = generate_slice(slice_len, data, labels, coef=1.0)
        x_250 = resample(slice_x, 250).reshape((1, -1, 1))
        x_500 = resample(slice_x, 500).reshape((1, -1, 1))
        x_1000 = slice_x.reshape((1, -1, 1))
        prep[j] = preprocess.predict([x_250, x_500, x_1000])
        prep_y[j] = y
    return prep, prep_y


def make_dataset(size, data, labels):
    prep = np.empty((size, slice_len))
    prep_y = np.empty(size)
    for j in xrange(size):
        slice_x, y = generate_slice(slice_len, data, labels, coef=1.0)
        prep[j] = slice_x.reshape(-1)
        prep_y[j] = y
    return prep, prep_y


def resample_slice(slice_x):
    x_250 = resample(slice_x, 250).reshape((1, -1, 1))
    x_500 = resample(slice_x, 500).reshape((1, -1, 1))
    x_1000 = slice_x.reshape((1, -1, 1))
    return [x_250, x_500, x_1000]


def preprocess_dataset(data):
    prep = np.empty((data.shape[0], prep_size))
    for j in xrange(len(data)):
        prep[j] = preprocess.predict(resample_slice(data[j]))
    return prep


def fft_dataset(data):
    return np.abs(fft(data, axis=1))[:, :fft_size]


def find_max_acc_thr(y_true, y_pred):
    space = np.linspace(np.min(y_pred) - 1e-10, np.max(y_pred) + 1e-10, 100)
    return space[np.argmax([accuracy_score(y_true, y_pred > thr) for thr in space])]


def calc_max_acc(y_true, y_pred):
    n = y_true.shape[0]
    
    thr_1 = find_max_acc_thr(y_true[:n/2], y_pred[:n/2])
    acc_1 = accuracy_score(y_true[n/2:], y_pred[n/2:] > thr_1)
    
    thr_2 = find_max_acc_thr(y_true[n/2:], y_pred[n/2:])
    acc_2 = accuracy_score(y_true[:n/2], y_pred[:n/2] > thr_1)
    
    return acc_1, thr_1, acc_2, thr_2


def calc_result(y_true, y_pred):
    result = {}
    result['auc'] = roc_auc_score(y_true, y_pred)
    result['acc'] = accuracy_score(y_true, y_pred > 0.5)
    acc_1, thr_1, acc_2, thr_2 = calc_max_acc(y_true, y_pred)
    result['max_acc'] = {'acc': [acc_1, acc_2], 'thr': [thr_1, thr_2]}
    result['y_pred'] = y_pred
    result['y_true'] = y_true
    return result


def slice_wt_prep(slice_x, w):
    level = 4 # level of wavelet coef
    result = []
    a = slice_x
    for i in xrange(level):
        (a, d) = pywt.dwt(a, w)
        result += list(d)
    return np.array(list(a) + result)


def preprocess_wv(data):
    w = pywt.Wavelet('db2')
    new_data = np.empty((data.shape[0], wv_size))
    for i in xrange(len(data)):
        new_data[i] = slice_wt_prep(data[i], w)
    return new_data


def exctract_result(result, key):
    acc_it, auc_it, max_acc_it = [], [], []
    for it in result.keys():
        acc_it.append(np.mean([result[it][fold][key]['acc'] for fold in result[it].keys()]))
        auc_it.append(np.mean([result[it][fold][key]['auc'] for fold in result[it].keys()]))
        max_acc_it.append(np.mean([np.mean(result[it][fold][key]['max_acc']['acc']) for fold in result[it].keys()]))
    return {
        'acc': np.mean(acc_it),
        'auc': np.mean(auc_it),
        'max_acc': np.mean(max_acc_it)
    }
