import numpy as np
from data_handler import DataHandler


def relu(z):
    return np.maximum(z, 0)


def relu_deriv(z):
    return z > 0


def softmax(z):
    exp_z = np.exp(z - np.max(z))
    return exp_z / exp_z.sum(axis=0)


def init_params(columns, layer_sizes):
    initial_params = []
    for i in range(len(layer_sizes)):
        weight = 0
        if i == 0:
            weight = np.random.rand(layer_sizes[i], columns)
        else:
            weight = np.random.rand(layer_sizes[i], layer_sizes[i - 1])
        bias = np.random.rand(layer_sizes[i], 1)
        initial_params.append((weight, bias))
    return initial_params


def update_params(params, slopes, rate, layer_sizes):
    for i in range(len(layer_sizes)):
        new_weight = params[i][0] - rate * slopes[i][0]
        new_bias = params[i][1] - rate * np.reshape(slopes[i][1], (layer_sizes[i], 1))
        params[i] = (new_weight, new_bias)
    return params


def one_hot_encode(y):
    encoded = np.zeros((y.max() + 1, y.size))
    encoded[y, np.arange(y.size)] = 1
    return encoded


def fwd_prop(x, params):
    fixed_params = []
    for i in range(len(params)):
        curr_z = params[i][0].dot(x) + params[i][1]
        curr_a = 0
        if i + 1 == len(params):
            curr_a = relu(curr_z)
        else:
            curr_a = softmax(curr_z)
        fixed_params.append((curr_z, curr_a))
    return fixed_params


def bwd_prop(x, y, fixed_params, params, samples):
    encoded = one_hot_encode(y)
    deriv_params = []
    for i in range(len(fixed_params) - 1, -1, -1):
        curr_dz, curr_dw, curr_db = 0, 0, 0
        if i + 1 == len(fixed_params):
            curr_dz = (fixed_params[i][1] - encoded) * 2
            curr_dw = (1 / samples) * curr_dz.dot(fixed_params[i - 1][1].T)
            curr_db = (1 / samples) * np.sum(curr_dz, 1)
        else:
            curr_dz = params[i + 1][0].T.dot((fixed_params[i + 1][1] - encoded) * 2) * relu_deriv(fixed_params[i][0])
            curr_dw = (1 / samples) * curr_dz.dot(x.T)
            curr_db = (1 / samples) * np.sum(curr_dz, 1)
        deriv_params.append((curr_dw, curr_db))
    return deriv_params


def get_predictions(fixed_params):
    max_args = []
    for i in range(fixed_params - 1, 0, -1):
        max_args.append(np.argmax(fixed_params[i][1]))
    return max_args


def get_accuracy(predictions, y): 
    np.sum(predictions == y) / y.size


def grad_desc(x, y, rate, epochs, layer_sizes): 
    columns, samples = x.shape
    params = init_params(columns, layer_sizes)
    for epoch in range(epochs):
        fixed_params = fwd_prop(x, params)
        deriv_params = bwd_prop(x, y, fixed_params, params, samples)
        params = update_params(params, deriv_params, rate, layer_sizes)
    return params


def process_data(data, label_colname, unwanted_cols=None):
    # Clean dataframe from unwanted columns and assign them new column names
    if unwanted_cols:
        for colname in unwanted_cols:
            data = data.drop(unwanted_cols, axis=1)
    # Move diagnosis column to be the first column
    diagnosis_col = data.pop(10)
    data.insert(0, label_colname, diagnosis_col)
    # Apply function to diagnosis column for it to represent boolean values instead of 2s and 4s
    data[label_colname] = data[label_colname].apply(lambda x: 1 if x == 4 else 0)
    # Normalize columns
    for colname in data.drop(label_colname, axis=1).columns:
        col_max, col_min = data[colname].max(), data[colname].min()
        data[colname] = (data[colname] - col_min)  / (col_max - col_min)
    return data


def fit():
    df_gen = DataHandler("pre-data\\Breast-Cancer\\breast-cancer-wisconsin.data")
    df = process_data(df_gen.train, 'Diagnosis', unwanted_cols=[0])
    x = df.drop('Diagnosis', axis=1).to_numpy().T
    print(x.shape)
    y = df['Diagnosis'].to_numpy()
    rate = 0.001
    epochs = 10000
    layer_sizes = [9, 2]
    return grad_desc(x, y, rate, epochs, layer_sizes)


def predict(sample, params):
    fixed_params = fwd_prop(sample, params)
    return get_predictions(fixed_params)[-1]


if __name__ == '__main__':
    params = init_params(9, [9, 2])
    for p in params:
        print(f"w = {p[0].shape} | b = {p[1].shape}")
