import os
import joblib
import numpy as np
from df_generator import DataframeGenerator

w1, b1, w2, b2 = 0, 0, 0, 0

# Activation functions
def relu(z):
    """
    Stands for Rectified Linear Unit, operates through an array and determines its maximum value as long as it's greater than zero.

    Args:
        z (ndarray): the array to be evaluated

    Returns:
        float: maximum value within the array or 0 if the value is less than 0
    """
    return np.maximum(z, 0)


def relu_derivative(z):
    """
    Returns a boolean array with values bigger than 0 as 1s and the rest as 0s.
    
    Args:
        z (ndarray): the array to be evaluated
    """
    return z > 0


def softmax(z):
    """Compute softmax values for each set of scores in an array.

    Args:
        z (ndarray): array to be evaluated
    """
    exp_z = np.exp(z - np.max(z))
    return exp_z / exp_z.sum(axis=0)


# Parametrization
def init_params(size, first_layer_neurons, second_layer_neurons):
    """Use a normally random value to initialize weights and biases.

    Args:
        size (int): number of columns in the current dataset
        first_layer_neurons (int): number of neurons for the first layer
        second_layer_neurons (int): number of neurons for the second layer

    Returns:
        w1, b1, w2, b2 (tuple): the randomly initialized weights and biases
    """
    # Substract 0.5 since we want random values between -0.5 and 0.5 (not 0 and 1)
    w1 = np.random.rand(first_layer_neurons, size) - 0.5
    b1 = np.random.rand(first_layer_neurons, 1) - 0.5
    w2 = np.random.rand(second_layer_neurons, first_layer_neurons) - 0.5
    b2 = np.random.rand(second_layer_neurons, 1) - 0.5
    return w1, b1, w2, b2


def update_params(w1, b1, w2, b2, dw1, db1, dw2, db2, rate, first_layer_neurons, second_layer_neurons):
    """Update the model's parameters and reshape them if necessary.

    Args:
        w1 (ndarray): the array of weights for the first neural layer
        b1 (ndarray): the array of biases for the first neural layer
        w2 (ndarray): the array of weights for the second neural layer
        b2 (ndarray): the array of biases for the second neural layer
        dw1 (ndarray): the array of weight errors for the first neural layer
        db1 (ndarray): the array of bias errors for the first neural layer
        dw2 (ndarray): the array of weight errors for the second neural layer
        db2 (ndarray): the array of bias errors for the second neural layer
        rate (float): the learning rate of the model
        first_layer_neurons (int): the number of neurons for the first layer
        second_layer_neurons (int): the number of neurons for the second layer

    Returns:
        w1, b1, w2, b2 (tuple): the newly adjusted parameters
    """
    w1 -= rate * dw1
    b1 -= rate * np.reshape(db1, (first_layer_neurons, 1))
    w2 -= rate * dw2
    b2 -= rate * np.reshape(db2, (second_layer_neurons, 1))
    return w1, b1, w2, b2


# Helper functions
def one_hot_encode(y_data):
    """Apply one-hot encoding by returning a zeros array with ones only at the index where the maximum value of each y is found.

    Args:
        y_data (ndarray): The y values to encode
    """
    encoded_data = np.zeros((y_data.max() + 1, y_data.size))
    encoded_data[y_data, np.arange(y_data.size)] = 1
    return encoded_data


# Forward/backward propagation
def forward_propagation(x_data, w1, b1, w2, b2):
    """Run the neurons to generate a prediction for each layer.

    Args:
        x_data (ndarray): the x values of the dataset
        w1 (ndarray): the array of weights for the first layer
        b1 (ndarray): the array of biases for the first layer
        w2 (ndarray): the array of weights for the second layer
        b2 (ndarray): the array of biases for the second layer

    Returns:
        z1, a1, z2, a2 (tuple): the result for each neural layer
    """
    # First layer values
    z1 = w1.dot(x_data) + b1
    a1 = relu(z1)
    # Second layer values
    z2 = w2.dot(a1) + b2
    a2 = softmax(z2)
    return z1, a1, z2, a2


def backward_propagation(x_data, y_data, z1, a1, a2, w2, samples):
    """Determine the errors of the output of each neural layer.

    Args:
        x_data (ndarray): the x data of the dataset
        y_data (ndarray): the y data of the dataset
        z1 (ndarray): the linear result of the first layer
        a1 (ndarray): the activation results of the first layer
        a2 (ndarray): the activation results of the second layer
        w2 (ndarray): the weight array for the second layer
        samples (int): number of samples within the dataset

    Returns:
        dw1, db1, dw2, db2 (tuple): the calculated errors for each layer's weights and biases
    """
    # Encode y data
    encoded_data = one_hot_encode(y_data)
    # Second layer errors
    dz2 = (a2 - encoded_data) * 2
    dw2 = 1 / samples * dz2.dot(a1.T)
    db2 = 1 / samples * np.sum(dz2, 1)
    # First layer errors
    dz1 = w2.T.dot(dz2) * relu_derivative(z1)
    dw1 = 1 / samples * dz1.dot(x_data.T)
    db1 = 1 / samples * np.sum(dz1, 1)
    return dw1, db1, dw2, db2


# Output functions
def get_predictions(a2):
    """Return the max for each row if its value is higher than 0, else return 0.

    Args:
        a2 (ndarray): the activation results of the second layer

    Returns:
        ndarray: the indices for the output's maximum values
    """
    return np.argmax(a2, 0)


def get_accuracy(predicts, y_data):
    """Estimate the accuracy of the model with the current weights and biases.

    Args:
        predicts (ndarray): array of indices where the highest values are located
        y_data (ndarray): the y values of the current dataset

    Returns:
        ndarray: the average of the model's error
    """
    return np.sum(predicts == y_data) / y_data.size


def gradient_descent(x_data, y_data, rate, epochs, first_layer_neurons, second_layer_neurons):
    """Perform gradient descent on each epoch to determine the best result by trying to diminish the error on each epoch.1

    Args:
        x_data (ndarray): the x data of the dataset
        y_data (ndarray): the y data of the dataset
        rate (float): the learning rate of the model
        epochs (int): the number of iterations to be performed
        first_layer_neurons (int): the number of neurons for the first neural layer
        second_layer_neurons (int): the number of neurons for the second neural layer

    Returns:
        w1, b1, w2, b2: the most recently adjusted weights and biases for the model
    """
    # Get current data size
    size, samples = x_data.shape
    # Initalize parameters
    w1, b1, w2, b2 = init_params(size, first_layer_neurons, second_layer_neurons)
    # Update parameters
    for epoch in range(epochs):
        z1, a1, z2, a2 = forward_propagation(x_data, w1, b1, w2, b2)
        dw1, db1, dw2, db2 = backward_propagation(x_data, y_data, z1, a1, a2, w2, samples)
        w1, b1, w2, b2 = update_params(w1, b1, w2, b2, dw1, db1, dw2, db2, rate, first_layer_neurons, second_layer_neurons)
        
        if (epoch + 1) % 1000 == 0 or epoch + 1 == epochs:
            predicts = get_predictions(a2)
            print(f"--- Epoch # {epoch} ---")
            print(f"Accuracy: {get_accuracy(predicts, y_data):.3%}")

    return w1, b1, w2, b2


def make_predictions(x_data, w1, b1, w2, b2):
    """Get the output of the second neural layer with the current weights and biases.

    Args:
        x_data (ndarray): the x values of the dataset
        w1 (ndarray): the weights for the first neural layer
        b1 (ndarray): the biases for the first neural layer
        w2 (ndarray): the weights for the second neural layer
        b2 (ndarray): the biases for the second neural layer

    Returns:
        ndarray: array of indices with the labels that the neural network chose
    """
    a2 = forward_propagation(x_data, w1, b1, w2, b2)[3]
    predicts = get_predictions(a2)
    return predicts


def show_predictions(x_data, y_data, w1, b1, w2, b2, index):
    """Displays the results of the calculations made by the model

    Args:
        x_data (ndarray): the x values of the data set
        y_data (ndarray): the x values of the data set
        w1 (ndarray): the weights for the first neural layer
        b1 (ndarray): the biases for the first neural layer
        w2 (ndarray): the weights for the second neural layer
        b2 (ndarray): the biases for the second neural layer
        index (int): the index of the x values to be evaluated
    """
    x_sample = x_data[:, index, None]
    predicts = make_predictions(x_sample, w1, b1, w2, b2)
    
    label = y_data[index]
    print(f"Prediction: {predicts}")
    print(f"Label: {label}")


def process_data(data):
    """Fix data in order for it to be properly normalized and structured.

    Args:
        data (DataFrame): the original dataset

    Returns:
        data (DataFrame): the fixed dataset
    """
    # Clean dataframe from unwanted columns and assign them new column names
    data = data.drop(0, axis=1)
    # Move diagnosis column to be the first column
    diagnosis_col = data.pop(10)
    data.insert(0, 'diagnosis', diagnosis_col)
    # Apply function to diagnosis column for it to represent boolean values instead of 2s and 4s
    data['diagnosis'] = data['diagnosis'].apply(lambda x: 1 if x == 4 else 0)
    # Normalize columns
    for colname in data.drop('diagnosis', axis=1).columns:
        col_max, col_min = data[colname].max(), data[colname].min()
        data[colname] = (data[colname] - col_min)  / (col_max - col_min)
    return data


def fit():
    # Import dataframe
    df_gen = DataframeGenerator("../pre-data/Breast-Cancer/breast-cancer-wisconsin.data")
    df = process_data(df_gen.train)
    # Split dataframe into separate arrays
    x = df.drop('diagnosis', axis=1).to_numpy().T
    y = df['diagnosis'].to_numpy()
    # Run the model
    first_layer_neurons, second_layer_neurons = 8, 2
    w1, b1, w2, b2 = gradient_descent(x, y, 0.001, 10000, first_layer_neurons, second_layer_neurons)
    return w1, b1, w2, b2


def predict_from_sample(sample, w1, b1, w2, b2):
    a2 = forward_propagation(sample, w1, b1, w2, b2)
    result = get_predictions(a2)
    return result


if __name__ == '__main__':
    w1, b1, w2, b2 = init_params(9, 8, 2)
    print(w1.shape, b1.shape)
    print(w2.shape, b2.shape)