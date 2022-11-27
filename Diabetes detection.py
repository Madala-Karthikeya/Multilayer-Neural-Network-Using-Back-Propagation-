from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
np.random.seed(1)
def sigmoid(x):
    return 1/(1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

diabetes_dataset = pd.read_csv("/Users/neelima/Documents/SEM-5/ANN/CASE STUDY/db_data.csv")
diabetes_dataset['Polyuria'] = diabetes_dataset['Polyuria'].map(
    {'Yes': 1, 'No': 0})
diabetes_dataset['Polydipsia'] = diabetes_dataset['Polydipsia'].map(
    {'Yes': 1, 'No': 0})
diabetes_dataset['sudden weight loss'] = diabetes_dataset['sudden weight loss'].map(
    {'Yes': 1, 'No': 0})
diabetes_dataset['weakness'] = diabetes_dataset['weakness'].map(
    {'Yes': 1, 'No': 0})
diabetes_dataset['Polyphagia'] = diabetes_dataset['Polyphagia'].map(
    {'Yes': 1, 'No': 0})
diabetes_dataset['Genital thrush'] = diabetes_dataset['Genital thrush'].map(
    {'Yes': 1, 'No': 0})
diabetes_dataset['visual blurring'] = diabetes_dataset['visual blurring'].map(
    {'Yes': 1, 'No': 0})
diabetes_dataset['Itching'] = diabetes_dataset['Itching'].map(
    {'Yes': 1, 'No': 0})
diabetes_dataset['Irritability'] = diabetes_dataset['Irritability'].map(
    {'Yes': 1, 'No': 0})
diabetes_dataset['delayed healing'] = diabetes_dataset['delayed healing'].map(
    {'Yes': 1, 'No': 0})
diabetes_dataset['partial paresis'] = diabetes_dataset['partial paresis'].map(
    {'Yes': 1, 'No': 0})
diabetes_dataset['muscle stiffness'] = diabetes_dataset['muscle stiffness'].map(
    {'Yes': 1, 'No': 0})
diabetes_dataset['Alopecia'] = diabetes_dataset['Alopecia'].map(
    {'Yes': 1, 'No': 0})
diabetes_dataset['Obesity'] = diabetes_dataset['Obesity'].map(
    {'Yes': 1, 'No': 0})
diabetes_dataset['Gender'] = diabetes_dataset['Gender'].map(
    {'Male': 1, 'Female': 0})
diabetes_dataset['class'] = diabetes_dataset['class'].map(
    {'Positive': 1, 'Negative': 0})

# print(diabetes_dataset)

X = diabetes_dataset.drop(columns='class', axis=1)
# print(X.shape)
Y = diabetes_dataset['class']
inputs = X.to_numpy()
expected_output = Y.to_numpy()
expected_output = np.reshape(expected_output, (520, 1))

# # inputs = np.array([[0,1,1,0,1,0,1,0,1,1,1,2,1,0,1,0],
# #                    [1,0,1,0,1,0,1,0,0,1,1,1,1,1,1,0]])
# # expected_output = np.array([[0],[1]])

epochs = 30000
lr = 0.00005
inputLayerNeurons, hiddenLayerNeurons, outputLayerNeurons = 16, 8, 1

hidden_weights = np.random.uniform(
    size=(inputLayerNeurons, hiddenLayerNeurons))
hidden_bias = np.random.uniform(size=(1, hiddenLayerNeurons))
output_weights = np.random.uniform(
    size=(hiddenLayerNeurons, outputLayerNeurons))
output_bias = np.random.uniform(size=(1, outputLayerNeurons))


for _ in range(epochs):
    # Forward Propagation
    hidden_layer_activation = np.dot(inputs, hidden_weights)
    hidden_layer_activation += hidden_bias
    hidden_layer_output = sigmoid(hidden_layer_activation)

    output_layer_activation = np.dot(hidden_layer_output, output_weights)
    output_layer_activation += output_bias
    predicted_output = sigmoid(output_layer_activation)

    error = expected_output - predicted_output
    d_predicted_output = error * sigmoid_derivative(predicted_output)

    error_hidden_layer = d_predicted_output.dot(output_weights.T)
    d_hidden_layer = error_hidden_layer * \
        sigmoid_derivative(hidden_layer_output)

    # Updating Weights and Biases
    hidden_weights += inputs.T.dot(d_hidden_layer) * lr
    hidden_bias += np.sum(d_hidden_layer, axis=0, keepdims=True) * lr
    output_weights += hidden_layer_output.T.dot(d_predicted_output) * lr
    output_bias += np.sum(d_predicted_output, axis=0, keepdims=True) * lr
    

# print("Final hidden weights: ", end='')
# print(*hidden_weights)
# print("Final hidden bias: ", end='')
# print(*hidden_bias)
# print("Final output weights: ", end='')
# print(*output_weights)
# print("Final output bias: ", end='')
# print(*output_bias)

# print("\nOutput from neural network after epochs: ", end='')
print(*predicted_output)
# print(expected_output.shape)
# test_data_accuracy = accuracy_score(predicted_output, expected_output)
# print(test_data_accuracy)
np.save('hidden_weights.npy', hidden_weights)
np.save('hidden_bias.npy', hidden_bias)
np.save('output_weights.npy', output_weights)
np.save('output_bias.npy', output_bias)

print(predicted_output.shape)
print(expected_output.shape)

predicted_output[predicted_output > 0.5] = 1
predicted_output[predicted_output < 0.5] = 0
print(predicted_output)
accuracy = accuracy_score(predicted_output, expected_output)
print(accuracy)
