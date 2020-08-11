# Keras Neural Finder

Python module that will help in finding the best Neural network.

## What does it do ?
The module helps in automating the processing of finding the best design of neural network by taking in all possible options. Runs the model for each possible combination and returns the result. The results can be used to find the best model. 

## Usage

Create a dictionary as below with the choice of options for each parameter

```
        
        param_grid = {}
        param_grid['hidden_layer_neurons'] = [[11, 8, 6], [6, 6], [6, 5, 3], [6]]
        param_grid['output_layer_neurons'] = [1]
        param_grid['hidden_layer_activation'] = ['sigmoid', 'tanh', 'relu']
        param_grid['output_layer_activation'] = ['sigmoid', 'tanh', 'relu']
        param_grid['kernel_inializers'] = ['glorot_normal', 'random_normal', 'glorot_uniform', 'zeros', 'ones']
        param_grid['bias_initializers'] = ['glorot_normal', 'random_normal', 'glorot_uniform', 'zeros', 'ones']
        param_grid['optimizers'] = ['Adadelta', 'Adagrad', 'Adam', 'Adamax', 'Ftrl', 'Nadam', 'RMSprop', 'SGD']
        param_grid['epochs'] = [10]
        param_grid['learning_rate'] = [0.001, 0.01, 0.1]
        param_grid['loss'] = ['categorical_crossentropy', 'sparse_categorical_crossentropy']

```
Note:
```
        hidden_layer_neurons: There are 2 possible options for 
        hidden layer configuration. 

        Option 1:
        --------
        Ex- [[11, 8, 5], [8, 6]] 

        1st = 3 layers with 11,8 and 5 neurons in each layer
        2nd = 2 layers with 8 and 6 neurons in each layer

        Option 2:
        --------

        {1:(8, 13), 2:(5, 13), 3:(5, 8), 4:(5,6)}

        Key = refers to the layer number
        Value = range of number of neurons that can be configured in that layer
        
        Ex: 
        1 = refers to the 1st layer
        (8, 13) = means that the 1st layer can have from 8 to 13 neurons
        2 = refers to the 2nd layer
        (5, 13) = means that the 2nd layer can have from 5 to 13 neurons
```

## Execution

```
knf = KerasNeuralFinder()
results = knf.fit(param_grid=param_grid, store_results=True, silent_mode=True, X=X_train, y=y_train)

```

The returned results dataframe will have the list of all executions, from which you can choose the best option

```
KerasNeuralFinder.get_best_result(results).T
```

