# Keras Neural Finder

As many would have experienced, finding the best neural network is always a challenge, as the optimal number of hidden layers and the number of neurons in each layers, is always difficult to find it. This module will help us in training, testing and validating the model with various combinations of neural layers and neurons and suggest us with the best configuration. 

## What does it do ?
The implementation takes input as a dictionary with list of possible values for different neural configurations. Then creates, trains, fits, validates and evaluates the model's performance for all the possible combination of param's provided. And output's the following. 

* best_params_ -> Best combination of parameter based on low loss and high accuracy
* best_result_ -> Results of the best neural network. Results include the mean, minimum and maximum values of different metrics based on the confidence interval provided. 
* results_ -> Run results of all the models trained with all possible combination of parameters. 

## Usage

Create a dictionary as below with the choice of options for each parameter

```
param_grid = {}
param_grid['hidden_layer_neurons'] ={1:(6,12), 2:(6, 6), 3:(6,8)}
param_grid['output_layer_neurons'] = [1]
param_grid['hidden_layer_activations'] = ['sigmoid', 'tanh', 'relu']
param_grid['output_layer_activations'] = ['sigmoid', 'tanh', 'relu']
param_grid['kernel_inializers'] = ['glorot_normal']
param_grid['bias_initializers'] = ['glorot_normal']
param_grid['optimizers'] = ['Adadelta', 'Adagrad', 'Adam', 'Adamax', 'Ftrl', 'Nadam', 'RMSprop', 'SGD']
param_grid['epochs'] = [20]
param_grid['learning_rate'] = [0.01, 0.1]
param_grid['loss_functions'] = ['binary_crossentropy']

```
Note:

#### hidden_layer_neurons 
There are 2 possible options for providing values. 

###### Option 1:
```
Ex- [[11, 8, 5], [8, 6]] 
```

1st = 3 layers with 11,8 and 5 neurons in each layer
2nd = 2 layers with 8 and 6 neurons in each layer

###### Option 2:
```
{1:(8, 13), 2:(5, 13), 3:(5, 8), 4:(5,6)}
```

Key = refers to the layer number
Value = range of number of neurons that can be configured in that layer
        
Ex: 
1 = refers to the 1st layer, (8, 13) = means that the 1st layer can have from 8 to 13 neurons
2 = refers to the 2nd layer, (5, 13) = means that the 2nd layer can have from 5 to 13 neurons


## Execution
Create an instance of the KerasNeuralFinder class as below
```
knf = KerasNeuralFinder()
```

You can find the number of different choices based on the param_grid configured as below 

```
knf.estimate_run(param_grid)
Output: 'Total number of choices 108'
```

You can run the implementation as below:

```
results = knf.fit(param_grid=param_grid,
                  X=X_train, y=y_train, fold=5,
                  confidence_interval=95,
                  metrics=['binary_accuracy', 'accuracy'])

```

## Outputs of the run

```
knf.best_params_ - Returns the best params that produced results with minimum loss and maximum accuracy
knf.best_result_ - Returns the dataframe with the results (includes mean, min and max of all metrics as well)
knf.results_ - Returns details of all the runs for further manual validation and analysis.
```

Can use below method as well to get the best result 
```
knf.get_best_result(by={'accuracy': 'high'}, count=1)
by -> Filters based on which the results has to be sorted out. Can either be str of dict. If str, the entries with maximum value for that parameter will be returned. 
If dict, key will be the parameter name and the value should either be 'high' or 'low'. 
```
