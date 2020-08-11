import pandas as pd
from tensorflow import keras
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import math
import IPython
import datetime
import time
import traceback


class KerasNeuralFinder:
    """
    Keras Neural Finder (KNF can be used to find the best neural network by providing variety of possible parameters.
    """

    __version__ = "1.1"

    # Default value for the inputs
    _default_hidden_layer_neurons = [[11, 6]]
    _default_output_layer_neurons = [1]
    _default_hidden_layer_activation = ['relu']
    _default_output_layer_activation = ['relu']
    _default_optimizers = ['Adam']
    _default_epochs = ['10']
    _default_learning_rate = [0.01]
    _default_loss = ['binary_crossentropy']
    _default_kernel_initializers = ['glorot_normal']
    _default_bias_initializers = ['random_normal']

    # Input name to be used by the users, while providig param_grid
    _input_hidden_layer_neurons = 'hidden_layer_neurons'
    _input_output_layer_neurons = 'output_layer_neurons'
    _input_hidden_layer_activations = 'hidden_layer_activations'
    _input_output_layer_activations = 'output_layer_activations'
    _input_optimizers = 'optimizers'
    _input_epochs = 'epochs'
    _input_learning_rates = 'learning_rates'
    _input_loss_functions = 'loss_functions'
    _input_kernel_inializers = 'kernel_inializers'
    _input_bias_initializers = 'bias_initializers'

    def _create_model(self):
        """
        Creates an empty sequential keras model
        """
        model = Sequential()
        return model

    def _add_layers(self, model, hidden_layer_neurons=None, output_layer_neurons=None,
                    activation_fn_hidden_layer=None, activation_fn_output_layer=None,
                    kernel_inializers=None, bias_initializers=None, X=None):
        """
        Adds the layers to the provided model # of layers is based on the number of different # of neurons
        provided in number_of_neurons_in_hidden_layer.
        Sets the Activation function for Hidden & Output layer as provided
        :param model: Model for which the layers are to be added
        :param hidden_layer_neurons: array of values denoting the # of neurons in each layer
        :param output_layer_neurons: number of neurons to be specified in output layer
        :param activation_fn_hidden_layer: activation function to be used in hidden layer
        :param activation_fn_output_layer: activation function to be used in output layer
        :param kernel_inializers: Kernel initializer to be used for generating weights
        :param bias_initializers: Bias initializer
        :param X: dataset with independent variables, used to find the input_shape for weights
        :return:
        """
        number_of_hidden_layers = len(hidden_layer_neurons)
        i = 0
        for layer in range(0, number_of_hidden_layers):
            model.add(Dense(hidden_layer_neurons[i], input_shape=(X.shape[1],),
                            activation=activation_fn_hidden_layer, kernel_initializer=kernel_inializers,
                            bias_initializer=bias_initializers))
            i += 1
        model.add(Dense(output_layer_neurons, input_shape=(X.shape[1],), activation=activation_fn_output_layer))

        return model

    def _get_layer_options(self, min_neurons_in_hidden_layers):
        """
        Generates all possible combinations of layers based on the ranges provided
        :param min_neurons_in_hidden_layers: dict defining the range of neuron count in each layer.
        Ex: {1:(8, 13), 2:(5, 13), 3:(5, 8), 4:(5,6)}
        Key-refers to the layer count
        Value-refers to the range of neurons to be configured in that layer
        :return: List of lists indicating the number of neurons to be configured in each layer
        Ex:[[8], [5], [3], [8,5], [8,3], [5,3]..]
        """
        layer_options = []

        neurons_ranges = {}

        for layer_number in min_neurons_in_hidden_layers.keys():
            min_max = min_neurons_in_hidden_layers[layer_number]
            min_neurons = min_max[0]
            max_neurons = min_max[1]
            neurons_ranges[layer_number] = list(range(min_neurons, max_neurons + 1))

        prev_layer_neurons = []

        for layer_count in neurons_ranges.keys():
            new_prev_layer_list = []

            if layer_count == 1:
                for neuron_count in neurons_ranges[layer_count]:
                    temp = []
                    temp.append(neuron_count)
                    layer_options.append(temp)
                    prev_layer_neurons.append(neuron_count)
            else:

                for prev_layer_neuron in prev_layer_neurons:
                    for curr_layer_neuron in neurons_ranges[layer_count]:
                        temp = []

                        if isinstance(prev_layer_neuron, list):
                            for prev_layer_neuron_counts in prev_layer_neuron:
                                temp.append(prev_layer_neuron_counts)
                        else:
                            temp.append(prev_layer_neuron)

                        temp.append(curr_layer_neuron)
                        layer_options.append(temp)
                        new_prev_layer_list.append(temp)
            if layer_count > 1:
                prev_layer_neurons = new_prev_layer_list

        return layer_options

    def _get_optimizer(self, optimizer_choice, learning_rate):
        """
        Returns an optimizer instance based on the choice of optimizer provided
        :param optimizer_choice: Optimizer method
        :param learning_rate: Learning rate at which the weights has to be adjusted for variables
        :return: Instance based on the optimizer method provided
        """
        optimizer_instance = keras.optimizers.get(optimizer_choice)
        return optimizer_instance

    def _compile_model(self, model, optimizer_instance, loss):
        """
        Compiles the model with default loss of binary_crossentropy
        Other parameters are Optimizer instance and Learning rate
        :param model: Model configured with layers
        :param optimizer_instance: Optimizer method to be used
        :return: compiled model ready to be fit with dataset
        """
        model.compile(loss=loss, optimizer=optimizer_instance, metrics=['accuracy'])
        return model

    def _fit_model(self, model, epoch, batch_size, local_X=None, local_y=None):
        """
        Fits the model with the provided X and y values
        :param model: Compiled Model that has to be fit with the dataset
        :param epoch: Number of epochs to be used while doing gradient descent
        :param batch_size: Batch size to be considered for every iteration
        :param local_X: Independent variables
        :param local_y: Outcome varaibles
        :return: Model fit with the dataset ready to be evaluated
        """
        model.fit(local_X, local_y, epochs=epoch, batch_size=500, verbose=0)
        return model

    def _evaluate_model(self, model, local_X, local_y):
        """
        Evalutes the model for Loss and accuracy metrics
        :param model: model that has to be evaluated
        :param local_X: independent variables
        :param local_y: outcome variable
        :return: dict {'loss': loss, 'accuracy': acc}
        """
        loss, acc = model.evaluate(local_X, local_y, verbose=0)
        return {'loss': loss, 'accuracy': acc}

    def _convert_list_to_string(self, input_list):
        """
        Util function to convert the List value stored in DataFrame to a String
        :param input_list: list that has to be concatenated as a string delimited with comma
        :return: String
        """

        if isinstance(input_list, int):
            return str(input_list)

        final_value = ''
        for value in input_list:
            if final_value:
                final_value = final_value + "," + str(value)
            else:
                final_value = str(value)

        return final_value

    def _convert_string_to_list(self, input_string):
        """
        Util function to convert the String in to a List to be stored in DataFrame
        :param input_string: String that has to be converted in a list with delimited as comma
        :return: List
        """
        final_list = []
        split_value = input_string.split(",")
        for value in split_value:
            value = int(value)
            final_list.append(value)

        return final_list

    def _output_df_to_csv(self, df=None):
        """
        Outputs the dataframe in to a csv file
        :param df: DataFrame to be stored in a csv file
        :return: None
        """
        file_name_prefix = 'model_hp_results_'
        df_name = df.rename
        if df_name:
            file_name_prefix = df_name
        curr_date_time = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
        filename = file_name_prefix + curr_date_time + ".csv"
        df.to_csv(filename)
        print(f"Hyper parameter testing results saved to file - {filename}")

    @staticmethod
    def estimate_run(param_grid):
        """
        Estimates the number of choices possible based on the number of parameters defined
        :param param_grid: dictionary of params with wich the model is to be run
        :return: Total number of choices
        """
        total_diff_choices = 1
        for value in param_grid.values():
            total_diff_choices *= len(value)

        approx_time_taken = total_diff_choices * 1.5

        print(f"There are around {total_diff_choices} choices based on the list of parameters defined")
        print(f"And it might take approximately {approx_time_taken} seconds assuming it would take 1.5 seconds for "
              f"every epoch")
        return total_diff_choices

    def _validate_params(self, param_grid):
        """
        Validates and transforms the params, as required
        :param param_grid: dictionary of list of values for each parameter
        :return: validated and formatted param_grid
        """
        for keys, values in param_grid.items():
            if not isinstance(values, list):
                if keys == self._input_hidden_layer_neurons:
                    param_grid[keys] = self._get_layer_options(values)
                else:
                    param_grid[keys] = [values]
        return param_grid

    @staticmethod
    def get_param_info():

        print("""
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

        Note:
        hidden_layer_neurons: Ex- [[11, 8, 5], [8, 6]] means there are 2 possible options for 
        hidden layer configuration. 
        1st = 3 layers with 11,8 and 5 neurons in each layer
        2nd = 2 layers with 8 and 6 neurons in each layer

        Alternate way to configure hidden_layer_neurons
        get_layer_options(range)
        range can be defined in a dict as {1:(8, 13), 2:(5, 13), 3:(5, 8), 4:(5,6)}
        Key = refers to the layer number
        Value = range of number of neurons that can be configured in that layer
        Ex: 
        1 = refers to the 1st layer
        (8, 13) = means that the 1st layer can have from 8 to 13 neurons
        2 = refers to the 2nd layer
        (5, 13) = means that the 2nd layer can have from 5 to 13 neurons
        """)

    def fit(self, param_grid, silent_mode=False, store_results=False, X=None, y=None):
        """
        * Creates model for every possible combination of hyper parameters defined
        * Fits the model with the training dataset
        * Evaluates the model with Training and Validation dataset
        * Stores the evaluation results for every iteration

        :param param_grid: Dictionary of parameters on which the model has to be created and run
        :param silent_mode: to avoid asked for a user confirmation before proceeding with the run
        :param store_results: to confirm if the results are to be stored in a csv file
        :param X: independent variables
        :param y: outcome variables
        :return: dataframe with the choices and as well corresponding loss and accuracy

        Sample param_grid dict:

        param_grid = {}
        param_grid['hidden_layer_neurons'] = [[11, 8, 6], [6, 6], [6, 5, 3], [6]]
        param_grid['output_layer_neurons'] = 1
        param_grid['hidden_layer_activation'] = ['sigmoid', 'tanh', 'relu']
        param_grid['output_layer_activation'] = ['sigmoid', 'tanh', 'relu']
        param_grid['kernel_inializers'] = ['glorot_normal', 'random_normal', 'glorot_uniform', 'zeros', 'ones']
        param_grid['bias_initializers'] = ['glorot_normal', 'random_normal', 'glorot_uniform', 'zeros', 'ones']
        param_grid['optimizers'] = ['Adadelta', 'Adagrad', 'Adam', 'Adamax', 'Ftrl', 'Nadam', 'RMSprop', 'SGD']
        param_grid['epochs'] = 20
        param_grid['learning_rate'] = [0.001, 0.01, 0.1]
        param_grid['loss'] = ['categorical_crossentropy', 'sparse_categorical_crossentropy']

        """
        overall_time_start = 0
        model_results = None

        try:

            should_run = False

            param_grid = self._validate_params(param_grid)

            # get the provided list of values for all the parameters

            hidden_layer_neurons = param_grid.get(self._input_hidden_layer_neurons, self._default_hidden_layer_neurons)
            output_layer_neurons = param_grid.get(self._input_output_layer_neurons, self._default_output_layer_neurons)
            hidden_layer_activations = param_grid.get(self._input_hidden_layer_activations,
                                                      self._default_hidden_layer_activation)
            output_layer_activations = param_grid.get(self._input_output_layer_activations,
                                                      self._default_output_layer_activation)
            optimizers = param_grid.get(self._input_optimizers, self._default_optimizers)
            epochs = param_grid.get(self._input_epochs, self._default_epochs)
            learning_rates = param_grid.get(self._input_learning_rates, self._default_learning_rate)
            loss_functions = param_grid.get(self._input_loss_functions, self._default_loss)
            kernel_inializers = param_grid.get(self._input_kernel_inializers, self._default_kernel_initializers)
            bias_initializers = param_grid.get(self._input_bias_initializers, self._default_bias_initializers)

            print(f"validated...")
            print(param_grid)

            total_diff_choices = self.estimate_run(param_grid)

            if not silent_mode:
                choice = input("Are you sure to proceed ? (yes/no)")
                if choice == "yes":
                    should_run = True
                else:
                    should_run = False

            # confirms with user if they really intend to run the iterative model for hyperparameter tuning
            if silent_mode or should_run:
                choice = 1

                model_results = []
                model_results_df = None

                overall_time_start = time.time()

                print(f"Starting ...")

                for choice_hidden_layer_neurons in hidden_layer_neurons:
                    for choice_output_layer_neurons in output_layer_neurons:
                        for choice_hidden_layer_activations in hidden_layer_activations:
                            for choice_output_layer_activations in output_layer_activations:
                                for choice_optimizers in optimizers:
                                    for choice_epochs in epochs:
                                        for choice_learning_rates in learning_rates:
                                            for choice_loss_functions in loss_functions:
                                                for choice_kernel_inializers in kernel_inializers:
                                                    for choice_bias_initializers in bias_initializers:
                                                        # Resetting the variables for every iteration
                                                        model = None
                                                        model_result = {}
                                                        start_time = time.time()

                                                        # Creating Model, adding layers, compiling and fit
                                                        model = self._create_model()
                                                        model = self._add_layers(model,
                                                                                 choice_hidden_layer_neurons,
                                                                                 choice_output_layer_neurons,
                                                                                 choice_hidden_layer_activations,
                                                                                 choice_output_layer_activations,
                                                                                 choice_kernel_inializers,
                                                                                 choice_bias_initializers,
                                                                                 X)

                                                        optimizer_instance = self._get_optimizer(choice_optimizers,
                                                                                                 choice_learning_rates)
                                                        model = self._compile_model(model,
                                                                                    optimizer_instance,
                                                                                    choice_loss_functions)

                                                        # Sets the batch size to 10% of the total size
                                                        batch_size = math.ceil(X.shape[0] * 0.1)
                                                        model = self._fit_model(model, choice_epochs, batch_size, X, y)

                                                        # Evaluating model with Training data
                                                        model_evaluation_train_result = self._evaluate_model(model, X, y)

                                                        # Capturing the values for every iteration to be stored in a dataframe
                                                        model_result['choice'] = choice
                                                        model_result['Hidden Layer(# of neurons)'] = \
                                                            self._convert_list_to_string(choice_hidden_layer_neurons)
                                                        model_result['Hidden Layer(# of layers)'] = \
                                                            len(choice_hidden_layer_activations)
                                                        model_result['Hidden Layer(Activation Fn)'] = \
                                                            choice_hidden_layer_activations
                                                        model_result['Output Layer(# of neurons)'] = \
                                                            self._convert_list_to_string(choice_output_layer_neurons)
                                                        model_result['Output Layer(Activation Fn)'] = \
                                                            choice_output_layer_activations
                                                        model_result[self._input_optimizers] = choice_optimizers
                                                        model_result[self._input_learning_rates] = choice_learning_rates
                                                        model_result[self._input_kernel_inializers] = choice_kernel_inializers
                                                        model_result[self._input_bias_initializers] = choice_bias_initializers
                                                        model_result[self._input_epochs] = choice_epochs
                                                        model_result['loss'] = model_evaluation_train_result['loss']
                                                        model_result['accuracy'] = model_evaluation_train_result['accuracy']

                                                        model_results.append(model_result)

                                                        loss = round(model_evaluation_train_result['loss'], 3)
                                                        acc = round(model_evaluation_train_result['accuracy'], 3)

                                                        end_time = time.time()
                                                        time_taken = round((end_time - start_time), 2)

                                                        # Clearing the output for every iteration
                                                        IPython.display.clear_output(wait=True)

                                                        print(f"Choice {choice}/{total_diff_choices}..., epoch = {choice_epochs},"
                                                              f"loss={loss}, acc={acc}, Time taken={time_taken} secs")
                                                    choice += 1

            # Aborts if the user decides to cancel the run
            else:
                print("Not running the model...")

        except KeyboardInterrupt:
            print(f"User interrupted the process. Model run until choice {choice}")

        except Exception as e:
            print(f"Exception occurred !!!, {str(e)}")
            traceback.print_exc()

        finally:
            overall_time_end = time.time()
            overall_time_taken = round((overall_time_end - overall_time_start), 2)
            print(f"Completed running ... overall time taken = {overall_time_taken}")
            model_results_df = pd.DataFrame(model_results)

            if store_results:
                self._output_df_to_csv(model_results_df)

            return model_results_df


"""
X = np.random.randn(1000).reshape(1000, 1)
y = np.random.randn(1000).reshape(1000, 1)


param_grid = {}
param_grid['hidden_layer_neurons'] = {1:(1,2)}
param_grid['output_layer_neurons'] = 1
param_grid['hidden_layer_activation'] = ['sigmoid']
param_grid['output_layer_activation'] = ['sigmoid']
param_grid['kernel_inializers'] = ['glorot_normal']
param_grid['bias_initializers'] = ['glorot_normal']
param_grid['optimizers'] = [ 'Adam']
param_grid['epochs'] = 1
param_grid['learning_rate'] = [0.1]
param_grid['loss'] = ['binary_crossentropy']

knf = KerasNeuralFinder()
results = knf.fit(param_grid=param_grid, X=X, y=y)
print(results.T)
"""