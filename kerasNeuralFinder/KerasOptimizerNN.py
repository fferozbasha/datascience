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
from sklearn.model_selection import StratifiedKFold, cross_val_score
import itertools


class KerasNeuralFinder:
    """
    Keras Neural Finder (KNF can be used to find the best neural network by providing variety of possible parameters.
    """

    __version__ = "1.1"

    _confidence_interval_std = {80: 1.282, 85: 1.440, 90: 1.645, 95: 1.960, 99: 2.576, 99.5: 2.807, 99.9: 3.291}

    _model_weights = []

    # Default value for the inputs
    _default_hidden_layer_neurons = [11, 6]
    _default_output_layer_neurons = 1
    _default_hidden_layer_activation = 'relu'
    _default_output_layer_activation = 'relu'
    _default_optimizers = 'Adam'
    _default_epochs = '10'
    _default_learning_rate = 0.01
    _default_loss = 'binary_crossentropy'
    _default_kernel_initializers = 'glorot_normal'
    _default_bias_initializers = 'random_normal'

    # Input name to be used by the users, while providig param_grid
    _choice = 'choice'
    _input_hidden_layer_neurons = 'hidden_layer_neurons'
    _input_output_layer_neurons = 'output_layer_neurons'
    _input_hidden_layer_activations = 'hidden_layer_activations'
    _input_output_layer_activations = 'output_layer_activations'
    _input_optimizers = 'optimizers'
    _input_epochs = 'epochs'
    _input_learning_rates = 'learning_rate'
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
            model.add(Dense(hidden_layer_neurons[i], input_shape=(X.shape[1], ),
                            activation=activation_fn_hidden_layer, kernel_initializer=kernel_inializers,
                            bias_initializer=bias_initializers, name='hidden_layer_' + str(i)))
            i += 1
        model.add(Dense(output_layer_neurons, input_shape=(X.shape[1],), activation=activation_fn_output_layer, name='output_layer'))

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

    def _stratified_crossvalidation(self, model, epoch, fold, batch_size, x, y, metrics):

        #training_accuarcy_score = []
        #validation_accuarcy_score = []
        #training_loss_score = []
        #validation_loss_score = []
        results = []
        skf = StratifiedKFold(n_splits=fold, shuffle=False)

        for train_indices, val_indices in skf.split(x, y):
            iter_result = {}
            xtrain, xval = x[train_indices], x[val_indices]
            ytrain, yval = y[train_indices], y[val_indices]

            history = self._fit_model(model, epoch, batch_size, xtrain, ytrain, xval, yval)

            for metric, values in history.history.items():
                iter_result[metric] = values[-1]

            #training_accuarcy_score.append(accuracy_history[-1])
            #validation_accuarcy_score.append(val_accuracy_history[-1])
            #training_loss_score.append(loss_history[-1])
            #validation_loss_score.append(val_loss_history[-1])
            results.append(iter_result)

        return results


    def _compile_model(self, model, optimizer_instance, loss, metrics):
        """
        Compiles the model with default loss of binary_crossentropy
        Other parameters are Optimizer instance and Learning rate
        :param model: Model configured with layers
        :param optimizer_instance: Optimizer method to be used
        :return: compiled model ready to be fit with dataset
        """
        model.compile(loss=loss, optimizer=optimizer_instance, metrics=metrics)
        return model

    def _retain_weights(self, model, model_info):
        """
        To retain the weights for re-run use to ensure reproducible results
        :param model: model which will be iterated for layers and corresponding weights will be retained
        :return: returns the dictionary of weights for each layer (layer name as key)
        """
        for layer in model.layers:
            weights = layer.get_weights()
            model_info[layer.name] = weights
        self._model_weights.append(model_info)

    def _set_weights(self, model, prev_weights, model_info, choice):

        if prev_weights.empty:
            print("Invalid file provided for the historical weights.")
            return

        layer_info = {self._choice: model_info[self._choice],
                      self._input_hidden_layer_neurons: model_info[self._input_hidden_layer_neurons],
                      self._input_hidden_layer_activations: model_info[self._input_hidden_layer_activations],
                      self._input_output_layer_activations: model_info[self._input_output_layer_activations],
                      self._input_optimizers: model_info[self._input_optimizers],
                      self._input_learning_rates: model_info[self._input_learning_rates],
                      self._input_kernel_inializers: model_info[self._input_kernel_inializers],
                      self._input_bias_initializers: model_info[self._input_bias_initializers],
                      self._input_epochs: model_info[self._input_epochs],
                      self._input_loss_functions: model_info[self._input_loss_functions]}

        filtered = pd.DataFrame()
        idx = 0
        if not prev_weights.empty:
            for keys in layer_info.keys():
                value = layer_info[keys]
                if keys in prev_weights.columns:
                    if idx == 0:
                        filtered = prev_weights[prev_weights[keys] == value]
                    elif filtered.empty:
                        print(f"No previous weights found for {choice}")
                        return model
                    else:
                        filtered = filtered[filtered[keys] == value]
                idx += 1

        if not filtered.empty:
            for layer in model.layers:
                earlier_layer_weights = filtered.get(layer.name, None)
                if not earlier_layer_weights.empty:
                    wts = np.asarray(earlier_layer_weights[choice-1])
                    _all_weights_list = []
                    for wt in wts:
                        _individual_wts_array = np.array(wt)
                        _all_weights_list.append(_individual_wts_array)
                    layer.set_weights(_all_weights_list)

            return model

        return model

    def _fit_model(self, model, epoch, batch_size, x, y, xval, yval):
        """
        Fits the model with the provided X and y values
        :param model: Compiled Model that has to be fit with the dataset
        :param epoch: Number of epochs to be used while doing gradient descent
        :param batch_size: Batch size to be considered for every iteration
        :param local_X: Independent variables
        :param local_y: Outcome varaibles
        :return: Model fit with the dataset ready to be evaluated
        """
        history = model.fit(x, y, epochs=epoch, batch_size=500, verbose=0, validation_data=(xval, yval))
        return history

    def _evaluate_model(self, model, x, y):
        """
        Evalutes the model for Loss and accuracy metrics
        :param model: model that has to be evaluated
        :param local_X: independent variables
        :param local_y: outcome variable
        :return: dict {'loss': loss, 'accuracy': acc}
        """
        loss, acc = model.evaluate(x, y, verbose=0)
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

    def _output_df_to_csv(self, df=None, name_prefix="results", suffix=True):
        """
        Outputs the dataframe in to a csv file
        :param df: DataFrame to be stored in a csv file
        :return: None
        """
        curr_date_time = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
        if suffix:
           filename = name_prefix + "_" + curr_date_time + ".csv"
        else:
            filename = name_prefix + ".csv"
        df.to_csv(filename)
        print(f"Hyper parameter testing results saved to file - {filename}")

    def _output_df_to_json(self, df, name_prefix="weights", suffix=True):
        curr_date_time = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
        if suffix:
            filename = name_prefix + "_" + curr_date_time + ".json"
        else:
            filename = name_prefix + ".json"
        file = open(filename, 'w+')
        df.to_json(file)
        print(f"Weights stored in file {filename}")

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
    def usage():

        print("""
        Create a dictionary as below with the choice of options for each parameter
        =========================================================================
        
        version 2

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
        ====
        hidden_layer_neurons: Ex- [[11, 8, 5], [8, 6]] means there are 2 possible options for 
        hidden layer configuration. 
        1st = 3 layers with 11,8 and 5 neurons in each layer
        2nd = 2 layers with 8 and 6 neurons in each layer

        Alternate way to configure hidden_layer_neurons:
        dict as {1:(8, 13), 2:(5, 13), 3:(5, 8), 4:(5,6)}
        Key = refers to the layer number
        Value = range of number of neurons that can be configured in that layer
        Ex: 
        1 = refers to the 1st layer
        (8, 13) = means that the 1st layer can have from 8 to 13 neurons
        2 = refers to the 2nd layer
        (5, 13) = means that the 2nd layer can have from 5 to 13 neurons

        Execution:
        ========

        results = knf.fit(param_grid=param_grid, store_results=True, silent_mode=True, X=X_train, y=y_train)
        The returned results dataframe will have the list of all executions, from which you can choose the best option
        """)

    @staticmethod
    def get_best_result(df):
        return df.sort_values(by=['loss'], ascending=True).head(10).sort_values(by='accuracy', ascending=False).head(1)

    def _get_all_param_options(self, param_grid):
        all_options = []
        for key, values in param_grid.items():
            key_level = []
            for value in values:
                key_level.append({key: value})
            all_options.append(key_level)
        all_options

        return list(itertools.product(*all_options))

    def _format_model_train_results(self, fit_results):

        fit_results_combined = {}
        for results in fit_results:
            for keys, values in results.items():
                if keys in fit_results_combined:
                    fit_results_combined[keys].append(values)
                else:
                    fit_results_combined[keys] = [values]

        return fit_results_combined

    def _update_model_test_results(self, fit_results_combined, model_result, confidence_interval, std_limit):
        for keys, values in fit_results_combined.items():
            mean = np.mean(fit_results_combined[keys]) * 100
            std = np.std(fit_results_combined[keys]) * 100
            model_result['mean_' + keys] = mean
            model_result['min_' + keys + '(' + str(confidence_interval) + '%)'] = np.round(mean - (std_limit * std), 2)
            model_result['max_' + keys + '(' + str(confidence_interval) + '%)'] = np.round(mean + (std_limit * std), 2)

        return model_result

    def fit(self, param_grid, silent_mode=False, store_results=False, X=None, y=None, fold=5, confidence_interval=95,
            metrics=['accuracy'], reuse_weights=True, retain_weights=True):
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
        :param fold: Number of KFold's to be used for CrossValidation (Stratified KFold)
        :param metrics: Metrics to be estimated while fitting the model.
            Ref https://keras.io/api/metrics/#available-metrics
        :param reuse_weights: to force the models to re-use the already created weights
        :param retain_weights: to confirm if the user wants to store the weights for re-run again
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
        prev_run_weights = None

        try:

            should_run = False
            param_grid = self._validate_params(param_grid)
            std_limit = self._confidence_interval_std[confidence_interval]

            if reuse_weights:
                prev_run_weights_file = input("Please provide the weights file to be referred \n")
                prev_run_weights = pd.read_json(prev_run_weights_file)

            # get the provided list of values for all the parameters

            param_options = self._get_all_param_options(param_grid)

            #TODO
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

                for options in param_options:

                    option = {}
                    for curr_option in options:
                        option.update(curr_option)

                    choice_hidden_layer_neurons = option.get(self._input_hidden_layer_neurons,
                                                              self._default_hidden_layer_neurons)
                    choice_hidden_layer_activations = option.get(self._input_hidden_layer_activations,
                                                                 self._default_hidden_layer_activation)
                    choice_output_layer_neurons = option.get(self._input_output_layer_neurons,
                                                             self._default_output_layer_neurons)
                    choice_output_layer_activations = option.get(self._input_output_layer_activations,
                                                                 self._default_output_layer_activation)
                    choice_optimizers = option.get(self._input_optimizers, self._default_optimizers)
                    choice_epochs = option.get(self._input_epochs, self._default_epochs)
                    choice_learning_rates = option.get(self._input_learning_rates, self._default_learning_rate)
                    choice_loss_functions = option.get(self._input_loss_functions, self._default_loss)
                    choice_kernel_initializers = option.get(self._input_kernel_inializers,
                                                            self._default_kernel_initializers)
                    choice_bias_initializers = option.get(self._input_bias_initializers,
                                                          self._default_bias_initializers)

                    # Resetting the variables for every iteration
                    model = None
                    model_result = {}
                    start_time = time.time()

                    # Creating Model, adding layers, compiling and fit
                    model = self._create_model()
                    model = self._add_layers(model, choice_hidden_layer_neurons, choice_output_layer_neurons,
                                             choice_hidden_layer_activations, choice_output_layer_activations,
                                             choice_kernel_initializers, choice_bias_initializers, X)

                    option[self._input_hidden_layer_neurons] = self._convert_list_to_string(choice_hidden_layer_neurons)
                    option[self._input_output_layer_neurons] = self._convert_list_to_string(choice_hidden_layer_neurons)

                    # Capturing the values for every iteration to be stored in a dataframe
                    model_result = option
                    model_result[self._choice] = choice

                    if retain_weights:
                        self._retain_weights(model, model_result)

                    if reuse_weights:
                        model = self._set_weights(model, prev_run_weights, model_result, choice)

                    optimizer_instance = self._get_optimizer(choice_optimizers,
                                                             choice_learning_rates)
                    model = self._compile_model(model, optimizer_instance, choice_loss_functions, metrics)

                    # Sets the batch size to 10% of the total size
                    batch_size = math.ceil(X.shape[0] * 0.1)

                    fit_results = self._stratified_crossvalidation(model, choice_epochs, fold, batch_size, X, y,metrics)

                    fit_results_combined = self._format_model_train_results(fit_results)

                    model_result = self._update_model_test_results(fit_results_combined, model_result,
                                                                   confidence_interval, std_limit)

                    model_results.append(model_result)

                    end_time = time.time()
                    time_taken = round((end_time - start_time), 2)

                    # Clearing the output for every iteration
                    #IPython.display.clear_output(wait=True)

                    print(
                        f"Choice {choice}/{total_diff_choices}..., epoch = {choice_epochs}")

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

            if retain_weights:
                weight_df = pd.DataFrame(self._model_weights)
                self._output_df_to_json(weight_df, "model_weights", suffix=True)

            overall_time_end = time.time()
            overall_time_taken = round((overall_time_end - overall_time_start), 2)
            print(f"Completed running ... overall time taken = {overall_time_taken}")
            model_results_df = pd.DataFrame(model_results)

            if store_results:
                self._output_df_to_csv(model_results_df, "model_results", suffix=False)

            return model_results_df
