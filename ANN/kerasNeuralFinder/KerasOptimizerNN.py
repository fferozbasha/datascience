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
import multiprocessing

class KerasNeuralFinder:
    """
    Keras Neural Finder (KNF can be used to find the best neural network by providing variety of possible parameters.
    """

    __version__ = "1.1"

    # lookup dictionary to get the std dev based on the confidence interval
    _confidence_interval_std = {80: 1.282, 85: 1.440, 90: 1.645, 95: 1.960, 99: 2.576, 99.5: 2.807, 99.9: 3.291}

    # list to append the results of every iteration
    _model_weights = []

    # Default value for the inputs
    _default_hidden_layer_neurons = [11, 6]
    _default_output_layer_neurons = 1
    _default_hidden_layer_activation = 'relu'
    _default_output_layer_activation = 'relu'
    _default_optimizers = 'Adam'
    _default_epochs = '10'
    _default_learning_rate = 0.01
    _default_loss_functions = 'binary_crossentropy'
    _default_kernel_initializers = 'glorot_normal'
    _default_bias_initializers = 'random_normal'

    # Input name to be used by the users, while providing param_grid
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
    _time_taken = 'time_taken'

    _valid_input_params = (_input_hidden_layer_neurons, _input_output_layer_neurons, _input_hidden_layer_activations,
                           _input_output_layer_activations, _input_optimizers, _input_epochs, _input_learning_rates,
                           _input_loss_functions, _input_kernel_inializers, _input_bias_initializers)


    # Output instance variables after the run

    # best param combination estimated after the iterations
    best_params_ = None
    # best result with the values for all the metrics as well (based on CI)
    best_result_ = None
    # dataframe with the results of every iteration
    results_ = pd.DataFrame()

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
        Adds the layers to the provided model. # of layers is based on the number of different # of neurons
        provided in hidden_layer_neurons and output_layer_neurons, Sets the Activation function for Hidden & Output
        layer as provided. Other hyper parameters are kernel and bias initializer

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
                            bias_initializer=bias_initializers, name='hidden_layer_' + str(i)))
            i += 1
        model.add(Dense(output_layer_neurons, input_shape=(X.shape[1],), activation=activation_fn_output_layer,
                        name='output_layer'))

        return model

    def _get_layer_options(self, min_neurons_in_hidden_layers):
        """
        Generates all possible combinations of layers based on the ranges provided

        :param min_neurons_in_hidden_layers: dict defining the range of neuron count in each layer. Ex: {1:(8, 13),
        2:(5, 13), 3:(5, 8), 4:(5,6)}. Key-refers to the layer countm Value-refers to the range of neurons to be
        configured in that layer
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
                    layer_options.append([neuron_count])
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
        """
        Performs stratified cross validation for the model provided with the X and y variables

        :param model: compiled keras model that has to be used for cv
        :param epoch: number of epochs to be used
        :param fold: number of kfolds to be used for cross validation
        :param batch_size: number of rows to be sent in each batch while filling the model
        :param x: independent variables in array format
        :param y: outcome variable
        :metrics: metrics to be retrieved after the cv
        """

        results = []
        skf = StratifiedKFold(n_splits=fold, shuffle=False)

        for train_indices, val_indices in skf.split(x, y):
            iter_result = {}
            xtrain, xval = x[train_indices], x[val_indices]
            ytrain, yval = y[train_indices], y[val_indices]

            history = self._fit_model(model, epoch, batch_size, xtrain, ytrain, xval, yval)

            for metric, values in history.history.items():
                iter_result[metric] = values[-1]

            results.append(iter_result)

        return results

    def _compile_model(self, model, optimizer_instance, loss, metrics):
        """
        Compiles the model with loss, optimizer instance and as well the list of metrics provided.
        Other parameters are Optimizer instance and Learning rate

        :param model: Model configured with layers
        :param optimizer_instance: Optimizer method to be used
        :param loss: loss function to be used while evaluating the model
        :param metrics: list of metrics to be retrieved from the result
        :return: compiled model ready to be fit with dataset
        """
        model.compile(loss=loss, optimizer=optimizer_instance, metrics=metrics)
        return model

    # dormant implementation to be enhanced in future version
    def _retain_weights(self, model, model_info):
        """
        To retain the weights for re-run use to ensure reproducible results
        :param model: model which will be iterated for layers and corresponding weights will be retained
        :return: returns the dictionary of weights for each layer (layer name as key)
        """
        for layer in model.layers:
            weights = layer.get_weights()
            model_info[layer.name] = weights

        return model_info
        # self._model_weights.append(model_info)

    # dormant implementation to be enhanced in future version
    def _set_weights(self, model, prev_weights, model_info):

        if prev_weights.empty:
            print("Invalid file provided for the historical weights.")
            return model

        layer_info = {self._input_hidden_layer_neurons: model_info[self._input_hidden_layer_neurons],
                      self._input_hidden_layer_activations: model_info[self._input_hidden_layer_activations],
                      self._input_output_layer_activations: model_info[self._input_output_layer_activations],
                      self._input_optimizers: model_info[self._input_optimizers],
                      self._input_learning_rates: model_info[self._input_learning_rates],
                      self._input_kernel_inializers: model_info[self._input_kernel_inializers],
                      self._input_bias_initializers: model_info[self._input_bias_initializers],
                      self._input_epochs: model_info[self._input_epochs],
                      self._input_loss_functions: model_info[self._input_loss_functions]}

        idx = 0
        if not prev_weights.empty:
            for keys in layer_info.keys():
                value = layer_info[keys]
                if keys in prev_weights.columns:
                    if idx == 0:
                        filtered = prev_weights[prev_weights[keys] == value]
                    elif filtered.empty:
                        print(f"No matching previous weights found")
                        return model
                    else:
                        filtered = filtered[filtered[keys] == value]
                        if filtered.empty:
                            print(f"No matching previous weights found")
                            return model
                idx += 1

        if not filtered.empty:
            for layer in model.layers:
                model_weights = []
                layer_weights = filtered.get(layer.name, None)
                if not layer_weights.empty:
                    layer_weights_list = []
                    for weights in layer_weights:
                        for weight in weights:
                            weight_array = np.asarray(weight)
                            layer_weights_list.append(weight_array)
                        layer.set_weights(layer_weights_list)

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
        Evalutes the model for Loss accuracy and other metrics, as provided.

        :param model: model that has to be evaluated
        :param x: independent variables
        :param y: outcome variable
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

    def estimate_run(self, param_grid):
        """
        Estimates the number of choices possible based on the number of parameters defined

        :param param_grid: dictionary of params with wich the model is to be run
        :return: Total number of choices
        """
        return "Total number of choices " + str(len(self._get_all_param_options(self._validate_params(param_grid))))

    def _validate_params(self, param_grid):
        """
        Validates and transforms the params, as required

        :param param_grid: dictionary of list of values for each parameter
        :return: validated and formatted param_grid
        """
        for keys, values in param_grid.items():
            if keys in self._valid_input_params:
                if not isinstance(values, list):
                    if keys == self._input_hidden_layer_neurons:
                        param_grid[keys] = self._get_layer_options(values)
                    else:
                        param_grid[keys] = [values]
            else:
                print(f"Ignoring input {keys} as this is an unsupported parameter")
        return param_grid


    def get_best_result(self, by={'accuracy': 'high'}, count=1):
        """
        To get the best result based on the filter provided. By default, provides the rows with maximum accuracy.

        :param by: the filter based on which the results has to be retrieved. Can be provided as a string, in which case
        the result with maximum value of that parameter is returned. Else, can be provided as dict with key as the
        parameter and the value as either 'high' or 'low'. The filter is done for the keys provided in the same
        sequence as provided
        :param count: number of top entries to be returned
        :return: Dataframe with the filtered rows
        """

        if isinstance(by, str):
            by = {by: 'high'}

        if isinstance(by, dict):
            temp = pd.DataFrame()
            for by_variable in by.keys():
                if temp.empty:
                    temp = self.results_
                ascending = True if by[by_variable] == 'low' else False
                temp = temp.sort_values(by=by_variable, ascending=ascending)
        else:
            print(f"Invalid format provided for {by}")
            return None

        return temp.head(count)

    def get_best_params(self):
        self.best_params_ = {}
        for keys in self._valid_input_params:
            value = self.best_result_[keys].values[0]
            if keys in (self._input_hidden_layer_neurons, self._input_output_layer_neurons):
                value = value.split(',')
            self.best_params_[keys] = value

        return self.best_params_

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
            model_result[keys] = mean
            model_result['min_' + keys + '(' + str(confidence_interval) + '%)'] = np.round(mean - (std_limit * std), 2)
            model_result['max_' + keys + '(' + str(confidence_interval) + '%)'] = np.round(mean + (std_limit * std), 2)

        return model_result

    def fit(self, param_grid,X=None, y=None, fold=5, confidence_interval=95,
            metrics=['accuracy']):
        """
        Creates and fits the model with all the combinations based on param_grid
        :param param_grid: Dictionary of parameters on which the model has to be created and run. Use help() to get to
        know the list of possible params
        :param X: independent variables (as an ndarray)
        :param y: outcome variables (as an ndarray)
        :param fold: Number of KFold's to be used for CrossValidation (Stratified KFold)
        :param confidence_interval: confidence interval to provide minimum/maximum of metric values
        :param metrics: Metrics to be estimated while fitting the model. Ref
        https://keras.io/api/metrics/#available-metrics
        :return: best params based on minimum loss and high accuracy
        """

        overall_time_start = 0
        model_results = None
        loss_history = []
        accuracy_history = []
        time_taken_history = []

        # prev_run_weights = None

        try:

            param_grid = self._validate_params(param_grid)
            std_limit = self._confidence_interval_std[confidence_interval]

            # get the provided list of values for all the parameters
            param_options = self._get_all_param_options(param_grid)

            # TODO
            total_diff_choices = len(param_options)

            # confirms with user if they really intend to run the iterative model for hyperparameter tuning

            choice = 1

            model_results = []
            overall_time_start = time.time()

            print(f"Starting ...")

            for options in param_options:

                option = {}
                for curr_option in options:
                    option.update(curr_option)

                """
                Getting the list of values for the current iteration
                """
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
                choice_loss_functions = option.get(self._input_loss_functions, self._default_loss_functions)
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

                # getting an optimizer instance based on optimizer & learning_rate
                optimizer_instance = self._get_optimizer(choice_optimizers,
                                                         choice_learning_rates)

                # compiling the model based on the loss function and the required metrics
                model = self._compile_model(model, optimizer_instance, choice_loss_functions, metrics)

                # Capturing the values for every iteration to be stored in a dataframe
                model_result = option
                model_result[self._choice] = choice

                # Converting the hidden/output neuron option from [a, b, c] to 'a,b,c'
                option[self._input_hidden_layer_neurons] = self._convert_list_to_string(choice_hidden_layer_neurons)
                option[self._input_output_layer_neurons] = self._convert_list_to_string(choice_output_layer_neurons)

                # Sets the batch size to 10% of the total size
                batch_size = math.ceil(X.shape[0] * 0.1)

                """
                Future enhancement 
                if reuse_weights:
                    model = self._set_weights(model, prev_run_weights, model_result)
                """

                # Running stratified cv for the model with specified folds
                fit_results = self._stratified_crossvalidation(model, choice_epochs, fold, batch_size, X, y,
                                                               metrics)

                # getting the mean/min/max metric values for the run based on the confidence interval
                fit_results_combined = self._format_model_train_results(fit_results)

                # updating the run results to the dict
                model_result = self._update_model_test_results(fit_results_combined, model_result,
                                                               confidence_interval, std_limit)

                # maintaining a list of loss/accuracy and time taken
                loss = round(model_result['loss'],2)
                accuracy = round(model_result['accuracy'],2)
                loss_history.append(loss)
                accuracy_history.append(accuracy)

                end_time = time.time()
                time_taken = round((end_time - start_time), 2)

                time_taken_history.append(time_taken)

                model_result[self._time_taken] = time_taken

                # appending the run details to the list
                model_results.append(model_result)

                # Clearing the output for every iteration
                IPython.display.clear_output(wait=True)

                avg_time_taken = np.average(time_taken_history)
                remaining_choices = total_diff_choices - choice
                remaining_elapsed_time = round(remaining_choices * avg_time_taken,2)

                # Printing summary of run until now

                print(f"Running model for next choice.....")

                print(f"\nMinimum loss until now {min(loss_history)}")
                print(f"Maximum accuracy until now {max(accuracy_history)} %")

                print(f"\nPrevious run")
                print(f"============")
                print(
                    f"Choice {choice}/{total_diff_choices}..., epoch = {choice_epochs}, "
                    f"time_taken={time_taken} seconds, loss={loss}, accuracy={accuracy}")

                print("\nParameters are :")
                print("================\n")
                for option in options:
                    for key in option.keys():
                        print(f"{key} = {option[key]}")

                if remaining_choices > 0:
                    print(f"\nApprox time elapsed {remaining_elapsed_time} seconds")

                choice += 1

        except KeyboardInterrupt:
            print(f"User interrupted the process. Model run until choice {choice}")

        except Exception as e:
            print(f"Exception occurred !!!, {str(e)}")
            traceback.print_exc()

        finally:

            overall_time_end = time.time()
            overall_time_taken = round((overall_time_end - overall_time_start), 2)

            self.results_ = pd.DataFrame(model_results)
            self.best_result_ = self.get_best_result()
            self.best_params_ = self.get_best_params()

            print(f"\nCompleted running ... overall time taken = {overall_time_taken}")

            return self.best_params_
