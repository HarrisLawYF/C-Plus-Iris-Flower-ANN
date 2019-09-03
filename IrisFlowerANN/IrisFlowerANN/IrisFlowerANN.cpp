// IrisFlowerANN.cpp : This file contains the 'main' function. Program execution begins and ends there.
// https://cs230.stanford.edu/files/C1M3.pdf

#include "pch.h"
#include <iostream>
#include <vector>
#include <cstdlib>
#include <assert.h>
#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

using namespace std;
struct Connection {
	double weight;
	double deltaWeight;
};

bool is_number(const std::string& s)
{
	try
	{
		std::stod(s);
	}
	catch (...)
	{
		return false;
	}
	return true;
}

void showVectorVals(string label, vector<double> &v)
{
	cout << label << " ";
	for (unsigned i = 0; i < v.size(); ++i) {
		cout << v[i] << " ";
	}

	cout << endl;
}

class Neuron; // so that the typedef recognizes this
typedef vector<Neuron> Layer;
class Neuron {
public:
	Neuron(unsigned num_outputs, unsigned n_index);
	void setOutputVal(double val) { m_output_val = val; }
	double getOutputVal(void) const { return m_output_val; }
	double getInputVal(void) const { return m_input_val; }
	void feedForward(const Layer &prev_layer);
	void calculateOutputGradients(double target_val, Layer &prev_layer);
	void calculateHiddenGradients(const Layer &next_layer, const Layer &input_layer);
	void updateInputWeights(Layer &prev_layer);
private:
	static double eta;		// [0.0...1.0] overall training rate, 0.0 -> slow learner, 0.2 -> medium learner, 1.0 -> reckless learner
	static double alpha;	// [0.0....n] multiplier of last weight change (momentum), 0.0 -> no momentum, 0.5 -> moderate momentum
	static double transferFunction(double x);
	static double transferFunctionDerivative(double x);
	static double randomWeight(void) { return rand() / double(RAND_MAX); }
	double costFunc(const Layer &next_layer) const;
	double m_output_val;
	double m_input_val;
	double delta;
	vector<Connection> m_output_weights;
	unsigned m_n_index;
	double m_gradient;
};

/*The learning rate is a hyperparameter that controls how much to change the model in response to the estimated error each time the model
weights are updated.
Choosing the learning rate is challenging as a value too small may result in a long training process that could get stuck,
whereas a value too large may result in learning a sub - optimal set of weights too fast or an unstable training process.*/
double Neuron::eta = 0.15;  // overall net learning rate
//speed up convergence of first order optimization methods like gradient descent
double Neuron::alpha = 0.5; // momentum, multiplier of last deltaWeight

Neuron::Neuron(unsigned num_outputs, unsigned n_index) {
	for (unsigned c = 0; c < num_outputs; ++c) {
		m_output_weights.push_back(Connection());
		m_output_weights.back().weight = randomWeight();
	}
	m_n_index = n_index;
}

void Neuron::feedForward(const Layer &prev_layer) {
	double sum = 0.0;

	//output = sum of(i * w)
	for (unsigned n = 0; n < prev_layer.size(); ++n) {
		sum += prev_layer[n].getOutputVal() * prev_layer[n].m_output_weights[m_n_index].weight;
	}
	// store the sum as input value
	m_input_val = sum;
	// set the output of a specific node, this is only for hidden nodes, this value will get replaced by other
	// in feedForwardSoftmax
	m_output_val = Neuron::transferFunction(sum);
}

void Neuron::calculateOutputGradients(double target_val, Layer &prev_layer) {
	delta = m_output_val - target_val;
	m_gradient = prev_layer[m_n_index].getOutputVal() * delta;
}

void Neuron::calculateHiddenGradients(const Layer &next_layer, const Layer &input_layer) {
	// sum of derivative of next layer
	double dow = costFunc(next_layer);
	for (unsigned i = 0; i < input_layer.size(); ++i) {
		m_gradient += input_layer[i].getOutputVal() * dow * Neuron::transferFunctionDerivative(m_input_val);
	}
}

void Neuron::updateInputWeights(Layer &prev_layer) {
	for (unsigned n = 0; n < prev_layer.size(); ++n) {
		Neuron &neuron = prev_layer[n];
		double oldDeltaWeight = neuron.m_output_weights[m_n_index].deltaWeight;
		/*double newDeltaWeight = eta* neuron.getOutputVal() * m_gradient+ alpha * oldDeltaWeight;*/
		double newDeltaWeight = eta * m_gradient;
		// delta weight of a node
		neuron.m_output_weights[m_n_index].deltaWeight = newDeltaWeight;
		// weight of a node
		neuron.m_output_weights[m_n_index].weight -= newDeltaWeight;
	}
}

double Neuron::costFunc(const Layer &next_layer) const {
	double sum = 0.0;

	for (unsigned n = 0; n < next_layer.size() - 1; ++n) {
		sum += m_output_weights[n].weight * next_layer[n].delta;
	}

	return sum;
}

double Neuron::transferFunction(double x) {
	// tanh - output range [-1.0..1.0]
	// Sigmoid function can be used here as well, we just need to scale it
	//return tanh(x);
	return 1 / (1 + exp(-x));
}

double Neuron::transferFunctionDerivative(double x) {
	//return 1.0 - tanh(x) * tanh(x); //this is the derivative of tanh
	return Neuron::transferFunction(x) * (1- Neuron::transferFunction(x));
}


class Net {
public:
	Net(const vector<unsigned> &topology);
	void feedForward(const vector<double> &input_vals);
	void feedForwardSoftmax();
	void backProp(const vector<double> &target_vals);
	void getResults(vector<double> &result_vals) const;
	double getRecentAverageError() { return m_recent_average_error; }
private:
	vector<Layer> m_layers;
	double m_error;
	double m_recent_average_error;
	double m_recent_average_smoothing_factor;
};

Net::Net(const vector<unsigned> &topology) {
	unsigned num_layers = topology.size();

	// Create number of layers
	for (unsigned layer_num = 0; layer_num < num_layers; layer_num++) {
		m_layers.push_back(Layer());
		unsigned num_outputs = layer_num == topology.size() - 1 ? 0 : topology[layer_num + 1];
		// Create nodes in each layer
		// <= includes bias neuron
		for (unsigned neuron_num = 0; neuron_num <= topology[layer_num]; ++neuron_num) {
			// .back() give the most recent container
			// Neuron object generates a random value automatically when initialized
			m_layers.back().push_back(Neuron(num_outputs, neuron_num));
		}

		// Set the bias output value as 1 by default (bias doesn't require input)
		m_layers.back().back().setOutputVal(1.0);
	}
};

void Net::feedForward(const vector<double> &input_vals) {
	assert(input_vals.size() == m_layers[0].size() - 1);
	for (unsigned i = 0; i < input_vals.size(); ++i) {
		m_layers[0][i].setOutputVal(input_vals[i]);
	}

	// Forward propagation for hidden layers
	for (unsigned layer_num = 1; layer_num < m_layers.size()-1; ++layer_num) {
		Layer &prev_layer = m_layers[layer_num - 1];
		Layer &current_layer = m_layers[layer_num - 1];
		for (unsigned n = 0; n < m_layers[layer_num].size() - 1; ++n) {
			m_layers[layer_num][n].feedForward(prev_layer);
		}
	}
}

void Net::feedForwardSoftmax() {
	Layer &output_layer = m_layers.back();
	Layer &last_hidden_layer = m_layers[m_layers.size()-2];
	double sum = 0.0;
	for (unsigned n = 0; n < output_layer.size()-1; ++n) {

		//This is to get the exp of output in each node
		output_layer[n].feedForward(last_hidden_layer);
		sum += exp(output_layer[n].getInputVal());
	}

	//softmax final result
	for (unsigned n = 0; n < output_layer.size()-1; ++n) {
		output_layer[n].setOutputVal(exp(output_layer[n].getInputVal())/sum);
	}
}

void Net::backProp(const vector<double> &target_vals) {
	// Calculate overall net error (RMS of output neuron_errors)
	Layer &output_layer = m_layers.back();
	Layer &prev_layer = m_layers[m_layers.size()-2];
	//------------------------------------- calculate error rate--------------------------------------------------
	m_error = 0.0; // accummulate sum of error, it should be a new error for every training

	for (unsigned n = 0; n < output_layer.size() - 1; ++n) {
		double delta = target_vals[n] - output_layer[n].getOutputVal();
		m_error += delta * delta;
	}

	m_error /= output_layer.size() - 1; //get average error squared
	m_error = sqrt(m_error); //RMS

	// Implement a recent average measurement (To show how good the training is)
	m_recent_average_error = (m_recent_average_error * m_recent_average_smoothing_factor + m_error) / (m_recent_average_smoothing_factor + 1.0);
	//------------------------------------------------------------------------------------------------------------
	// Calculate output layer gradients
	for (unsigned n = 0; n < output_layer.size() - 1; ++n) {
		output_layer[n].calculateOutputGradients(target_vals[n], prev_layer);
	}

	// Calculate gradients on hidden layers
	// Note: one node only has a gradient to update all the weight values that it contains to all next nodes
	// therefore, the calculation of gradient involves the sum of all weights
	// also, it is related to the gradient of the next nodes
	for (unsigned layer_num = m_layers.size() - 2; layer_num > 0; --layer_num) {
		Layer &hidden_layer = m_layers[layer_num];
		Layer &next_layer = m_layers[layer_num + 1];
		Layer &first_layer = m_layers[0];
		for (unsigned n = 0; n < hidden_layer.size(); ++n) {
			hidden_layer[n].calculateHiddenGradients(next_layer, first_layer);
		}
	}

	// For all layers from outputs to first hidden layer, update connection weights
	for (unsigned layer_num = m_layers.size() - 1; layer_num > 0; --layer_num) {
		Layer &layer = m_layers[layer_num];
		Layer &prev_layer = m_layers[layer_num - 1];

		for (unsigned n = 0; n < layer.size() - 1; ++n) {
			layer[n].updateInputWeights(prev_layer);
		}
	}
}


void Net::getResults(vector<double> &result_vals) const {
	result_vals.clear();

	for (unsigned n = 0; n < m_layers.back().size() - 1; ++n) {
		result_vals.push_back(m_layers.back()[n].getOutputVal());
	}
};

class Data {
public:
	Data(string file_name);
	vector<vector<double>> getOutputs() { return m_outputs; };
	vector<vector<double>> getInputs() { return m_inputs; };
	int getNumOfInputs() { return m_inputs[0].size(); };
private:
	vector<vector<double>> m_inputs;
	vector<vector<double>> m_outputs;
};

Data::Data(string file_name) {
	string line;
	string value;
	ifstream m_file(file_name);
	if (m_file.is_open())
	{
		while (getline(m_file, line)) {
			stringstream ss(line);
			vector<double> m_input;
			vector<double> m_output;
			if (line.length() > 0) {
				while (getline(ss, value, ',')) {
					if (is_number(value)) {
						m_input.push_back(stod(value));
					}
					else {
						if (value.compare("setosa") == 0) {
							m_output.push_back(1);
							m_output.push_back(0);
							m_output.push_back(0);
						}
						else if (value.compare("versicolor") == 0) {
							m_output.push_back(0);
							m_output.push_back(1);
							m_output.push_back(0);
						}
						else {
							m_output.push_back(0);
							m_output.push_back(0);
							m_output.push_back(1);
						}
					}
				}
				m_inputs.push_back(m_input);
				m_outputs.push_back(m_output);
			}
			else {
				break;
			}
			
		}
		m_file.close();
	}
}

int main()
{
	//TODO: Randomize the data inputs to make it more generic (to avoid overfitting)
	//TODO: use 20% of the data as the dataset after randomized

	Data myData("iris.txt");
	vector<unsigned> topology;
	topology.push_back(myData.getNumOfInputs());
	for (unsigned count = 0; count < 10; count++) {
		topology.push_back(10);
	}
	topology.push_back(3);
	Net myNet(topology);

	vector<double> input_vals, target_vals, result_vals;
	unsigned repeating = 0;
	for (unsigned count = 0; count < 5000; count++) {

		if (repeating >= myData.getInputs().size()) {
			repeating = 0;
		}
		input_vals = myData.getInputs()[repeating];
		showVectorVals("Inputs:", input_vals);
		myNet.feedForward(input_vals);
		myNet.feedForwardSoftmax();

		// Collect the net's actual output results:
		myNet.getResults(result_vals);
		showVectorVals("Outputs:", result_vals);

		// Train the net what the outputs should have been:
		target_vals = myData.getOutputs()[repeating];
		showVectorVals("Targets:", target_vals);
		assert(target_vals.size() == topology.back());

		myNet.backProp(target_vals);

		repeating++;
		// Report how well the training is working, average over recent samples:
		cout << "Net recent average error: "<< myNet.getRecentAverageError() << endl;
	}
}