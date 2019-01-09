//
// Created by rneptune on 1/9/19.
//

#include <random>
#include <fstream>
#include "NeuralNetwork.h"

NeuralNetwork::NeuralNetwork(int totalLayers, int *nodeMap, double learningRate)
	: LEARNINGRATE(learningRate),
	  TOTALLAYERS(totalLayers) {

	this->nodeMap = new int[totalLayers];
	for(int i=0;i<TOTALLAYERS;i++) {
		this->nodeMap[i] = nodeMap[i];
	}

	nodes = new double*[TOTALLAYERS];
	error = new double*[TOTALLAYERS];
	for(int i=0;i<TOTALLAYERS;i++) {
		nodes[i] = new double[nodeMap[i]];
		error[i] = new double[nodeMap[i]];
	}

	weights = new double**[TOTALLAYERS-1];
	biasWeights = new double*[TOTALLAYERS];
	for(int i=0;i<TOTALLAYERS-1;i++) {
		weights[i] = new double*[nodeMap[i]];
		biasWeights[i] = new double[nodeMap[i+1]];
		for(int j=0;j<nodeMap[i];j++) {
			weights[i][j] = new double[nodeMap[i+1]];
		}
	}

	randomizeWeights(-1,1);
}

NeuralNetwork::~NeuralNetwork() {
	for(int i=0;i<TOTALLAYERS;i++) {
		delete(nodes[i]);
		delete(error[i]);
	}
	delete(nodes);
	delete(error);

	for(int i=0;i<TOTALLAYERS-1;i++) {
		for(int j=0;j<nodeMap[i];j++) {
			delete(weights[i][j]);
		}
		delete(weights[i]);
	}
	delete(weights);
	delete(biasWeights);

	delete(nodeMap);
}

void NeuralNetwork::randomizeWeights(double min, double max) {
	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_real_distribution<double> dist(min, max);

	for(int i=0;i<TOTALLAYERS-1; i++) {
		for(int j=0;j<nodeMap[i+1];j++) {
			for(int k=0;k<nodeMap[i];k++) {
				weights[i][k][j] = dist(gen);
			}
			biasWeights[i][j] = dist(gen);
		}
	}
}

double *NeuralNetwork::forwardPropagate(double *inputs) {
	for(int i=0;i<nodeMap[0];i++) {
		nodes[0][i] = inputs[i];
	}

	for(int i=1;i<TOTALLAYERS;i++) {
		for(int j=0;j<nodeMap[i];j++) {
			addNodeInputs(i,j);
		}
	}

	return nodes[TOTALLAYERS-1];
}

void NeuralNetwork::addNodeInputs(int layer, int node) {
	nodes[layer][node] = 0;
	for(int i=0;i<nodeMap[layer-1];i++) {
		nodes[layer][node] += nodes[layer-1][i] * weights[layer-1][i][node];
	}
	nodes[layer][node] += biasWeights[layer-1][node];
	nodes[layer][node] = sigmoid(nodes[layer][node]);
}

void NeuralNetwork::addErrorInputs(int layer, int node) {
	error[layer][node] = 0;
	for(int i=0;i<nodeMap[layer+1];i++) {
		error[layer][node] += error[layer+1][i] * weights[layer][node][i];
		updateWeight(layer,node,i);
	}

	if(node+MAXTHREADS<nodeMap[layer])
		addErrorInputs(layer,node+MAXTHREADS);
}

void NeuralNetwork::updateWeight(int layer, int inputNode,int outputNode) {
//	weights[layer][inputNode][outputNode] += LEARNINGRATE * error[layer+1][outputNode] * dLReLU(nodes[layer+1][outputNode]) * nodes[layer][inputNode];
	weights[layer][inputNode][outputNode] += LEARNINGRATE * error[layer+1][outputNode] * dsigmoid(nodes[layer+1][outputNode]) * nodes[layer][inputNode];
}


double *NeuralNetwork::doLearningTick(double *inputs, double *outputs) {
	forwardPropagate(inputs);
	return backPropagate(outputs);
}

double *NeuralNetwork::backPropagate(double *correct) {
	for(int i=0;i<nodeMap[TOTALLAYERS-1];i++) {
		error[TOTALLAYERS-1][i] = correct[i]-nodes[TOTALLAYERS-1][i];
	}

	for(int i=TOTALLAYERS-2;i>=0;i--) {
		for(int j=0;j<nodeMap[i];j++) {
			error[i][j] = 0;
			for(int k=0;k<nodeMap[i+1];k++) {
				error[i][j] += error[i+1][k] * weights[i][j][k];
			}
		}
	}

	for(int i=0;i<TOTALLAYERS-1;i++) {
		for(int j=0;j<nodeMap[i];j++) {
			for(int k=0;k<nodeMap[i+1];k++) {
				weights[i][j][k] += error[i+1][k] * dsigmoid(nodes[i+1][k]) * nodes[i][j] * LEARNINGRATE;
			}
		}
		for(int j=0;j<nodeMap[i+1];j++) {
			biasWeights[i][j] += error[i+1][j] * dsigmoid(nodes[i+1][j]) * LEARNINGRATE;
		}
	}

//	LEARNINGRATE = (4*LEARNINGRATE + fabs(error[TOTALLAYERS-1][0])*10)/5;

	return error[TOTALLAYERS-1];
}

void NeuralNetwork::saveWeightsToFile(std::string fileLocation) {
	std::ofstream outputFile;
	outputFile.open(fileLocation);

	for(int i=0;i<TOTALLAYERS-1; i++) {
		for(int j=0;j<nodeMap[i];j++) {
			for(int k=0;k<nodeMap[i+1];k++) {
				outputFile << weights[i][j][k] << std::endl;
			}
		}
	}

	for(int i=0;i<TOTALLAYERS-1; i++) {
		for(int j=0;j<nodeMap[i+1];j++) {
			outputFile << biasWeights[i][j] << std::endl;
		}
	}

	outputFile.close();
}

void NeuralNetwork::loadWeightsFromFile(std::string fileLocation) {
	std::ifstream inputFile;
	inputFile.open(fileLocation);

	std::vector<double> weightVector;
	double weight;
	while(inputFile >> weight) {
		weightVector.push_back(weight);
	}

	int iter = 0;
	for(int i=0;i<TOTALLAYERS-1; i++) {
		for(int j=0;j<nodeMap[i];j++) {
			for(int k=0;k<nodeMap[i+1];k++) {
				weights[i][j][k] = weightVector[iter++];
			}
		}
	}

	for(int i=0;i<TOTALLAYERS-1; i++) {
		for(int j=0;j<nodeMap[i+1];j++) {
			biasWeights[i][j] = weightVector[iter++];
		}
	}
}

void NeuralNetwork::mutateWeights(double stdDev) {
	std::random_device rd;
	std::mt19937 gen(rd());
	std::normal_distribution<double> dist(0,stdDev);

	for(int i=0;i<TOTALLAYERS-1; i++) {
		for(int j=0;j<nodeMap[i+1];j++) {
			for(int k=0;k<nodeMap[i];k++) {
				weights[i][k][j] += dist(gen);
			}
			biasWeights[i][j] += dist(gen);
		}
	}
}

void NeuralNetwork::copyWeights(NeuralNetwork *nn) {
	for(int i=0;i<TOTALLAYERS-1; i++) {
		for(int j=0;j<nodeMap[i+1];j++) {
			for(int k=0;k<nodeMap[i];k++) {
				weights[i][k][j] = nn->getWeights()[i][k][j];
			}
			biasWeights[i][j] = nn->getBiases()[i][j];
		}
	}
}

double *NeuralNetwork::getInputError() {
	return error[0];
}
