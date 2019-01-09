//
// Created by rneptune on 1/9/19.
//

#ifndef GAN_NAME_GENERATOR_BP_NEURALNETWORK_H
#define GAN_NAME_GENERATOR_BP_NEURALNETWORK_H


#include <cmath>
#include <string>

class NeuralNetwork {
public:
	NeuralNetwork(int totalLayers, int *nodeMap, double learningRate);
	~NeuralNetwork();
	double *doLearningTick(double *inputs, double *outputs);
	double *forwardPropagate(double *inputs);
	double *backPropagate(double *correct);
	double *getInputError();
	void saveWeightsToFile(std::string fileLocation);
	void loadWeightsFromFile(std::string fileLocation);
	void mutateWeights(double stdDev);
	void copyWeights(NeuralNetwork *nn);
	double ***getWeights() { return weights; }
	double **getBiases() { return  biasWeights; }
	int *nodeMap;
private:
	double LEARNINGRATE;
	int const TOTALLAYERS;
	int static const MAXTHREADS = 1;

	double ***weights;
	double **biasWeights;
	double **nodes;
	double **error;

	double sigmoid(double in) { return 1/(1+exp(-1*in)); }
	double dsigmoid(double in) { return in*(1-in); }

	void randomizeWeights(double min, double max);
	void addNodeInputs(int layer, int node);
	void addErrorInputs(int layer, int node);
	void updateWeight(int layer, int inputNode,int outputNode);
};


#endif //GAN_NAME_GENERATOR_BP_NEURALNETWORK_H
