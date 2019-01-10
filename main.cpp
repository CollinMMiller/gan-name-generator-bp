#include <iostream>
#include <vector>
#include <fstream>
#include <random>
#include "NeuralNetwork.h"

void setInputfromString(double *, std::string);
void minMaxOutput(double *);
std::string getWordFromOutput(double *);

//NOTE: 0 is fake; 1 is real
int genNodeMap[] = {0,100,0};
int disNodeMap[] = {0,100,2};
int randomNodes = 10;
int maxLetters = 16;

int main() {
	int numOfLayers = sizeof(genNodeMap) / sizeof(int);
	genNodeMap[0] = randomNodes;
	genNodeMap[numOfLayers - 1] = 27 * maxLetters;
	disNodeMap[0] = 27 * maxLetters;

	std::vector<std::string> realNames;
	std::ifstream nameFile("/home/rneptune/Desktop/boynames.txt");
	std::string name;
	while (nameFile >> name)
		realNames.push_back(name);

	NeuralNetwork generator(numOfLayers,genNodeMap,.1);
	generator.loadWeightsFromFile("/home/rneptune/Desktop/Gweights.txt");
	NeuralNetwork discriminator(numOfLayers,disNodeMap,.1);
	discriminator.loadWeightsFromFile("/home/rneptune/Desktop/Dweights.txt");

	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_real_distribution<double> dist(-1, 1);

	int discriminatorRight = 0;

	double inputs[randomNodes];
	double realInput[27 * maxLetters];
	double genOutputs[27 * maxLetters];
	double *disOutputs;
	double expectedDisOutput[2];

	//Infinite Trials
	for(int trial=0;trial>-1;trial++) {

		expectedDisOutput[0] = 1;
		expectedDisOutput[1] = 0;

		for(int i=0;i<randomNodes;i++) {
			inputs[i] = dist(gen);
		}

		double *rawOutputs = generator.forwardPropagate(inputs);
		for(int i=0;i<27 * maxLetters;i++) {
			genOutputs[i] = rawOutputs[i];
		}
		minMaxOutput(genOutputs);

		disOutputs = discriminator.forwardPropagate(genOutputs);
		if(disOutputs[0] > disOutputs[1])
			discriminatorRight++;
		discriminator.backPropagate(expectedDisOutput);

		double *inputError = discriminator.getInputError();
		for(int i=0;i<27 * maxLetters;i++) {
			inputError[i] *= -1;
		}
		minMaxOutput(inputError);

		generator.backPropagate(inputError);

		expectedDisOutput[0] = 0;
		expectedDisOutput[1] = 1;

		std::string selectedName = realNames[rand()%realNames.size()];
		setInputfromString(realInput, selectedName);
		discriminator.doLearningTick(realInput, expectedDisOutput);


		if(trial % 1000 == 0) {
			std::cout << trial << ": " << getWordFromOutput(rawOutputs) << " | " <<discriminatorRight << std::endl;
			generator.saveWeightsToFile("/home/rneptune/Desktop/Gweights.txt");
			discriminator.saveWeightsToFile("/home/rneptune/Desktop/Dweights.txt");
			discriminatorRight = 0;
		}
	}
}


void setInputfromString(double *input, std::string str) {
	for(int i=0;i<27*maxLetters;i++)
		input[i] = 0;
	for(int i=0;i<maxLetters;i++) {
		if(i >= str.length())
			input[i*27] = 1;
		else {
			int c = int(str[i]);
			input[i*27+(c-64)] = 1;
		}
	}
}

void minMaxOutput(double *input) {
	for(int i=0;i<maxLetters;i++) {
		int bestIndex = -1;
		double bestValue = -1000;
		for(int j=0;j<27;j++) {
			if(input[27*i + j] > bestValue) {
				bestIndex = j;
				bestValue = input[27*i + j];
			}
			input[27*i+j] = 0;
		}
		input[i*27+bestIndex] = 1;
	}
}


std::string getWordFromOutput(double *input) {
	std::string name;
	for(int i=0;i<maxLetters;i++) {
		int bestIndex = -1;
		double bestValue = -1000;
		for(int selected=0;selected<27;selected++) {
			if(input[i*27+selected] > bestValue) {
				bestIndex = selected;
				bestValue = input[i*27+selected];
			}
		}
		if(bestIndex == 0) {
			name.append(" ");
		} else {
			char c = (char)(bestIndex+64);
			name.append(std::string(1,c));
		}
	}

	return name;
}