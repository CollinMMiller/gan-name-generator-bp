#include <iostream>
#include <vector>
#include <fstream>
#include <random>
#include <unistd.h>
#include "NeuralNetwork.h"

void setInputfromString(double *, std::string);
void minMaxOutput(double *);
std::string getWordFromOutput(double *);

/*
 * These arrays define the layout of the neural networks
 * Each element is a layer, the value defines the number of nodes in that layer
 * Adding more elements creates more layers
 * First and last elements will be overwritten
 * example: ...NodeMap[] = {0, 100, 600, 350, 0};
 */
int genNodeMap[] = {0,500,500,0};	//For the generator
int disNodeMap[] = {0,500,500,0};	//For the discriminator
bool continueNameAfterSpace = false;	//Allows letters following a ' '(Space) in generated names

int randomNodes = 50;			//Number of random input nodes for generator
int maxLetters = 16;			//Maximum letters in name
int trialsPerPrint = 1000;		//Number of cycles before printing an output

int discrimCount = 0;
int maxCountBeforeReset = 3;

int main(int argc, char **argv) {

	if(argc < 2) {
		std::cout << "Please include path to training file" << std::endl;
		return 0;
	}

	//Sets values for genNodeMap
	int genNumberOfLayers = sizeof(genNodeMap) / sizeof(int);
	genNodeMap[0] = randomNodes;
	genNodeMap[genNumberOfLayers - 1] = 27 * maxLetters;

	//Sets values for disNodeMap
	int disNumberOfLayers = sizeof(disNodeMap)/ sizeof(int);
	disNodeMap[0] = 27 * maxLetters;
	disNodeMap[disNumberOfLayers-1] = 1;

	//Loads training names into realNames
	std::vector<std::string> realNames;
	std::ifstream nameFile(argv[1]);
	std::string name;
	while (nameFile >> name)
		realNames.push_back(name);


	//Creates the generator and discriminator with .1 as the learning rate
	NeuralNetwork generator(genNumberOfLayers,genNodeMap, .1, true);
	NeuralNetwork discriminator(disNumberOfLayers,disNodeMap,.1, true);

	//Loads saved weights from file
//	generator.loadWeightsFromFile("Gweights.txt");
//	discriminator.loadWeightsFromFile("Dweights.txt");

	//Creates random double generator between 1,1
	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_real_distribution<double> dist(1,1);

	int discriminatorRight = 0;	//Defines and initializes number of times the discriminator is right

	//Initializes various variables
	double inputs[randomNodes];
	double realInput[27 * maxLetters];
	double *rawOutputs;
	double genOutputs[27 * maxLetters];
	double *disOutputs;
	double *inputError;
	double expectedDisOutput[2];

	for(int trial=0;trial>-1;trial++) {
		/*
		 * Generated Name
		 */

		//Sets the generator inputs to random values
		for(int i=0;i<randomNodes;i++) {
			inputs[i] = dist(gen);
		}
		rawOutputs = generator.forwardPropagate(inputs);	//Runs generator
		for(int i=0;i<27 * maxLetters;i++) {
			genOutputs[i] = rawOutputs[i];			//Copies rawOutputs array to genOutputs
		}
		minMaxOutput(genOutputs);	//Sets all elements genOutputs to 1 or 0,
		// 1 if the are the highest letter, 0 if not

		//Sets expected output nodes for discriminator
		expectedDisOutput[0] = 0;	//1 if it thinks the name is real, 0 if it thinks it is fake
		disOutputs = discriminator.forwardPropagate(genOutputs);	//Runs Discriminator
		if(disOutputs[0] < .5)	//if the discriminator thinks it's fake
			discriminatorRight++;

		//discriminator corrects itself based on the expected output
		discriminator.backPropagate(expectedDisOutput);
		//Gets the calculated error for the inputs of the discriminator
		inputError = discriminator.getInputError();
		//The inputs are multiplied by -1 and set to 0 or 1
		for(int i=0;i<27 * maxLetters;i++) {
			inputError[i] *= -1;
		}
		minMaxOutput(inputError);

		//The generator corrects itself
		generator.backPropagate(inputError);

		/*
		 * Real Name
		 */

		//Expected output form real name
		expectedDisOutput[0] = 1;

		std::string selectedName = realNames[rand() % realNames.size()];	//Chooses a random name

		setInputfromString(realInput, selectedName);		//Set realInput to 1s and 0s based on name

		//Same as before just with a real name
		disOutputs = discriminator.forwardPropagate(realInput);
		discriminator.backPropagate(expectedDisOutput);

		//Print info about training
		if(trial % trialsPerPrint == 0) {
			double disPercentage = 100 * ((double)discriminatorRight/(double)trialsPerPrint);
			std::cout << trial << ": " << getWordFromOutput(rawOutputs) << " | " << "Discriminator Wins: "
				<< disPercentage << "%" << std::endl;
//			generator.saveWeightsToFile("Gweights.txt");
//			discriminator.saveWeightsToFile("Dweights.txt");

			if (discriminatorRight == trialsPerPrint) {
				discrimCount++;
				if (discrimCount >= maxCountBeforeReset) {
					std::cout << "Reset Discriminator" << std::endl;
					discriminator.randomize();
				}
			} else {
				discrimCount = 0;
			}

			discriminatorRight = 0;	//Resets discriminatorRight
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
	bool endEarly = false;
	for(int i=0;i<maxLetters;i++) {
		int bestIndex = -1;
		double bestValue = -1000;
		for(int selected=0;selected<27;selected++) {
			if(input[i*27+selected] > bestValue) {
				bestIndex = selected;
				bestValue = input[i*27+selected];
			}
		}
		if(bestIndex == 0 || endEarly) {
			name.append(" ");
			if(!continueNameAfterSpace)
				endEarly = true;
		} else {
			char c = (char)(bestIndex+64);
			name.append(std::string(1,c));
		}
	}

	return name;
}
