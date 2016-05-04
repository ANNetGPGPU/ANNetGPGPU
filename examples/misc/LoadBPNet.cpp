/*
 * main.cpp
 *
 *  Created on: 12.04.2010
 *      Author: dgrat
 */

#include <ANNet>
#include <ANContainers>
#include <ANMath>

#include "Samples.h"

#include <ctime>
#include <iostream>


int main(int argc, char *argv[]) {
	ANN::TrainingSet input;
	input.AddInput(fInp1, 3);
	input.AddOutput(fOut1, 6);
	input.AddInput(fInp2, 3);
	input.AddOutput(fOut2, 6);
	input.AddInput(fInp3, 3);
	input.AddOutput(fOut3, 6);
	input.AddInput(fInp4, 3);
	input.AddOutput(fOut4, 6);
	
	//SimpleNet net;
	ANN::BPNet net;
	net.ImpFromFS("foo.bar");

	net.SetTrainingSet(input);
	std::cout<< net <<std::endl;

	return 0;
}
