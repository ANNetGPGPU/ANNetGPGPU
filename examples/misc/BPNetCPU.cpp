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
	ANN::BPNet cpu_one;

	ANN::BPLayer layer1(56, ANN::ANLayerInput);
        ANN::BPLayer layer2(64, ANN::ANLayerHidden);
        ANN::BPLayer layer3(64, ANN::ANLayerHidden);
        ANN::BPLayer layer4(64, ANN::ANLayerHidden);
	ANN::BPLayer layer5(9, ANN::ANLayerOutput);

	layer1.ConnectLayer(&layer2);
	layer2.ConnectLayer(&layer3);
        layer3.ConnectLayer(&layer4);
        layer4.ConnectLayer(&layer5);
        
	cpu_one.AddLayer(&layer1);
	cpu_one.AddLayer(&layer2);
        cpu_one.AddLayer(&layer3);
        cpu_one.AddLayer(&layer4);
        cpu_one.AddLayer(&layer5);

	ANN::TrainingSet input;
	input.AddInput(fInp1, 56);
	input.AddOutput(fOut1, 9);
	input.AddInput(fInp2, 56);
	input.AddOutput(fOut2, 9);
	input.AddInput(fInp3, 56);
	input.AddOutput(fOut3, 9);
	input.AddInput(fInp4, 56);
	input.AddOutput(fOut4, 9);
	input.AddInput(fInp5, 56);
	input.AddOutput(fOut5, 9);
	input.AddInput(fInp6, 56);
	input.AddOutput(fOut6, 9);
	input.AddInput(fInp7, 56);
	input.AddOutput(fOut7, 9);
	input.AddInput(fInp8, 56);
	input.AddOutput(fOut8, 9);
	input.AddInput(fInp9, 56);
	input.AddOutput(fOut9, 9);
	input.AddInput(fInp10, 56);
	input.AddOutput(fOut10, 9);
	
	std::vector<float> errors;
	cpu_one.SetLearningRate(0.5);
	cpu_one.SetMomentum(0);
	cpu_one.SetWeightDecay(0);
	cpu_one.SetTrainingSet(input);

	bool b = false;
	float f;
	errors = cpu_one.TrainFromData(300, 0, b, f);
	std::cout<< cpu_one <<std::endl;
/*
	cpu_one.ExpToFS("foo.bar");
	cpu_one.ImpFromFS("foo.bar");

	cpu_one.SetTrainingSet(input);
	std::cout<< cpu_one <<std::endl;
*/
	return 0;
}
