/*
 * main.cpp
 *
 *  Created on: 12.04.2010
 *      Author: dgrat
 */

#include <ANNet>
#include <ANContainers>
#include <ANMath>

#include <Samples.h>

#include <ctime>
#include <iostream>


int main(int argc, char *argv[]) {
	ANN::BPNet<float, ANN::fcn_log<float>> cpu_one;
        
	ANN::BPLayer<float, ANN::fcn_log<float>> *layer1 = cpu_one.AddLayer(56, ANN::ANLayerInput);
	ANN::BPLayer<float, ANN::fcn_log<float>> *layer2 = cpu_one.AddLayer(64, ANN::ANLayerHidden);
	ANN::BPLayer<float, ANN::fcn_log<float>> *layer3 = cpu_one.AddLayer(9, ANN::ANLayerOutput);

	layer1->ConnectLayer(layer2);
	layer2->ConnectLayer(layer3);
	
	ANN::TrainingSet<float> input;
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
	
	ANN::HebbianConf<float> conf = {0.25, 0, 0};
	cpu_one.Setup(conf);
	cpu_one.SetTrainingSet(input);

	bool b = false;
	float f;
	errors = cpu_one.TrainFromData(500, 0, b, f);
	std::cout<< cpu_one <<std::endl;

	cpu_one.ExpToFS("foo.bar");
	ANN::BPNet<float, ANN::fcn_log<float>> cpu_two;
	cpu_two.ImpFromFS("foo.bar");
	cpu_two.SetTrainingSet(input);
	
	std::cout<< cpu_two <<std::endl;
	return 0;
}
