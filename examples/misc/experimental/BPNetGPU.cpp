/*
 * main.cpp
 *
 *  Created on: 12.04.2010
 *      Author: dgrat
 */

#include <ANNet>
#include <ANGPGPU>
#include <ANContainers>
#include <ANMath>

#include "Samples.h"

#include <ctime>
#include <iostream>


int main(int argc, char *argv[]) {
	ANNGPGPU::BPNetGPU gpu;
	ANN::BPNet cpu;
	
	ANN::BPLayer layer1(3, ANN::ANLayerInput);
	layer1.AddFlag(ANN::ANBiasNeuron);
	ANN::BPLayer layer2(2048, ANN::ANLayerHidden);
	layer2.AddFlag(ANN::ANBiasNeuron);
	ANN::BPLayer layer3(2048, ANN::ANLayerHidden);
	layer3.AddFlag(ANN::ANBiasNeuron);
	ANN::BPLayer layer4(6, ANN::ANLayerOutput);
	layer4.AddFlag(ANN::ANBiasNeuron);

	layer1.ConnectLayer(&layer2);
	layer2.ConnectLayer(&layer3);
	layer3.ConnectLayer(&layer4);

	gpu.AddLayer(&layer1);
	gpu.AddLayer(&layer2);
	gpu.AddLayer(&layer3);
	gpu.AddLayer(&layer4);

	cpu.AddLayer(&layer1);
	cpu.AddLayer(&layer2);
	cpu.AddLayer(&layer3);
	cpu.AddLayer(&layer4);

	ANN::TrainingSet input;
	input.AddInput(fInp1, 3);
	input.AddOutput(fOut1, 6);
	input.AddInput(fInp2, 3);
	input.AddOutput(fOut2, 6);
	input.AddInput(fInp3, 3);
	input.AddOutput(fOut3, 6);
	input.AddInput(fInp4, 3);
	input.AddOutput(fOut4, 6);

	gpu.SetLearningRate(0.05);
	gpu.SetMomentum(0.1);
	gpu.SetWeightDecay(0);
	gpu.SetTrainingSet(input);


	cpu.SetLearningRate(0.05);
	cpu.SetMomentum(0.1);
	cpu.SetWeightDecay(0);
	cpu.SetTrainingSet(input);

	bool b = false;
	float f;
	std::vector<float> errors;
	
      /*
	errors = cpu.TrainFromData(10, 0, b, f);
	cpu.ExpToFS(>Foo.bar>);
	std::cout<<>cpu: \n><<cpu<<std::endl;
      */

	errors = gpu.TrainFromData(100, 0.01, b, f);
	gpu.ExpToFS("foo.bar");
	std::cout<<"gpu: \n"<<gpu<<std::endl;

	return 0;
}
