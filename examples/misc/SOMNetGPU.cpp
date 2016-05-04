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

#include "QSOMReader.h"
#include "Samples.h"

#include <ctime>
#include <iostream>


int main(int argc, char *argv[]) {
	QApplication a(argc, argv);

	ANN::TrainingSet input;
	input.AddInput(red);
	input.AddInput(green);
	input.AddInput(dk_green);
	input.AddInput(blue);
	input.AddInput(dk_blue);
	input.AddInput(yellow);
	input.AddInput(orange);
	input.AddInput(purple);
	input.AddInput(black);
	input.AddInput(white);

	std::vector<float> vCol(3);
	int w1 = 64;
	int w2 = 4;

	ANNGPGPU::SOMNetGPU gpu;
	gpu.CreateSOM(3, 1, w1,w1);
	gpu.SetTrainingSet(input);
	gpu.SetDistFunction(&ANN::Functions::fcn_gaussian);

	gpu.Training(500);

	SOMReader w(w1, w1, w2);
	for(int x = 0; x < w1*w1; x++) {
		ANN::SOMNeuron *pNeur = (ANN::SOMNeuron*)((ANN::SOMLayer*)gpu.GetOPLayer())->GetNeuron(x);
		vCol[0] = pNeur->GetConI(0)->GetValue();
		vCol[1] = pNeur->GetConI(1)->GetValue();
		vCol[2] = pNeur->GetConI(2)->GetValue();

		w.SetField(QPoint(pNeur->GetPosition()[0], pNeur->GetPosition()[1]), vCol );
	}
	w.Save("ColorsByGPU.png");

	return 0;
}
