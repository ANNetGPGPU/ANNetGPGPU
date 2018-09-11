/*
 * main.cpp
 *
 *  Created on: 12.04.2010
 *      Author: dgrat
 */

#include <ANNet>
#include <ANContainers>
#include <ANMath>

#include "QSOMReader.h"
#include "Samples.h"


int main(int argc, char *argv[]) {
	QApplication a(argc, argv);

	ANN::TrainingSet<float> input;
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
	int w1 = 4;
	int w2 = 64;

	ANN::SOMNet<float, ANN::functor_gaussian<float>> cpu;
	cpu.CreateSOM(3, 1, w1,w1);
	cpu.SetTrainingSet(input);

	cpu.Training(1000);
	std::vector<ANN::Centroid<float>> vCen = cpu.FindCentroids();
	for(size_t i = 0; i < vCen.size(); i++) {
		std::cout << vCen.at(i)._unitID << ", "  << vCen.at(i)._distance << std::endl;
	}

	SOMReader w(w1, w1, w2);
	for(int x = 0; x < w1*w1; x++) {
	      ANN::SOMNeuron<float> *pNeur = (ANN::SOMNeuron<float>*)((ANN::SOMLayer<float>*)cpu.GetOPLayer())->GetNeuron(x);
	      vCol[0] = pNeur->GetConI(0)->GetValue();
	      vCol[1] = pNeur->GetConI(1)->GetValue();
	      vCol[2] = pNeur->GetConI(2)->GetValue();

	      w.SetField(QPoint(pNeur->GetPosition()[0], pNeur->GetPosition()[1]), vCol );
	}
	w.Save("ClustersByCPU.png");
	return 0;
}
