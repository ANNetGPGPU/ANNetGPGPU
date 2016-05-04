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

#include <ctime>
#include <iostream>


//////////////////////////////////////////////////////////////////////////////////////////////
#ifdef __CUDACC__
	__host__ __device__
#endif
inline static float
fcn_nearest_nhood (float sigma0, float T, float lambda) {
	return sqrt(2.f);
}
//////////////////////////////////////////////////////////////////////////////////////////////

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
	int w1 = 40;
	int w2 = 4;

	ANN::SOMNet cpu;
	cpu.CreateSOM(3, 1, w1,w1);
	cpu.SetTrainingSet(input);
	cpu.SetConscienceRate(0.1);
	
	ANN::DistFunction distFn = ANN::Functions::fcn_gaussian;
	distFn.rad_decay = fcn_nearest_nhood;
	cpu.SetDistFunction(distFn);

	cpu.Training(1000);

	SOMReader w(w1, w1, w2);
	for(int x = 0; x < w1*w1; x++) {
		ANN::SOMNeuron *pNeur = (ANN::SOMNeuron*)((ANN::SOMLayer*)cpu.GetOPLayer())->GetNeuron(x);
		vCol[0] = pNeur->GetConI(0)->GetValue();
		vCol[1] = pNeur->GetConI(1)->GetValue();
		vCol[2] = pNeur->GetConI(2)->GetValue();

		w.SetField(QPoint(pNeur->GetPosition()[0], pNeur->GetPosition()[1]), vCol );
	}
	w.Save("SimpFnExtByCPU.png");
	return 0;
}
