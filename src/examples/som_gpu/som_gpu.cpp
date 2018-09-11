/*
 * main.cpp
 *
 *  Created on: 12.04.2010
 *      Author: dgrat
 */

#include <iomanip>
#include <map>

#include <ANNet>
#include <ANGPGPU>
#include <ANContainers>
#include <ANMath>
#include <CentroidGPU.h>

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
	int w1 = 60;
	int w2 = 4;

	ANNGPGPU::SOMNetGPU<float, ANN::functor_gaussian<float>> gpu;
	gpu.CreateSOM(3, 1, w1,w1);
	gpu.SetTrainingSet(input);
/*
        // Clear initial weights
        for(int x = 0; x < w1*w1; x++) {
            ANN::SOMNeuron<float> *pNeur = (ANN::SOMNeuron<float>*)((ANN::SOMLayer<float>*)gpu.GetOPLayer())->GetNeuron(x);
            pNeur->GetConI(0)->SetValue(0); 
            pNeur->GetConI(1)->SetValue(0); 
            pNeur->GetConI(2)->SetValue(0); 
            // Except for one unit.
            if (x == 820) {
                pNeur->GetConI(0)->SetValue(1); 
                pNeur->GetConI(1)->SetValue(1); 
                pNeur->GetConI(2)->SetValue(1); 
            }
        }
*/
	gpu.Training(100);

	SOMReader w(w1, w1, w2);
	for(int x = 0; x < w1*w1; x++) {
		ANN::SOMNeuron<float> *pNeur = (ANN::SOMNeuron<float>*)((ANN::SOMLayer<float>*)gpu.GetOPLayer())->GetNeuron(x);
		vCol[0] = pNeur->GetConI(0)->GetValue();
		vCol[1] = pNeur->GetConI(1)->GetValue();
		vCol[2] = pNeur->GetConI(2)->GetValue();

		w.SetField(QPoint(pNeur->GetPosition()[0], pNeur->GetPosition()[1]), vCol );
	}
	ANN::printf("test\n");
	
	w.Save("ColorsByGPU.png");
	
	ANN::printf("Get centroids\n");
	
	std::vector<ANNGPGPU::Centroid<float>> cents = gpu.FindCentroidsGPU();
	
	ANN::printf("size_of cents %d\n", cents.size());

	for(auto centr : cents) {
		std::cout << centr << std::endl;
	}

	return 0;
}
