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


/*
 * Here we declare just the basic learning functions 
 */
template <class T>
inline T custom_learn(T fWeight, T fInfluence, T fInput) {
	return fWeight + (fInfluence*(fInput-fWeight) );
}

template <class T>
inline T custom_gaussian_nhood (T dist, T sigmaT) {
	return exp(-pow(dist, 2.f)/(2.f*pow(sigmaT, 2.f)));
}

template <class T>
inline T custom_rad_decay (T sigma0, T t, T lambda) {
	return std::floor(sigma0*exp(-t/lambda) + 0.5f);
}

template <class T>
inline T custom_lrate_decay (T sigma0, T t, T lambda) {
	return sigma0*exp(-t/lambda);
}

/*
 * Here we define the functor for the network 
 */
template<class T> using custom_functor = ANN::DistFunction<T, custom_learn<T>, custom_gaussian_nhood<T>, custom_rad_decay<T>, custom_lrate_decay<T> >;


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

	/*
	 * Here the compiler will generate our final code :D
	 */
	ANN::SOMNet<float, custom_functor<float>> cpu;
	cpu.CreateSOM(3, 1, w1,w1);
	cpu.SetTrainingSet(input);
	
	// Clear initial weights
	for(int x = 0; x < w1*w1; x++) {
		ANN::SOMNeuron<float> *pNeur = (ANN::SOMNeuron<float>*)((ANN::SOMLayer<float>*)cpu.GetOPLayer())->GetNeuron(x);
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
	
	cpu.Training(1);

	SOMReader w(w1, w1, w2);
	for(int x = 0; x < w1*w1; x++) {
		ANN::SOMNeuron<float> *pNeur = (ANN::SOMNeuron<float>*)((ANN::SOMLayer<float>*)cpu.GetOPLayer())->GetNeuron(x);
		vCol[0] = pNeur->GetConI(0)->GetValue();
		vCol[1] = pNeur->GetConI(1)->GetValue();
		vCol[2] = pNeur->GetConI(2)->GetValue();

		w.SetField(QPoint(pNeur->GetPosition()[0], pNeur->GetPosition()[1]), vCol );
	}
	w.Save("ColorsByGPU.png");

	return 0;
}
