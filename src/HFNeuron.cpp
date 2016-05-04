/*
 * HFNeuron.cpp
 *
 *  Created on: 22.02.2011
 *      Author: dgrat
 */

#include <iostream>

#include "HFNeuron.h"
#include "Edge.h"
#include "math/Functions.h"

using namespace ANN;


HFNeuron::HFNeuron(AbsLayer *parentLayer) : AbsNeuron(parentLayer) {
	SetTransfFunction(&Functions::fcn_binary);
	SetValue(1);
}

HFNeuron::~HFNeuron() {
	// TODO Auto-generated destructor stub
}

void HFNeuron::CalcValue() {
	if(GetConsI().size() == 0)
		return;

	/*
	 * Calculate the activation of this neuron by input value d weights of the other neurons
	 */
	HFNeuron *from 	= NULL;
	float fVal 			= 0.f;
	for(unsigned int i = 0; i < GetConsI().size(); i++) {
		from = (HFNeuron *)GetConI(i)->GetDestination(this);
		fVal += from->GetValue() * GetConI(i)->GetValue();
	}

	fVal = GetTransfFunction()->normal( fVal, 0 );
	SetValue(fVal);
}

void HFNeuron::AdaptEdges() {
	/**
	 * i have to do nothing in this hopfield net,
	 * because with CalculateMatrix() it is possible to calculate the exact weight matrix,
	 * but it would be possible to implement the "hebbsche lernregel" here too.
	 */
}

