/*
 * BPNeuron.cpp
 *
 *  Created on: 30.05.2009
 *      Author: dgrat
 */

#include <iostream>
#include <cassert>
//own classes
#include "math/Functions.h"
#include "Edge.h"
#include "AbsLayer.h"
#include "BPNeuron.h"

using namespace ANN;


BPNeuron::BPNeuron(AbsLayer *parentLayer) : AbsNeuron(parentLayer) {
	m_fLearningRate = 0.01f;
	m_fWeightDecay 	= 0.f;
	m_fMomentum 	= 0.f;

	// a stdard sigmoid function for the network
	SetTransfFunction(&Functions::fcn_log);
}

BPNeuron::BPNeuron(BPNeuron *pNeuron) : AbsNeuron(pNeuron) {
}

BPNeuron::~BPNeuron() {
}

void BPNeuron::SetLearningRate(const float &fVal) {
	m_fLearningRate = fVal;
}


void BPNeuron::SetWeightDecay(const float &fVal) {
	m_fWeightDecay = fVal;
}

void BPNeuron::SetMomentum(const float &fVal) {
	m_fMomentum = fVal;
}

void BPNeuron::CalcValue() {
	if(GetConsI().size() == 0)
		return;

	// bias neuron/term
	float fBias = 0.f;
	SetValue( 0.f );
	if(GetBiasEdge() ) {
		fBias = GetBiasEdge()->GetValue();
		SetValue(-1.f*fBias);
	}

	// sum from product of all incoming neurons with their weights (including bias neurons)
	AbsNeuron *from;
	for(unsigned int i = 0; i < GetConsI().size(); i++) {
		from = GetConI(i)->GetDestination(this);
		SetValue(GetValue() + (from->GetValue() * GetConI(i)->GetValue()));
	}

	float fVal = GetTransfFunction()->normal( GetValue(), fBias );
	SetValue(fVal);
}

void BPNeuron::AdaptEdges() {
	if(GetConsO().size() == 0)
		return;

	AbsNeuron 	*pCurNeuron;
	Edge 		*pCurEdge;
	float 		fVal;

	// calc error deltas
	fVal = GetErrorDelta();
	for(unsigned int i = 0; i < GetConsO().size(); i++) {
		pCurEdge 	= GetConO(i);
		pCurNeuron 	= pCurEdge->GetDestination(this);
		fVal += pCurNeuron->GetErrorDelta() * pCurEdge->GetValue();
	}
	fVal *= GetTransfFunction()->derivate( GetValue(), 0.f );
	SetErrorDelta(fVal);

	// adapt weights
	for(unsigned int i = 0; i < GetConsO().size(); i++) {
		pCurEdge = GetConO(i);
		if(pCurEdge->GetAdaptationState() == true) {
			//fVal = 0.f;	// delta for momentum
			// standard back propagation algorithm
			fVal = pCurEdge->GetDestination(this)->GetErrorDelta() * m_fLearningRate * GetValue()
			// weight decay term
			- m_fWeightDecay * pCurEdge->GetValue()
			// momentum term
			+ m_fMomentum * pCurEdge->GetMomentum();

			pCurEdge->SetMomentum( fVal );
			pCurEdge->SetValue( fVal+pCurEdge->GetValue() );
		}
	}
}

