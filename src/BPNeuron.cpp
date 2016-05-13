/*
 * BPNeuron.cpp
 *
 *  Created on: 30.05.2009
 *      Author: dgrat
 */
#include <iostream>
#include <cassert>

#include "AbsLayer.h"
#include "BPNeuron.h"
#include "Edge.h"
#include "math/Functions.h"

using namespace ANN;


template <class Type>
BPNeuron<Type>::BPNeuron() {
	m_fLearningRate = 0.01f;
	m_fWeightDecay 	= 0.f;
	m_fMomentum 	= 0.f;

	// a stdard sigmoid function for the network
	this->SetTransfFunction(&Functions::fcn_log);
}

template <class Type>
BPNeuron<Type>::BPNeuron(AbsLayer<Type> *parentLayer) : AbsNeuron<Type>(parentLayer) {
	m_fLearningRate = 0.01f;
	m_fWeightDecay 	= 0.f;
	m_fMomentum 	= 0.f;

	// a stdard sigmoid function for the network
	this->SetTransfFunction(&Functions::fcn_log);
}

template <class Type>
BPNeuron<Type>::BPNeuron(BPNeuron *pNeuron) : AbsNeuron<Type>(pNeuron) {
}

template <class Type>
BPNeuron<Type>::~BPNeuron() {
}

template <class Type>
void BPNeuron<Type>::SetLearningRate(const Type &fVal) {
	m_fLearningRate = fVal;
}

template <class Type>
void BPNeuron<Type>::SetWeightDecay(const Type &fVal) {
	m_fWeightDecay = fVal;
}

template <class Type>
void BPNeuron<Type>::SetMomentum(const Type &fVal) {
	m_fMomentum = fVal;
}

template <class Type>
void BPNeuron<Type>::CalcValue() {
	if(this->GetConsI().size() == 0)
		return;

	// bias neuron/term
	Type fBias = 0.f;
	this->SetValue( 0.f );
	if(this->GetBiasEdge() ) {
		fBias = this->GetBiasEdge()->GetValue();
		this->SetValue(-1.f*fBias);
	}

	// sum from product of all incoming neurons with their weights (including bias neurons)
	AbsNeuron<Type> *from;
	for(unsigned int i = 0; i < this->GetConsI().size(); i++) {
		from = this->GetConI(i)->GetDestination(this);
		this->SetValue(this->GetValue() + (from->GetValue() * this->GetConI(i)->GetValue()));
	}

	Type fVal = this->GetTransfFunction()->normal( this->GetValue(), fBias );
	this->SetValue(fVal);
}

template <class Type>
void BPNeuron<Type>::AdaptEdges() {
	if(this->GetConsO().size() == 0)
		return;

	AbsNeuron<Type> *pCurNeuron;
	Edge<Type> 	*pCurEdge;
	Type 		fVal;

	// calc error deltas
	fVal = this->GetErrorDelta();
	for(unsigned int i = 0; i < this->GetConsO().size(); i++) {
		pCurEdge 	= this->GetConO(i);
		pCurNeuron 	= pCurEdge->GetDestination(this);
		fVal += pCurNeuron->GetErrorDelta() * pCurEdge->GetValue();
	}
	fVal *= this->GetTransfFunction()->derivate( this->GetValue(), 0.f );
	this->SetErrorDelta(fVal);

	// adapt weights
	for(unsigned int i = 0; i < this->GetConsO().size(); i++) {
		pCurEdge = this->GetConO(i);
		if(pCurEdge->GetAdaptationState() == true) {
			//fVal = 0.f;	// delta for momentum
			// standard back propagation algorithm
			fVal = pCurEdge->GetDestination(this)->GetErrorDelta() * m_fLearningRate * this->GetValue()
			// weight decay term
			- m_fWeightDecay * pCurEdge->GetValue()
			// momentum term
			+ m_fMomentum * pCurEdge->GetMomentum();

			pCurEdge->SetMomentum( fVal );
			pCurEdge->SetValue( fVal+pCurEdge->GetValue() );
		}
	}
}


template class BPNeuron<float>;
template class BPNeuron<double>;
template class BPNeuron<long double>;
template class BPNeuron<short>;
template class BPNeuron<int>;
template class BPNeuron<long>;
template class BPNeuron<long long>;

