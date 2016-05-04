/*
 * SOMNeuron.cpp
 *
 *  Created on: 25.02.2012
 *      Author: dgrat
 */

#include "SOMNeuron.h"
#include "SOMLayer.h"
#include "Edge.h"
#include "math/Functions.h"
#include "math/Random.h"
#include <cmath>
#include <cassert>


namespace ANN {

SOMNeuron::SOMNeuron(SOMLayer *parent) : AbsNeuron(parent) {
	m_fLearningRate = 0.5f;
	m_fConscience = 0.f;

	// a standard sigmoid transfer function for the network
	SetTransfFunction(&Functions::fcn_log);

	// gives neuron random coordinates
	for(unsigned int i = 0; i < m_vPosition.size(); i++) {
		int iMax = parent->GetDim(i) * 10;
		m_vPosition[i] = RandInt(0, iMax);
	}
}

SOMNeuron::~SOMNeuron() {
	// TODO Auto-generated destructor stub
}

void SOMNeuron::AdaptEdges() {
	Edge 	*pEdge 	= NULL;
	float 	fInput 	= 0.f;
	float 	fWeight = 0.f;

    for(unsigned int i = 0; i < GetConsI().size(); i++) {
    	pEdge 	= GetConI(i);
    	fWeight = *pEdge;
    	fInput 	= *pEdge->GetDestination(this);

    	pEdge->SetValue(fWeight + (m_fInfluence*m_fLearningRate*(fInput-fWeight) ) );
    }
}

void SOMNeuron::CalcValue() {
	// TODO
}

void SOMNeuron::CalcDistance2Inp() {
	m_fValue = 0.f;
	for (unsigned int i=0; i < GetConsI().size(); ++i) {
		m_fValue += pow(*GetConI(i)->GetDestination(this) - *GetConI(i), 2);	// both have a float() operator!
	}
	//m_fValue = sqrt(fDist);
}

float SOMNeuron::GetLearningRate() const {
	return m_fLearningRate;
}

void SOMNeuron::SetLearningRate(const float &fVal) {
	m_fLearningRate = fVal;
}

float SOMNeuron::GetInfluence() const {
	return m_fInfluence;
}

void SOMNeuron::SetInfluence(const float &fVal) {
	m_fInfluence = fVal;
}

float SOMNeuron::GetDistance2Neur(const SOMNeuron &pNeurDst) {
	assert(this->GetPosition().size() == pNeurDst.GetPosition().size() );

	float fDist = 0.f;
	for(unsigned int i = 0; i < this->GetPosition().size(); i++) {
		fDist += pow(pNeurDst.GetPosition().at(i) - this->GetPosition().at(i), 2);
	}
	return sqrt(fDist);
}

void SOMNeuron::SetConscience(float &fVal) {
	m_fConscience = fVal;
}

void SOMNeuron::AddConscience(float &fVal) {
	m_fConscience += fVal;
}

float SOMNeuron::GetConscience() {
	return m_fConscience;
}

/*
 * friends
 */
float GetDistance2Neur(const SOMNeuron &pNeurSrc, const SOMNeuron &pNeurDst) {
	assert(pNeurSrc.GetPosition().size() == pNeurDst.GetPosition().size() );

	float fDist = 0.f;
	for(unsigned int i = 0; i < pNeurSrc.GetPosition().size(); i++) {
		fDist += pow(pNeurDst.GetPosition().at(i) - pNeurSrc.GetPosition().at(i), 2);
	}
	//std::cout<<"CPU distance: "<< sqrt(fDist) <<std::endl;
	return sqrt(fDist);
}

}
