/*
 * Edge.cpp
 *
 *  Created on: 30.05.2009
 *      Author: Xerces
 */
#include "AbsNeuron.h"
#include "Edge.h"
#include "Common.h"
#include "math/Random.h"

using namespace ANN;


template <class Type>
Edge<Type>::Edge() {
	Type 	fValue 		= 0.f;
	Type 	fMomentum 	= 0.f;
	bool 	bAdapt 		= true;
}

template <class Type>
Edge<Type>::Edge(Edge<Type> *pEdge) {
	assert(pEdge);

	Type 	fValue 		= pEdge->GetValue();
	Type 	fMomentum 	= pEdge->GetMomentum();
	bool 	bAdapt 		= pEdge->GetAdaptationState();

	this->SetValue(fValue);
	this->SetMomentum(fMomentum);
	this->SetAdaptationState(bAdapt);

}

template <class Type>
Edge<Type>::Edge(AbsNeuron<Type> *first, AbsNeuron<Type> *second) {
	assert(first);
	assert(second);

	m_pNeuronFirst 		= first;
	m_pNeuronSecond 	= second;

	m_fWeight 		= RandFloat(-0.5f, 0.5f);
	m_bAllowAdaptation 	= true;
	m_fMomentum 		= 0.f;
}

template <class Type>
Edge<Type>::Edge(AbsNeuron<Type> *first, AbsNeuron<Type> *second, Type fValue, Type fMomentum, bool bAdapt) {
	assert(first);
	assert(second);

	m_pNeuronFirst 		= first;
	m_pNeuronSecond 	= second;

	m_fWeight 			= fValue;
	m_bAllowAdaptation 	= bAdapt;
	m_fMomentum 		= fMomentum;
}

template <class Type>
AbsNeuron<Type> *Edge<Type>::GetDestination(AbsNeuron<Type> *source) const {
	assert(source);

	if(m_pNeuronFirst != source) {
		return m_pNeuronFirst;
	}
	else if(m_pNeuronSecond != source) {
		return m_pNeuronSecond;
	}
	else if(m_pNeuronFirst != source && m_pNeuronSecond != source) {
		std::cout<<"error: neuron does not belong to this chain"<<std::endl;
		return NULL;
	}
	else {
		std::cout<<"error: edge contains two identical neurons"<<std::endl;
		return NULL;
	}
}

template <class Type>
int Edge<Type>::GetDestinationID(AbsNeuron<Type> *pSource) const {
	return this->GetDestination(pSource)->GetID();
}

template <class Type>
const Type &Edge<Type>::GetValue() const {
	return m_fWeight;
}

template <class Type>
void Edge<Type>::SetValue(Type fValue) {
	m_fWeight = fValue;
}

template <class Type>
bool Edge<Type>::GetAdaptationState() const {
	return m_bAllowAdaptation;
}

template <class Type>
void Edge<Type>::SetAdaptationState(const bool &adapt)	{
	m_bAllowAdaptation = adapt;
}

template <class Type>
const Type &Edge<Type>::GetMomentum() const {
	return m_fMomentum;
}

template <class Type>
void Edge<Type>::SetMomentum(Type fValue) {
	m_fMomentum = fValue;
}

template <class Type>
Edge<Type>::operator Type() const {
	return this->GetValue();
}


template class Edge<float>;
template class Edge<double>;
template class Edge<long double>;
template class Edge<short>;
template class Edge<int>;
template class Edge<long>;
template class Edge<long long>;

