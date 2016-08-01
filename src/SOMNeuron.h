/*
#-------------------------------------------------------------------------------
# Copyright (c) 2012 Daniel <dgrat> Frenzel.
# All rights reserved. This program and the accompanying materials
# are made available under the terms of the GNU Lesser Public License v2.1
# which accompanies this distribution, and is available at
# http://www.gnu.org/licenses/old-licenses/gpl-2.0.html
# 
# Contributors:
#     Daniel <dgrat> Frenzel - initial API and implementation
#-------------------------------------------------------------------------------
*/
#pragma once

#ifndef SWIG
#include "AbsNeuron.h"
#include "math/Random.h"
#include "containers/ConTable.h"

#include <cassert>
#include <vector>
#endif


namespace ANN {

template <class T> class SOMLayer;

/**
 * @class SOMNeuron
 * @brief Implementation of a neuron in a self organizing map.
 */
template <class Type>
class SOMNeuron : public AbsNeuron<Type> {
protected:
	Type 	m_fLearningRate; 	// learning rate
	Type 	m_fInfluence; 		// distance of neurons in neighborhood to alterate
	Type 	m_fConscience; 		// bias for conscience mechanism
	Type	m_fSigma0;		// inital distance bias to get activated
	Type	m_fSigmaT;		// current epoch dependent distance bias

public:
	SOMNeuron(SOMLayer<Type> *parent = 0);
	virtual ~SOMNeuron();

	/**
	 * @brief Save neuron's content to filesystem
	 */
	virtual void ExpToFS(BZFILE* bz2out, int iBZ2Error);
	/**
	 * @brief Load neuron's content to filesystem
	 * @return The connections table of this neuron.
	 */
	virtual void ImpFromFS(BZFILE* bz2in, int iBZ2Error, ConTable<Type> &Table);
	
	/**
	 * @brief Overload to define how the net has to act while propagating back. I. e. how to modify the edges after calculating the error deltas.
	 */
	virtual void AdaptEdges();

	/**
	 * @brief Defines the starting activation distance. 
	 * Sets the initial Sigma0 value. This distance to a BMU determines whether a neuron can be influenced during a training step. 
	 * During training this distance shrinks and Sigma0 is just the starting value.
	 */
	 void SetSigma0(const Type &fVal);
	 
	/**
	 * @brief Defines the starting activation distance. 
	 * Sets the initial Sigma0 value. This distance to a BMU determines whether a neuron can be influenced during a training step. 
	 * During training this distance shrinks and Sigma0 is just the starting value.
	 */
	 Type GetSigma0();
	 
	/**
	 * @brief Calculates the value of the neuron
	 */
	virtual void CalcValue();

	/**
	 * @brief Calculates the distance of the neuron to the input vector
	 */
	virtual void CalcDistance2Inp();

	/**
	 * @brief Gets the current learning rate of the network.
	 * @return Returns the current learning rate
	 */
	Type GetLearningRate() const;

	/**
	 * @brief Sets the learning rate
	 * @param fVal Current learning rate
	 */
	void SetLearningRate(const Type &fVal);

	/**
	 * @brief Gets the current influence based on the neighborhood and training function.
	 * @return Returns the current influence
	 */
	Type GetInfluence() const;

	/**
	 * @brief Sets the learning rate for training process.
	 * @param fVal Current influence
	 */
	void SetInfluence(const Type &fVal);

	/**
	 * @brief Gets the euclidian distance between two neurons.
	 * @return Returns the current distance of the neuron to its input vector.
	 */
	Type GetDistance2Neur(const SOMNeuron<Type> &pNeurDst);
	
	/**
	 * @brief Sets the scalar for the conscience mechanism. If it is zero, then conscience is not applied.
	 */
	void SetConscience(const Type &fVal);
	
	/**
	 * @brief Adds the given value to the conscience scalar
	 */
	void AddConscience(const Type &fVal);
	
	/**
	 * @brief Returns the conscience scalar of the network. If it is zero, then conscience is not applied.
	 * @return Get the bias for the conscience mechanism
	 */
	Type GetConscience();
	
#ifdef __SOMNeuron_ADDON
	#include __SOMNeuron_ADDON
#endif
};

#include "SOMNeuron.tpp"

/**
 * @brief Gets the euclidian distance between two neurons.
 * @return Returns the current distance of the neuron to its input vector.
 */

template <typename T> T GetDistance2Neur(const SOMNeuron<T> &pNeurSrc, const SOMNeuron<T> &pNeurDst) {
	assert(pNeurSrc.GetPosition().size() == pNeurDst.GetPosition().size() );

	T fDist = 0.f;
	for(unsigned int i = 0; i < pNeurSrc.GetPosition().size(); i++) {
		fDist += pow(pNeurDst.GetPosition().at(i) - pNeurSrc.GetPosition().at(i), 2);
	}
	return sqrt(fDist);
}

}

