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

#ifndef SOMNEURON_H_
#define SOMNEURON_H_

#ifndef SWIG
#include "AbsNeuron.h"

#include <vector>
#endif


namespace ANN {

class SOMLayer;


/**
 * @class SOMNeuron
 * @brief Implementation of a neuron in a self organizing map.
 */
class SOMNeuron : public AbsNeuron {
protected:
	float 	m_fLearningRate; 	// learning rate
	float 	m_fInfluence; 		// distance of neurons in neighborhood to alterate
	float 	m_fConscience; 		// bias for conscience mechanism
	float	m_fSigma0;		// inital distance bias to get activated
	float	m_fSigmaT;		// current epoch dependent distance bias

public:
	SOMNeuron(SOMLayer *parent = 0);
	virtual ~SOMNeuron();

	/**
	 * @brief Save neuron's content to filesystem
	 */
	virtual void ExpToFS(BZFILE* bz2out, int iBZ2Error);
	/**
	 * @brief Load neuron's content to filesystem
	 * @return The connections table of this neuron.
	 */
	virtual void ImpFromFS(BZFILE* bz2in, int iBZ2Error, ConTable &Table);
	
	/**
	 * @brief Overload to define how the net has to act while propagating back. I. e. how to modify the edges after calculating the error deltas.
	 */
	virtual void AdaptEdges();

	/**
	 * @brief Defines the starting activation distance. 
	 * Sets the initial Sigma0 value. This distance to a BMU determines whether a neuron can be influenced during a training step. 
	 * During training this distance shrinks and Sigma0 is just the starting value.
	 */
	 void SetSigma0(const float &fVal);
	 
	/**
	 * @brief Defines the starting activation distance. 
	 * Sets the initial Sigma0 value. This distance to a BMU determines whether a neuron can be influenced during a training step. 
	 * During training this distance shrinks and Sigma0 is just the starting value.
	 */
	 float GetSigma0();
	 
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
	float GetLearningRate() const;

	/**
	 * @brief Sets the learning rate
	 * @param fVal Current learning rate
	 */
	void SetLearningRate(const float &fVal);

	/**
	 * @brief Gets the current influence based on the neighborhood and training function.
	 * @return Returns the current influence
	 */
	float GetInfluence() const;

	/**
	 * @brief Sets the learning rate for training process.
	 * @param fVal Current influence
	 */
	void SetInfluence(const float &fVal);

	/**
	 * @brief Gets the euclidian distance between two neurons.
	 * @return Returns the current distance of the neuron to its input vector.
	 */
	float GetDistance2Neur(const SOMNeuron &pNeurDst);

	/**
	 * @brief Gets the euclidian distance between two neurons.
	 * @return Returns the current distance of the neuron to its input vector.
	 */
	friend float GetDistance2Neur(const SOMNeuron &pNeurSrc, const SOMNeuron &pNeurDst);
	
	/**
	 * @brief Sets the scalar for the conscience mechanism. If it is zero, then conscience is not applied.
	 */
	void SetConscience(const float &fVal);
	
	/**
	 * @brief Adds the given value to the conscience scalar
	 */
	void AddConscience(const float &fVal);
	
	/**
	 * @brief Returns the conscience scalar of the network. If it is zero, then conscience is not applied.
	 * @return Get the bias for the conscience mechanism
	 */
	float GetConscience();
};

}

#endif /* SOMNEURON_H_ */
