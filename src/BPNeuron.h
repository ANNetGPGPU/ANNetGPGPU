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
#include <vector>
#include "AbsNeuron.h"
#endif

namespace ANN {

template <class T> class Edge;
template <class T> class AbsLayer;

/**
 * \brief Derived from ANAbsNeuron. Represents a neuron in a network.
 *
 * Pure virtual functions from abstract base are already implemented here.
 * You can modify the behavior of the complete net by overloading them.
 *
 * @author Daniel "dgrat" Frenzel
 */
template <class Type>
class BPNeuron : public AbsNeuron<Type> {
private:
	Type m_fLearningRate;	// 0,0 - 0,5
	Type m_fWeightDecay;	// 0,005 - 0,03
	Type m_fMomentum;	// 0,5 - 0,9

public:
	BPNeuron();
	
	/*
	 * CTOR
	 * & DTOR
	 */
	BPNeuron(AbsLayer<Type> *parentLayer);
	/**
	 * Copy constructor for creation of a new neuron with the "same" properties like *pNeuron
	 * this constructor can't copy connections (edges), because they normally have dependencies to other neurons.
	 * @param pNeuron object to copy properties from
	 */
	BPNeuron(BPNeuron<Type> *pNeuron);
	~BPNeuron();

	/**
	 * Sets the scalar of the learning rate.
	 */
	void SetLearningRate(const Type &fVal);
	/**
	 * Sets the scalar of the weight decay.
	 */
	void SetWeightDecay(const Type &fVal);
	/**
	 * Sets the scalar of the momentum.
	 */
	void SetMomentum(const Type &fVal);

	/**
	 * Defines how to calculate the values of each neuron.
	 */
	virtual void CalcValue();
	/**
	 * Defines how to calculate the error deltas of each neuron.
	 * Defines also how to change the weights.
	 */
	virtual void AdaptEdges();
};

}

