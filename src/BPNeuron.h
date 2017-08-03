/// -*- tab-width: 8; Mode: C++; c-basic-offset: 8; indent-tabs-mode: t -*-
/*
   This program is free software: you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.
   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.
   You should have received a copy of the GNU General Public License
   along with this program.  If not, see <http://www.gnu.org/licenses/>.
   
   Author: Daniel Frenzel (dgdanielf@gmail.com)
*/

#pragma once

#ifndef SWIG
#include <vector>
#include "AbsNeuron.h"
#include "math/Functions.h"
#endif

namespace ANN {
template <class T> class HebbianConf;
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
template <class Type, class Functor>
class BPNeuron : public AbsNeuron<Type> {
private:
	Functor m_TransferFunction;
	HebbianConf<Type> m_Setup = { 0.1f, 0, 0 };

public:
	BPNeuron(AbsLayer<Type> *parentLayer = nullptr);

	/**
	 * Define the learning rate, the weight decay and the momentum term.
	 */
	void Setup(const HebbianConf<Type> &config);

	/**
	 * Defines how to calculate the values of each neuron.
	 */
	virtual void CalcValue();
	/**
	 * Defines how to calculate the error deltas of each neuron.
	 * Defines also how to change the weights.
	 */
	virtual void AdaptEdges();
	
#ifdef __BPNeuron_ADDON
	#include __BPNeuron_ADDON
#endif
};

#include "BPNeuron.tpp"

}
