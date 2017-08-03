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
#include <iostream>
#include <cassert>

#include "math/Random.h"
#endif

namespace ANN {

template<class T> class AbsNeuron;

/**
 * @class Edge
 * @brief Represents an edge in the network. Always connecting two neurons.
 * Each egde connects two neurons and has a specific value which might gets changed by connected neurons.
 */
template <class Type>
class Edge {
private:
	Type m_fWeight = GetRandReal(-0.5f, 0.5f);
	Type m_fMomentum = static_cast<Type>(0);

	AbsNeuron<Type> *m_pNeuronFirst = nullptr;
	AbsNeuron<Type> *m_pNeuronSecond = nullptr;

	bool m_bAllowAdaptation = true;

public:
	/**
	 * @brief Creating a "weight" connecting two neurons.
	 * @param pFirst Pointer directing to neuron number one.
	 * @param pSecond Pointer directing to neuron number two.
	 * @param fValue Value of the new Edge.
	 * @param fMomentum Value of the current momentum of the edge.
	 * @param bAdapt Allows to change the weight.
	 */
	Edge(AbsNeuron<Type> *pFirst = nullptr, 
	     AbsNeuron<Type> *pSecond = nullptr, 
	     Type fValue = static_cast<Type>(0), 
	     Type fMomentum = static_cast<Type>(0), 
	     bool bAdapt = true);

	/**
	 * @brief Looking from neuron pSource. Is returning a pointer to the other neuron.
	 * @param pSource Pointer to a neuron building this edge.
	 * @return Returns a pointer to the neuron != pSource.
	 */
	AbsNeuron<Type> *GetDestination(AbsNeuron<Type> *pSource) const;
	/**
	 * @brief Looking from neuron pSource. Is returning an index to the other neuron.
	 * @param pSource Pointer to a neuron building this edge.
	 * @return Returns the index of the neuron != pSource.
	 */
	int GetDestinationID(AbsNeuron<Type> *pSource) const;

	/**
	 * @brief Value of this edge.
	 * @return Returns the value of this edge.
	 */
	const Type &GetValue() const;
	/**
	 * @brief Sets the value of this edge.
	 * @param fValue New value of this edge.
	 */
	void SetValue(Type fValue);

	/**
	 * @brief Momentum of this edge.
	 * @return Returns the momentum of this edge.
	 */
	const Type &GetMomentum() const;
	/**
	 * @brief Sets the momentum of this edge.
	 * @param fValue New momentum of this edge.
	 */
	void SetMomentum(Type fValue);

	/**
	 * @brief Returns whether weight is changeable or not.
	 * @return Indicates whether the connection is changeable
	 */
	bool GetAdaptationState() const;
	/**
	 * @brief Switch the state whether changeable or not.
	 * @param bAdaptState indicates whether the connection is changeable
	 */
	void SetAdaptationState(const bool &bAdaptState);

	/*
	 * Returns the value of the edge.
	 */
	operator Type() const;
	
#ifdef __Edge_ADDON
	#include __Edge_ADDON
#endif
};

#include "Edge.tpp"

};

