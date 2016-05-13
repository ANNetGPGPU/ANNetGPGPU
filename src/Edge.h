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

#include <iostream>
#include <cassert>


namespace ANN {

template<class T> class AbsNeuron;

/**
 * @class Edge
 * @brief Represents an edge in the network. Always connecting two neurons.
 * Each egde connects two neurons and has a specific value which might gets changed by connected neurons.
 * @author Daniel "dgrat" Frenzel
 */
template <class Type>
class Edge {
private:
	Type m_fWeight;
	Type m_fMomentum;

	AbsNeuron<Type> *m_pNeuronFirst;
	AbsNeuron<Type> *m_pNeuronSecond;

	bool m_bAllowAdaptation;

public:
	/**
	 * @brief Creating a "weight" connecting two neurons.
	 */
	Edge();

	/**
	 * @brief Creating a "weight" with the properties of: *pEdge.
	 * This constructor can't copy connections (edges), because they normally have dependencies to other neurons.
	 */
	Edge(Edge<Type> *pEdge);
	/**
	 * @brief Creating a "weight" connecting two neurons.
	 * @param pFirst Pointer directing to neuron number one.
	 * @param pSecond Pointer directing to neuron number two.
	 */
	Edge(AbsNeuron<Type> *pFirst, AbsNeuron<Type> *pSecond);
	/**
	 * @brief Creating a "weight" connecting two neurons.
	 * @param pFirst Pointer directing to neuron number one.
	 * @param pSecond Pointer directing to neuron number two.
	 * @param fValue Value of the new Edge.
	 * @param fMomentum Value of the current momentum of the edge.
	 * @param bAdapt Allows to change the weight.
	 */
	Edge(AbsNeuron<Type> *pFirst, AbsNeuron<Type> *pSecond, Type fValue, Type fMomentum, bool bAdapt);

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
};

}

