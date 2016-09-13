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
#include "containers/TrainingSet.h"
#include "containers/ConTable.h"
#include "math/Random.h"

#include <iostream>
#include <vector>
#include <string>
#include <bzlib.h>
#endif

namespace ANN {

template<class T> class ConTable;
template<class T> class Edge;
template<class T> class AbsLayer;
template<class T> class AbsNeuron;

/**
 * @class AbsNeuron
 * @brief Abstract class describing a basic neuron in a network.
 * Pure virtual functions must get implemented if deriving from this class.
 * These functions a doing the back-/propagation jobs.
 * You can modify the behavior of the complete net by overloading them.
 * @author Daniel "dgrat" Frenzel
 */
template <class Type>
class AbsNeuron {
protected:
	std::vector<Type> m_vPosition;			// x, y, z, .. coordinates of the neuron (e.g. SOM)
	Type m_fValue;					// value of the neuron in the net
	Type m_fErrorDelta;				// Current error delta of this neuron
	AbsLayer<Type> *m_pParentLayer;			// layer which is inheriting this neuron
	int m_iNeuronID;				// ID of this neuron in the layer

	std::vector<Edge<Type>*> m_lOutgoingConnections;
	std::vector<Edge<Type>*> m_lIncomingConnections;

public:
	AbsNeuron();
	
	/**
	 * @brief Creates a new neuron with parent layer: *pParentLayer
	 */
	AbsNeuron(AbsLayer<Type> *pParentLayer);
	/**
	 * @brief Copy constructor for creation of a new neuron with the "same" properties like *pNeuron. 
	 * This constructor can't copy connections (edges), because they normally have dependencies to other neurons.
	 * @param pNeuron object to copy properties from
	 */
	AbsNeuron(const AbsNeuron<Type> *pNeuron);
	virtual ~AbsNeuron();
	
	/**
	 * @brief Calls FreeMemory() and clears the lists
	 */
	virtual void EraseAll();

	/**
	 * @brief Pointer to the layer inherting this neuron.
	 */
	AbsLayer<Type> *GetParent() const;

	/**
	 * @brief Appends an edge to the list of incoming edges.
	 */
	virtual void AddConI(Edge<Type> *edge);
	/**
	 * @brief Appends an edge to the list of outgoing edges.
	 */
	virtual void AddConO(Edge<Type> *edge);

	virtual void SetConO(Edge<Type> *edge, const unsigned int iID);
	virtual void SetConI(Edge<Type> *edge, const unsigned int iID);

	/**
	 * @brief Returns a pointer to an incoming weight
	 * @return Pointer to an incoming edge
	 * @param iID Index of edge in m_lIncomingConnections
	 */
	virtual Edge<Type>* GetConI(const unsigned int &iID) const;
	/**
	 * @brief Returns a pointer to an outgoing weight
	 * @return Pointer to an outgoing edge
	 * @param iID Index of edge in m_lOutgoingConnections
	 */
	virtual Edge<Type>* GetConO(const unsigned int &iID) const;
	/**
	 * @brief Returns all incoming weights
	 * @return Array of pointers of all incoming edges
	 */
	virtual std::vector<Edge<Type> *> GetConsI() const;
	/**
	 * @brief Returns all outgoing weights
	 * @return Array of pointers of all outgoing edges
	 */
	virtual std::vector<Edge<Type> *> GetConsO() const;
	/**
	 * @brief Sets the index of this neuron to a certain value.
	 * @param iID New index of this neuron.
	 */
	virtual void SetID(const int iID);
	/**
	 * @brief Returns the index of this neuron.
	 * @return Index of this neuron.
	 */
	virtual unsigned int GetID() const;
	/**
	 * @brief Set the neuron to a certain value.
	 * @param fValue New value of this neuron.
	 */
	virtual void SetValue(const Type &fValue);
	/**
	 * @brief Returns the value of this neuron.
	 * @return Returns the value of this neuron.
	 */
	virtual const Type &GetValue() const;
	/**
	 * @brief Returns the postion coordinates of this neuron
	 * @return x, y, z, .. coordinates of the neuron (e.g. SOM)
	 */
	virtual const std::vector<Type> GetPosition() const;
	/**
	 * @brief Sets the current position of the neuron in the net.
	 * @param vPos Vector with Cartesian coordinates
	 */
	virtual void SetPosition(const std::vector<Type> &vPos);
	/**
	 * @brief Sets the error delta of this neuron to a certain value.
	 * @param fValue New error delts of this neuron.
	 */
	virtual void SetErrorDelta(const Type &fValue);
	/**
	 * @brief Returns the current error delta of this neuron.
	 * @return Returns the error delta of this neuron.
	 */
	virtual const Type &GetErrorDelta() const;

	/**
	 * @brief Overload to define how the net has to act while propagating back.
	 * I.e. how to modify the edges after calculating the error deltas.
	 */
	virtual void AdaptEdges() 	= 0;
	/**
	 * @brief Overload to define how the net has to act while propagating.
	 * I.e. which neurons/edges to use for calculating the new value of the neuron
	 */
	virtual void CalcValue() 	= 0;

	/**
	 * @brief Save neuron's content to filesystem
	 */
	virtual void ExpToFS(BZFILE* bz2out, int iBZ2Error);
	/**
	 * @brief Load neuron's content to filesystem
	 * @return The connections table of this neuron.
	 */
	virtual void ImpFromFS(BZFILE* bz2in, int iBZ2Error, ConTable<Type> &Table);

	/*
	 * Returns the value of the neuron.
	 */
	operator Type() const;
	
#ifdef __AbsNeuron_ADDON
	#include __AbsNeuron_ADDON
#endif
};

#include "AbsNeuron.tpp"
}

template <class T>
std::ostream& operator << (std::ostream &os, ANN::AbsNeuron<T> *op)
{
	os << "Value: \t" << op->GetValue() << std::endl;
	return os;     // Ref. auf Stream
}
