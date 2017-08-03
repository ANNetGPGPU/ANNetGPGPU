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
#include "containers/2DArray.h"
#include "containers/ConTable.h"

#include <iostream>
#include <vector>
#include <stdint.h>
#include <bzlib.h>
#endif

namespace ANN {
template<class T> class ConTable;
template<class T> class AbsNeuron;
template<class T> class Edge;


enum {
	ANLayerInput 	= 1 << 0,	// type of layer
	ANLayerHidden 	= 1 << 1,	// type of layer
	ANLayerOutput 	= 1 << 2	// type of layer
};
typedef uint32_t LayerTypeFlag;

/**
 * @class AbsLayer
 * @brief Represents a container for neurons in the network.
 * @author Daniel "dgrat" Frenzel
 */
template <class Type>
class AbsLayer {
protected:
	/**
	 * @brief Array of pointers to all neurons in this layer.
	 */
	std::vector<AbsNeuron<Type> *> m_lNeurons;

	/**
	 * @brief Index of the layer
	 */
	int m_iID;

	/**
	 * @brief Flag describing the type of layer.
	 * (i. e. input, hidden or output possible)
	 */
	LayerTypeFlag m_fTypeFlag;

public:
	virtual ~AbsLayer();

	/**
	 * @brief Calls FreeMemory() and clears the lists
	 */
	virtual void EraseAll();
	
	/**
	 * @brief Sets the current ID in the Network inheriting the layer. Useful for administration purposes.
	 */
	virtual void SetID(const int &iID);
	/**
	 * @brief Returns the current ID in the Network inheriting the layer. Useful for administration purposes.
	 */
	virtual int GetID() const;

	/**
	 * @brief Pointer to the neuron at index iID.
	 * @return Returns the pointer of the neuron at index iID
	 * @param iID Index of the neuron in m_lNeurons
	 */
	virtual AbsNeuron<Type> *GetNeuron(const unsigned int &iID) const;
	/**
	 * @brief List of all neurons in this layer.
	 * @return Returns an array with pointers of neurons in this layer.
	 */
	virtual const std::vector<AbsNeuron<Type> *> &GetNeurons() const;

	/**
	 * @brief Adds neurons to the layer
	 * @param iSize Number of neurons to add.
	 */
	virtual void AddNeurons(const unsigned int &iSize) = 0;

	/**
	 * @brief Sets the type of the layer (input, hidden or output layer)
	 * @param fType Flag describing the type of the layer.
	 */
	virtual void SetFlag(const LayerTypeFlag &fType);
	/**
	 * @brief Adds a flag if not already set.
	 * @param fType Flag describing the type of the layer.
	 */
	virtual void AddFlag(const LayerTypeFlag &fType);
	/**
	 * @brief Type of the layer
	 * @return Returns the flag describing the type of the layer.
	 */
	virtual LayerTypeFlag GetFlag() const;

	/////////////////////////////////////////////////////////////////////////////////////////////////////

	/** 
	 * @brief Exports the layer to a plain array
	 * \n Matrix layout:
	 * \n NEURON1	 		: edge1, edge2, edge[n < iWidth] ==> directing to input neuron 1
	 * \n NEURON2 			: edge1, edge2, edge[n < iWidth] ==> directing to input neuron 2
	 * \n NEURON3	 		: edge1, edge2, edge[n < iWidth] ==> directing to input neuron 3
	 * \n NEURON[i < iHeight] 	: edge1, edge2, edge[n < iWidth] ==> directing to input neuron i
	 * @return Returns a matrix: width=size_of_this_layer; height=size_previous_layer
	 */
	virtual F2DArray<Type> ExpEdgesIn() const;

	/** 
	 * @brief Exports the layer to a plain array
	 * \n Matrix layout:
	 * \n NEURON[iStart]		: edge1, edge2, edge[n < iWidth] ==> directing to input neuron 1
	 * \n NEURON[iStart+1]		: edge1, edge2, edge[n < iWidth] ==> directing to input neuron 2
	 * \n NEURON[iStart+2]	 	: edge1, edge2, edge[n < iWidth] ==> directing to input neuron 3
	 * \n NEURON[iStart+n]	 	: edge1, edge2, edge[n < iWidth] ==> directing to input neuron n
	 * \n NEURON[iStop] 		: edge1, edge2, edge[n < iWidth] ==> directing to input neuron iStop
	 * @param iStart Start index for export
	 * @param iStart Stop index for export
	 * @return Returns a matrix: width=size_of_this_layer; height=iStop-iStart
	 */
	virtual F2DArray<Type> ExpEdgesIn(int iStart, int iStop) const;
	
	/**
	 * @brief Imports weight informations from a weight matrix and saves them to the incoming edges.
	 * @param f2dEdges Array which stores the weight informations
	 */
	virtual void ImpEdgesIn(const F2DArray<Type> &f2dEdges);
	
	/**
	 * @brief Imports weight informations from a plain array and saves them to the incoming edges.
	 * Import for iStop-iStart edges.
	 * @param iStart Start index for import
	 * @param iStart Stop index for import
	 */
	virtual void ImpEdgesIn(const F2DArray<Type> &, int iStart, int iStop);

	/////////////////////////////////////////////////////////////////////////////////////////////////////
	
	/** 
	 * @brief Exports the layer to a plain array
	 * \n Matrix layout:
	 * \n NEURON1			: edge1, edge1, edge[n < iWidth] ==> directing to next neuron 1, 2, n
	 * \n NEURON2			: edge2, edge2, edge[n < iWidth] ==> directing to next neuron 1, 2, n
	 * \n NEURON3			: edge3, edge3, edge[n < iWidth] ==> directing to next neuron 1, 2, n
	 * \n NEURON[i < iHeight] 	: edge4, edge4, edge[n < iWidth] ==> directing to next neuron 1, 2, n
	 * @return Returns a matrix: width=size_this_layer; height=size_of_next_layer
	 */
	virtual F2DArray<Type> ExpEdgesOut() const;
	
	/** 
	 * @brief Exports the layer to a plain array
	 * \n Matrix layout:
	 * \n NEURON1			: edge1, edge1, edge[n < iWidth] ==> directing to next neuron 1, 2, n
	 * \n NEURON2			: edge2, edge2, edge[n < iWidth] ==> directing to next neuron 1, 2, n
	 * \n NEURON3			: edge3, edge3, edge[n < iWidth] ==> directing to next neuron 1, 2, n
	 * \n NEURON[i < iHeight] 	: edge4, edge4, edge[n < iWidth] ==> directing to next neuron 1, 2, n
	 * @param iStart Start index for export
	 * @param iStart Stop index for export
	 * @return Returns a matrix: width=size_this_layer; height=size_of_next_layer
	 */
	virtual F2DArray<Type> ExpEdgesOut(int iStart, int iStop) const;
	
	/**
	 * @brief Imports weight informations from a weight matrix and saves them to the outgoing edges.
	 * @param f2dEdges Matrix which stores the weight informations
	 */
	virtual void ImpEdgesOut(const F2DArray<Type> &f2dEdges);
	
	/**
	 * @brief Imports weight informations from a weight matrix and saves them to the outgoing edges.
	 * @param f2dEdges Matrix which stores the weight informations
	 * @param iStart Start index for import
	 * @param iStart Stop index for import
	 */
	virtual void ImpEdgesOut(const F2DArray<Type> &f2dEdges, int iStart, int iStop);

	/////////////////////////////////////////////////////////////////////////////////////////////////////

	/**
	 * @brief Exports the layer to a plain array
	 * \n Matrix layout:
	 * \n NEURON1			: X, Y, POS[n < iWidth] ==> directing to input
	 * \n NEURON2			: X, Y, POS[n < iWidth] ==> directing to input
	 * \n NEURON3			: X, Y, POS[n < iWidth] ==> directing to input
	 * \n NEURON[i < iHeight] 	: X, Y, POS[n < iWidth] ==> directing to input
	 * @return Returns a matrix of positions
	 */
	virtual F2DArray<Type> ExpPositions() const;
	
	/**
	 * @brief Exports the position coordinates of the neurons of this layer to an array
	 * \n Matrix layout:
	 * \n NEURON1			: X, Y, POS[n < iWidth] ==> directing to input
	 * \n NEURON2			: X, Y, POS[n < iWidth] ==> directing to input
	 * \n NEURON3			: X, Y, POS[n < iWidth] ==> directing to input
	 * \n NEURON[i < iHeight] 	: X, Y, POS[n < iWidth] ==> directing to input
	 * @param iStart Start index for export
	 * @param iStart Stop index for export
	 * @return Returns a matrix of positions
	 */
	virtual F2DArray<Type> ExpPositions(int iStart, int iStop) const;
	
	/**
	 * @brief Imports an layer from a plain array
	 * @param f2dPos Matrix which stores the position coordinates of the neurons
	 */
	virtual void ImpPositions(const F2DArray<Type> &f2dPos);
	
	/**
	 * @brief Imports the position coordinates from a matrix and saves them to the neurons in this layer
	 * @param iStart Start index for import
	 * @param iStart Stop index for import
	 */
	virtual void ImpPositions(const F2DArray<Type> &f2dPos, int iStart, int iStop);
	
// FILE SYSTEM
	/**
	 * @brief Save layer's content to filesystem
	 */
	virtual void ExpToFS(BZFILE* bz2out, int iBZ2Error);
	/**
	 * @brief Load layer's content to filesystem
	 * @return The ID of the current layer.
	 */
	virtual int ImpFromFS(BZFILE* bz2in, int iBZ2Error, ConTable<Type> &Table);
	
#ifdef __AbsLayer_ADDON
	#include __AbsLayer_ADDON
#endif
};

#include "AbsLayer.tpp"
}

template <class T>
std::ostream& operator << (std::ostream &os, ANN::AbsLayer<T> *op)
{
	os << "Nr. neurons: \t" << op->GetNeurons().size() << std::endl;
	return os;     // Ref. auf Stream
}
