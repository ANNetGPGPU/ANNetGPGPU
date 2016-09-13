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
#include "AbsLayer.h"
#include "Common.h"
#include "containers/ConTable.h"

#include <iostream>
#include <vector>
#include <stdint.h>
#endif

namespace ANN {

// own classes
class Function;
template <class T> class ConTable;
template <class T> class HebbianConf;

/**
 * \brief Represents a container for neurons in a back propagation network.
 *
 * Functions implemented here allow us to connect neurons with each other.
 *
 * @author Daniel "dgrat" Frenzel
 */
template <class Type, class Functor>
class BPLayer : public AbsLayer<Type> {
	int m_iZLayer;
	
public:
	BPLayer();
	
	/**
	 * Creates a new layer
	 */
	BPLayer(int iZLayer);
	/**
	 * Copy constructor for creation of a new layer with the "same" properties like *pLayer
	 * this constructor can't copy connections (edges), because they normally have dependencies to other layers.
	 * @param pLayer object to copy properties from
	 */
	BPLayer(const BPLayer<Type, Functor> *pLayer, int iZLayer);
	/**
	 * Creates a new layer
	 * @param iNumber Number of neurons of this layer.
	 *
	 * @param fType Flag describing the type of the layer.
	 */
	BPLayer(const unsigned int &iNumber, LayerTypeFlag fType, int iZLayer = -1);

	/**
	 * Sets the z-layer of the layer. The z-layer defines, when a layer gets processed by the network.
	 * If more than one layer shares the same z-layer the processing must happen at the same time or directly after one of got finished.
	 *
	 * @param iZLayer z-layer
	 */
	void SetZLayer(int iZLayer);

	/**
	 * Sets the z-layer of the layer. The z-layer defines, when a layer gets processed by the network.
	 * If more than one layer shares the same z-layer the processing must happen at the same time or directly after one of got finished.
	 *
	 * @return z-layer
	 */
	int GetZLayer();

	/**
	 * Resizes the layer. Deletes old neurons and adds new ones (initialized with random values).
	 * @param iSize New number of neurons.
	 * @param iShiftID Value which has to get added to the ID of each neuron.
	 */
	virtual void Resize(const unsigned int &iSize);

	/**
	 * Adds neurons to the layer
	 * @param iSize stands for the number of neurons which get added to the layer.
	 */
	virtual void AddNeurons(const unsigned int &iSize);

	/**
	 * Sets the type of the layer (input, hidden or output layer)
	 * @param fType Flag describing the type of the layer.
	 */
	virtual void SetFlag(const LayerTypeFlag &fType);
	/**
	 * Adds a flag if not already set.
	 * @param fType Flag describing the type of the layer.
	 */
	virtual void AddFlag(const LayerTypeFlag &fType);

	/**
	 * Connects this layer with another one "pDestLayer".
	 * @param pDestLayer is a pointer of the layer to connect with.
	 * @param Connections is an array which describes these connections.
	 * The first index of this array is equal to the ID of the neuron in the actual (this) layer.
	 * The second ID is equal to the ID of neurons in the other (pDestLayer).
	 * @param bAllowAdapt allows the change of the weights between both layers.
	 */
	void ConnectLayer(AbsLayer<Type> *pDestLayer, std::vector<std::vector<int> > Connections, const bool bAllowAdapt = true);
	/**
	 * Connects this layer with another one.
	 * Each neuron of this layer with each of the neurons in "pDestLayer".
	 * Neurons in "pDestLayer" get
	 * @param pDestLayer pointer to layer to connect with.
	 * @param bAllowAdapt allows the change of the weights between both layers.
	 */
	void ConnectLayer(AbsLayer<Type> *pDestLayer, const bool &bAllowAdapt = true);

	/**
	 * Define the learning rate, the weight decay and the momentum term.
	 */
	void Setup(const HebbianConf<Type> &config);

	/**
	 * TODO
	 */
	virtual void ImpMomentumsEdgesIn(const F2DArray<Type> &);
	/**
	 * TODO
	 */
	virtual void ImpMomentumsEdgesOut(const F2DArray<Type> &);

// FILE SYSTEM :
	/**
	 * @brief Save layer's content to filesystem
	 */
	virtual void ExpToFS(BZFILE* bz2out, int iBZ2Error);
	/**
	 * @brief Load layer's content to filesystem
	 * @return The ID of the current layer.
	 */
	virtual int ImpFromFS(BZFILE* bz2in, int iBZ2Error, ConTable<Type> &Table);
	
#ifdef __BPLayer_ADDON
	#include __BPLayer_ADDON
#endif
};

#include "BPLayer.tpp"

}
