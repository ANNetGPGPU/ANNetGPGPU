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
#include <cassert>
#include <omp.h>
#include <vector>

#include "AbsLayer.h"
#include "Common.h"
#include "containers/2DArray.h"
#endif


namespace ANN {

template <class T> class SOMNeuron;

/**
 * @class SOMLayer
 * @brief Container class for neurons in a self organizing map.
 */
template <class Type>
class SOMLayer : public AbsLayer<Type> {
private:
	std::vector<unsigned int> m_vDim;
	/*
	 * Flag describing the kind of layer.
	 * (i. e. input, hidden or output possible)
	 */
	LayerTypeFlag m_fTypeFlag;

public:
	/**
	 * @brief Creates an empty layer. 
	 */
	SOMLayer();
	/**
	 * @brief Copy CTOR. Creates a layer. 
	 */
	SOMLayer(const SOMLayer<Type> *pLayer);
	/**
	 * @brief Creates a layer with a certain number of neurons. 
	 * @param iSize Number of neurons in this layer. 
	 * @param fType Type of the layer. 
	 */
	SOMLayer(const unsigned int &iSize, LayerTypeFlag fType);
	/**
	 * @brief Creates a layer with iWidth X iHeight neurons. 
	 * @param iWidth Width of the layer. 
	 * @param iHeight Height of the layer.
	 * @param fType Type of the layer. 
	 */
	SOMLayer(const unsigned int &iWidth, const unsigned int &iHeight, LayerTypeFlag fType);
	/**
	 * @brief Creates a layer with (vDim[X] X vDim[Y] X vDim[Z] X .. X vDim[DimN]) neurons. 
	 * @param vDim Dimensions of the layer. 
	 * @param fType Type of the layer. 
	 */
	SOMLayer(const std::vector<unsigned int> &vDim, LayerTypeFlag fType);

	/**
	 * @brief Resizes the layer.
	 * @param iSize Number ofneurons in this layer.
	 */
	virtual void Resize(const unsigned int &iSize);
	/**
	 * @brief Resizes the layer.
	 * @param iWidth Width of the layer.
	 * @param iHeight Height of the layer.
	 */
	virtual void Resize(const unsigned int &iWidth, const unsigned int &iHeight);
	/**
	 * @brief Resizes the layer (vDim[X] X vDim[Y] X vDim[Z] X .. X vDim[DimN]).
	 * @param vDim Dimensions of the layer (vDim[X], vDim[Y], vDim[Z], vDim[DimN]).
	 */	
	virtual void Resize(const std::vector<unsigned int> &vDim);

	/**
	 * @brief Adds a given number of neurons to the network.
	 * @param iSize Number of neurons to add.
	 */
	virtual void AddNeurons(const unsigned int &iSize);

	/**
	 * @brief Connects this layer with another one.
	 * Each neuron of this layer with each of the destination layer.
	 * @param pDestLayer pointer to layer to connect with.
	 * @param bAllowAdapt allows the change of the weights between both layers.
	 */
	void ConnectLayer(AbsLayer<Type> *pDestLayer, const bool &bAllowAdapt = true);

	/** 
	 * @brief Connects this layer with another one.
	 * @param pDestLayer Destination layer to connect with.
	 * @param f2dEdgeMat Matrix containing the values of the edges of the network.
	 * @param bAllowAdapt Sets whether edges are changeable.
	 */
	void ConnectLayer(AbsLayer<Type> *pDestLayer, const F2DArray<Type> &f2dEdgeMat, const bool &bAllowAdapt = true);
	/**
	 * @brief Sets learning rate scalar of the network.
	 * @param fVal New value of the learning rate. Recommended: 0.005f - 1.0f
	 */
	void SetLearningRate(const float &fVal);

	/**
	 * @brief Returns the dimensions of the layer.
	 * @return Returns a vector of the dimensions of the layer (vDim[X], vDim[Y], vDim[Z], vDim[DimN]). 
	 */
	std::vector<unsigned int> GetDim() const;
	/**
	 * @brief Returns one dimension of the layer.
	 * @param iInd Index parameter.
	 * @return Returns the dimension of the layer at index iInd. 
	 */
	unsigned int GetDim(const unsigned int &iInd) const;
	
	/**
	 * @brief The Cartesian position of a neuron in the network
	 * @return std::vector<float> Returns the position.
	 */
	std::vector<float> GetPosition(const unsigned int iNeuronID);
	
#ifdef __SOMLayer_ADDON
	#include __SOMLayer_ADDON
#endif
};

#include "SOMLayer.tpp"

}

