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
#include <algorithm>
#include <cassert>
#include <limits>
#include <cmath>

#include <omp.h>

#include "AbsNet.h"
#include "Edge.h"

#include "SOMNet.h"
#include "SOMLayer.h"
#include "SOMNeuron.h"

#include "math/Functions.h"
#include "math/Random.h"

#include "containers/TrainingSet.h"
#include "containers/ConTable.h"
#include "containers/Centroid.h"

#include <vector>
#include <map>
#endif

namespace ANN {

template <class T> class SOMNeuron;

enum {
	ANRandomMode 	= 1 << 0,	// type of layer
	ANSerialMode 	= 1 << 1,	// type of layer
};
typedef uint32_t TrainingMode;

/**
 * @class SOMNet
 * @brief Implementation of a self organizing map.
 */
template<class Type, class Functor>
class SOMNet : public AbsNet<Type> {
protected:
	Functor 	m_DistFunction;
	SOMNeuron<Type> *m_pBMNeuron = nullptr;

	uint32_t 	m_iCycle = 0;	// current cycle step in learning progress
	uint32_t 	m_iCycles = 0;	// maximum of cycles

	/* first Ctor */
	std::vector<uint32_t> m_vDimI; 	// dimensions of the input layer (Cartesian coordinates)
	std::vector<uint32_t> m_vDimO; 	// dimensions of the output layer (Cartesian coordinates)

	/* second Ctor */
	uint32_t 	m_iWidthI = 0;	// width of the input layer
	uint32_t 	m_iHeightI = 0;	// height of the input layer
	uint32_t 	m_iWidthO = 0;	// width of the output layer
	uint32_t 	m_iHeightO = 0; // height of the output layer

	void TrainHelper(uint32_t);

protected:
	/**
	 * @brief Implements part of training process. 
	 * Calculates the initial Sigma0 value.
	 */
	void FindSigma0();		// size of the net
        
	/**
	 * @brief Implements part of training process. 
	 * Searches for the best matching unit (neuron which fits best to current input). 
	 */
	void FindBMNeuron();	// best matching unit

	/**
	 * @brief Implements part of training process. 
	 * Propagates through the network backwardly.
	 */
	void PropagateBW();

	/**
	 * @brief Propagates through the network forwardly.
	 */
	void PropagateFW();

	/**
	 * @brief Adds a new layer to the network. New layer will get appended to m_lLayers.
	 * @param pLayer Pointer to the new layer.
	 */
	virtual void AddLayer(AbsLayer<Type> *pLayer);

public:
	/**
	 * @brief Creates a self organizing map object.
	 */
	SOMNet(AbsNet<Type> *pNet = nullptr);

	/**
	 * @brief Creates a double layered network. Each layer with vDim[1] * vDim[2] * vDim[n+1] * .. neurons.
	 * @param vDimI vector inheriting the dimensions of the input layer: vDim[X], vDim[Y], vDim[Z], vDim[DimN], ..
	 * @param vDimO vector inheriting the dimensions of the output layer: vDim[X], vDim[Y], vDim[Z], vDim[DimN], ..
	 */
	SOMNet(const std::vector<uint32_t> &vDimI, const std::vector<uint32_t> &vDimO);

	/**
	 * @brief Creates a double layered network.
	 * @param iWidthI Width of the input layer
	 * @param iHeightI Height of the input layer
	 * @param iWidthO Width of the output layer
	 * @param iHeightO Height of the output layer
	 */
	SOMNet(	const uint32_t &iWidthI, const uint32_t &iHeightI,
		const uint32_t &iWidthO, const uint32_t &iHeightO);

	/**
	 * @brief Defines the starting activation distance. 
	 * Sets the initial Sigma0 value. This distance to a BMU determines whether a neuron can be influenced during a training step. 
	 * During training this distance shrinks and Sigma0 is just the starting value.
	 */
	 void SetSigma0(const Type &fVal);
	
	/**
	 * @brief Adds a layer to the network.
	 * @param iSize Number of neurons of the layer.
	 * @param flType Flag describing the type of the net.
	 */
	virtual AbsLayer<Type> *AddLayer(const uint32_t &iSize, const LayerTypeFlag &flType);

	/**
	 * @brief Creates the network based on a connection table.
	 * @param ConTable is the connection table
	 */
	void CreateNet(const ConTable<Type> &Net);

	/**
	 * @brief Creates a double layered network. Each layer with vDim[1] * vDim[2] * vDim[n+1] * .. neurons.
	 * The layers will get automatically connected properly, which means,
	 * every neuron in the output layer is connected to each neuron in the input layer.
	 * @param vDimI vector inheriting the dimensions of the input layer: vDim[X], vDim[Y], vDim[Z], vDim[DimN], ..
	 * @param vDimO vector inheriting the dimensions of the output layer: vDim[X], vDim[Y], vDim[Z], vDim[DimN], ..
	 */
	void CreateSOM(	const std::vector<uint32_t> &vDimI,
			const std::vector<uint32_t> &vDimO);

	/**
	 * @brief Creates a double layered network. Each layer with vDim[1] * vDim[2] * vDim[n+1] * .. neurons.
	 * The layers will get automatically connected properly, which means,
	 * every neuron in the output layer is connected to each neuron in the input layer.
	 * @param vDimI vector inheriting the dimensions of the input layer: vDim[X], vDim[Y], vDim[Z], vDim[DimN], ..
	 * @param vDimO vector inheriting the dimensions of the output layer: vDim[X], vDim[Y], vDim[Z], vDim[DimN], ..
	 * @param f2dEdgeMat Matrix containing the values of the edges of the network.
	 * @param f2dNeurPos Matrix containing the position coordinates of the network.
	 */
	void CreateSOM(	const std::vector<uint32_t> &vDimI,
			const std::vector<uint32_t> &vDimO,
			const F2DArray<Type> &f2dEdgeMat,
			const F2DArray<Type> &f2dNeurPos);

	/**
	 * @brief Creates a double layered network.
	 * @param iWidthI Width of the input layer
	 * @param iHeightI Height of the input layer
	 * @param iWidthO Width of the output layer
	 * @param iHeightO Height of the output layer
	 */
	void CreateSOM(	const uint32_t &iWidthI, const uint32_t &iHeightI,
			const uint32_t &iWidthO, const uint32_t &iHeightO);

	/**
	 * @brief Trains the network with given input until iCycles is reached.
	 * @param iCycles Maximum number of training cycles.
	 * @param eMode 
	 * Value: ANRandomMode is faster, because one random input pattern is presented and a new cycle starts.\n
	 * Value: ANSerialMode means, that all input patterns are presented in order. Then a new cycle starts.
	 */
	virtual void Training(const uint32_t &iCycles = 1000, const TrainingMode &eMode = ANN::ANRandomMode);
	
	/**
	 * @brief The Cartesian position of a neuron in the network
	 * @return std::vector<Type> Returns the position.
	 */
	std::vector<Type> GetPosition(const uint32_t iNeuronID);
	
	/**
	 * @brief Sets learning rate scalar of the network.
	 * @param fVal New value of the learning rate. Recommended: 0.005f - 1.0f
	 */
	void SetLearningRate (const Type &fVal);

	/**
	 * @brief Clustering results of the network.
	 * @return std::vector<Centroid> Iterates through the input list and calcs the euclidean distance based on the BMU.
	 */
	virtual std::vector<Centroid<Type>> FindAllCentroids();
	
#ifdef __SOMNet_ADDON
	#include __SOMNet_ADDON
#endif
};

#include "SOMNet.tpp"

}
