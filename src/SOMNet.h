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
#endif

namespace ANN {

template <class T> class SOMNeuron;
template <class T> class Centroid;

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
	Functor		m_DistFunction;
	SOMNeuron<Type> *m_pBMNeuron;

	unsigned int 	m_iCycle;	// current cycle step in learning progress
	unsigned int 	m_iCycles;	// maximum of cycles
	
	// Conscience mechanism
	Type 		m_fConscienceRate;

	/* first Ctor */
	std::vector<unsigned int> m_vDimI; // dimensions of the input layer (Cartesian coordinates)
	std::vector<unsigned int> m_vDimO; // dimensions of the output layer (Cartesian coordinates)

	/* second Ctor */
	unsigned int 	m_iWidthI;	// width of the input layer
	unsigned int 	m_iHeightI;	// height of the input layer
	unsigned int 	m_iWidthO;	// width of the output layer
	unsigned int 	m_iHeightO; 	// height of the output layer

	void TrainHelper(unsigned int);

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
	 * @brief Adds a layer to the network.
	 * @param iSize Number of neurons of the layer.
	 * @param flType Flag describing the type of the net.
	 */
	virtual void AddLayer(const unsigned int &iSize, const LayerTypeFlag &flType);

public:
	/**
	 * @brief Creates a self organizing map object.
	 */
	SOMNet();
	/**
	 * @brief Creates a self organizing map object.
	 */
	SOMNet(AbsNet<Type> *pNet);

	/**
	 * @brief Creates a double layered network. Each layer with vDim[1] * vDim[2] * vDim[n+1] * .. neurons.
	 * @param vDimI vector inheriting the dimensions of the input layer: vDim[X], vDim[Y], vDim[Z], vDim[DimN], ..
	 * @param vDimO vector inheriting the dimensions of the output layer: vDim[X], vDim[Y], vDim[Z], vDim[DimN], ..
	 */
	SOMNet(const std::vector<unsigned int> &vDimI, const std::vector<unsigned int> &vDimO);

	/**
	 * @brief Creates a double layered network.
	 * @param iWidthI Width of the input layer
	 * @param iHeightI Height of the input layer
	 * @param iWidthO Width of the output layer
	 * @param iHeightO Height of the output layer
	 */
	SOMNet(	const unsigned int &iWidthI, const unsigned int &iHeightI,
		const unsigned int &iWidthO, const unsigned int &iHeightO);

	virtual ~SOMNet();

	/**
	 * @brief Defines the starting activation distance. 
	 * Sets the initial Sigma0 value. This distance to a BMU determines whether a neuron can be influenced during a training step. 
	 * During training this distance shrinks and Sigma0 is just the starting value.
	 */
	 void SetSigma0(const Type &fVal);
	
	/**
	 * @brief Adds a new layer to the network. New layer will get appended to m_lLayers.
	 * @param pLayer Pointer to the new layer.
	 */
	virtual void AddLayer(AbsLayer<Type> *pLayer);

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
	void CreateSOM(	const std::vector<unsigned int> &vDimI,
			const std::vector<unsigned int> &vDimO);

	/**
	 * @brief Creates a double layered network. Each layer with vDim[1] * vDim[2] * vDim[n+1] * .. neurons.
	 * The layers will get automatically connected properly, which means,
	 * every neuron in the output layer is connected to each neuron in the input layer.
	 * @param vDimI vector inheriting the dimensions of the input layer: vDim[X], vDim[Y], vDim[Z], vDim[DimN], ..
	 * @param vDimO vector inheriting the dimensions of the output layer: vDim[X], vDim[Y], vDim[Z], vDim[DimN], ..
	 * @param f2dEdgeMat Matrix containing the values of the edges of the network.
	 * @param f2dNeurPos Matrix containing the position coordinates of the network.
	 */
	void CreateSOM(	const std::vector<unsigned int> &vDimI,
			const std::vector<unsigned int> &vDimO,
			const F2DArray<Type> &f2dEdgeMat,
			const F2DArray<Type> &f2dNeurPos);

	/**
	 * @brief Creates a double layered network.
	 * @param iWidthI Width of the input layer
	 * @param iHeightI Height of the input layer
	 * @param iWidthO Width of the output layer
	 * @param iHeightO Height of the output layer
	 */
	void CreateSOM(	const unsigned int &iWidthI, const unsigned int &iHeightI,
			const unsigned int &iWidthO, const unsigned int &iHeightO);

	/**
	 * @brief Trains the network with given input until iCycles is reached.
	 * @param iCycles Maximum number of training cycles.
	 * @param eMode 
	 * Value: ANRandomMode is faster, because one random input pattern is presented and a new cycle starts.\n
	 * Value: ANSerialMode means, that all input patterns are presented in order. Then a new cycle starts.
	 */
	void Training(const unsigned int &iCycles = 1000, const TrainingMode &eMode = ANN::ANRandomMode);

	/**
	 * @brief Clustering results of the network.
	 * @return std::vector<Centroid> Iterates through the input list and calcs the euclidean distance based on the BMU.
	 */
	std::vector<Centroid<Type>> GetCentroidList();

	/**
	 * @brief The Cartesian position of a neuron in the network
	 * @return std::vector<Type> Returns the position.
	 */
	std::vector<Type> GetPosition(const unsigned int iNeuronID);
	
	/**
	 * @brief Sets learning rate scalar of the network.
	 * @param fVal New value of the learning rate. Recommended: 0.005f - 1.0f
	 */
	void SetLearningRate 	(const Type &fVal);

	/**
	 * @brief Sets the scalar for the conscience mechanism. If it is zero, then conscience is not applied.
	 * A value of zero leads to the standard kohonen implementation.
	 * Value must be: 0.f < fVal < 1.f
	 */
	void SetConscienceRate(const Type &fVal);

	/**
	 * @brief Returns the conscience scalar of the network. If it is zero, then conscience is not applied.
	 * @return Returns the rate for the application of the conscience mechanism. 
	 * A value of zero leads to the standard kohonen implementation. 
	 * Value must be: 0.f < fVal < 1.f
	 */
	Type GetConscienceRate();
	
#ifdef __SOMNet_ADDON
	#include __SOMNet_ADDON
#endif
};

#include "SOMNet.tpp"

}
