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

#ifndef SOMNET_H_
#define SOMNET_H_

#ifndef SWIG
#include "AbsNet.h"

#include <vector>
#endif

namespace ANN {

class SOMNeuron;
class Centroid;
class DistFunction;

enum {
	ANRandomMode 	= 1 << 0,	// type of layer
	ANSerialMode 	= 1 << 1,	// type of layer
};
typedef uint32_t TrainingMode;

/**
 * @class SOMNet
 * @brief Implementation of a self organizing map.
 */
class SOMNet : public AbsNet {
protected:
	DistFunction 	*m_DistFunction;
	SOMNeuron 	*m_pBMNeuron;

	unsigned int 	m_iCycle;	// current cycle step in learning progress
	unsigned int 	m_iCycles;	// maximum of cycles
	float 		m_fSigma0;	// radius of the lattice at t0
	float 		m_fSigmaT;	// radius of the lattice at tx
	float 		m_fLambda;	// time constant
	float 		m_fLearningRateT;
	
	// Conscience mechanism
	float 		m_fConscienceRate;

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
	SOMNet(AbsNet *pNet);

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
	 * @brief Adds a new layer to the network. New layer will get appended to m_lLayers.
	 * @param pLayer Pointer to the new layer.
	 */
	virtual void AddLayer(AbsLayer *pLayer);

	/**
	 * @brief Creates the network based on a connection table.
	 * @param ConTable is the connection table
	 */
	void CreateNet(const ConTable &Net);

	/**
	 * @brief Returns a pointer to the SOM.
	 * @return the pointer to the SOM
	 */
	SOMNet *GetNet();

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
			const F2DArray &f2dEdgeMat,
			const F2DArray &f2dNeurPos);

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
	 * @return std::vector<Centroid> Returns to each input value the obtained centroid with the euclidean distance and the corresponding ID of the BMU.
	 */
	std::vector<Centroid> GetCentrOInpList();

	/**
	 * @brief Clustering results of the network.
	 * @return std::vector<Centroid> Returns the centroids found after training and the ID of the corresponding BMUs.
	 */
	std::vector<Centroid> GetCentroidList();

	/**
	 * @brief Sets learning rate scalar of the network.
	 * @param fVal New value of the learning rate. Recommended: 0.005f - 1.0f
	 */
	void SetLearningRate 	(const float &fVal);
	/**
	 * @brief Gets learning rate scalar of the network.
	 * @return Return the learning rate of the net.
	 */
	float GetLearningRate() const;

	/**
	 * @brief Sets the neighborhood and decay function of the network together.
	 * @param pFCN Kind of function the net has to use while back-/propagating.
	 */
	void SetDistFunction (const DistFunction *pFCN);

	/**
	 * @brief Sets the neighborhood and decay function of the network together.
	 * @param pFCN Kind of function the net has to use while back-/propagating.
	 */
	void SetDistFunction (const DistFunction &FCN);
	
	/**
	 * @brief Returns the currently used distance (neighborhood) function of the network.
	 * @return Return the kind of function the net has to use while back-/propagating.
	 */
	DistFunction* GetDistFunction();

	/**
	 * @brief Sets the scalar for the conscience mechanism. If it is zero, then conscience is not applied.
	 * A value of zero leads to the standard kohonen implementation.
	 * Value must be: 0.f < fVal < 1.f
	 */
	void SetConscienceRate(const float &fVal);

	/**
	 * @brief Returns the conscience scalar of the network. If it is zero, then conscience is not applied.
	 * @return Returns the rate for the application of the conscience mechanism. 
	 * A value of zero leads to the standard kohonen implementation. 
	 * Value must be: 0.f < fVal < 1.f
	 */
	float GetConscienceRate();
};

}

#endif /* SOMNET_H_ */
