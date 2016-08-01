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
#include "AbsLayer.h"
#include "Common.h"
#include "containers/TrainingSet.h"
#include "containers/ConTable.h"
#include "math/Random.h"

#include <bzlib.h>
#include <cassert>
#include <cmath>
#include <iostream>
#include <omp.h>
#include <sstream>
#include <string>
#include <vector>
#endif

namespace ANN {
template <class T> class ConTable;
template <class T> class TrainingSet;
template <class T> class AbsLayer;
template <class T> class AbsNeuron;

enum {
	ANNetSOM 	= 1 << 0,	// type of layer
	ANNetBP 	= 1 << 1,	// type of layer
	ANNetHopfield 	= 1 << 2,	// type of layer
	ANNetUndefined 	= 1 << 3
};
typedef uint32_t NetTypeFlag;


/**
 * @class AbsNet
 * @brief Represents a container for all layers in the network.
 * @author Daniel "dgrat" Frenzel
 */
template <class Type>
class AbsNet {
protected:
	NetTypeFlag m_fTypeFlag;

	TrainingSet<Type> *m_pTrainingData;			// list of training data

	// TODO maybe USE MAP for index administration?!
	std::vector<AbsLayer<Type>*> m_lLayers;			// list of all layers, layer->GetID() must be identical with indices of this array!
	AbsLayer<Type> *m_pIPLayer;				// pointer to input layer
	AbsLayer<Type> *m_pOPLayer;				// pointer to output layer

	/**
	 * @brief Adds a layer to the network.
	 * @param iSize Number of neurons of the layer.
	 * @param flType Flag describing the type of the net.
	 */
	virtual void AddLayer(const unsigned int &iSize, const LayerTypeFlag &flType) = 0;

public:
	AbsNet();
	virtual ~AbsNet();

	/**
	 * @brief Creates a network in memory from container structure
	 * @param Net container structure to create a network in memory.
	 */
	virtual void CreateNet(const ConTable<Type> &Net);

	/**
	 * @brief Implement to determine propagation behavior
	 */
	virtual void PropagateFW() = 0;
	
	/**
	 * @brief Implement to determine back propagation ( == learning ) behavior
	 */
	virtual void PropagateBW() = 0;

	/**
	 * @brief Sets the type of the net
	 * @param fType Flag describing the type of the net.
	 */
	virtual void SetFlag(const NetTypeFlag &fType);
	
	/**
	 * @brief Adds a flag if not already set.
	 * @param fType Flag describing the type of the net.
	 */
	virtual void AddFlag(const NetTypeFlag &fType);
	
	/**
	 * @brief Type of the net
	 * @return Returns the flag describing the type of the net.
	 */
	NetTypeFlag GetFlag() const;

	/**
	 * @brief Cycles the input from m_pTrainingData and Checks total error of the output returned from SetExpectedOutputData()
	 * @return Returns the total error of the net after every training step.
	 * @param iCycles Maximum number of training cycles
	 * @param fTolerance Maximum error value (working as a break condition for early break-off)
	 */
	virtual std::vector<Type> TrainFromData(const unsigned int &iCycles, const Type &fTolerance, const bool &bBreak, Type &fProgress);

	/**
	 * @brief Adds a new layer to the network. New layer will get appended to m_lLayers.
	 * @param pLayer Pointer to the new layer.
	 */
	virtual void AddLayer(AbsLayer<Type> *pLayer);
	
	/**
	 * @brief List of all layers of the net.
	 * @return Returns an array with pointers to every layer.
	 */
	virtual std::vector<AbsLayer<Type> *> GetLayers() const;

	/**
	 * @brief Deletes the complete network (all connections and all values).
	 */
	virtual void EraseAll();

	/**
	 * @brief Set the value of neurons in the input layer to new values
	 * @param inputArray New values of the input layer
	 */
	virtual void SetInput(const std::vector<Type> &inputArray);		// only usable if input or output layer was set
	
	/**
	 * @brief Set the value of neurons in the input layer to new values
	 * @param inputArray New values of the input layer
	 * @param iLayerID Index of the layer in m_lLayers
	 */
	virtual void SetInput(const std::vector<Type> &inputArray, const unsigned int &iLayerID);
	
	/**
	 * @brief Set the value of neurons in the input layer to new values
	 * @param pInputArray New values of the input layer
	 * @param iLayerID Index of the layer in m_lLayers
	 * @param iSize Number of values in pInputArray
	 */
	virtual void SetInput(Type *pInputArray, const unsigned int &iSize, const unsigned int &iLayerID);

	/**
	 * @brief Set the values of the neurons equal to the values of the outputArray. Also calcs the error delta of each neuron in the output layer.
	 * @return returns the total error of the output layer ( sum(pow(delta, 2)/2.f )
	 * @param outputArray New values of the output layer
	 */
	virtual Type SetOutput(const std::vector<Type> &outputArray); 	// only usable if input or output layer was set
	
	/**
	 * @brief Set the values of the neurons equal to the values of the outputArray. Also calcs the error delta of each neuron in the output layer.
	 * @return returns the total error of the output layer ( sum(pow(delta, 2)/2.f )
	 * @param outputArray New values of the output layer
	 * @param iLayerID Index of the layer in m_lLayers
	 */
	virtual Type SetOutput(const std::vector<Type> &outputArray, const unsigned int &iLayerID);
	
	/**
	 * @brief Set the values of the neurons equal to the values of the outputArray. Also calcs the error delta of each neuron in the output layer.
	 * @return returns the total error of the output layer ( sum(pow(delta, 2)/2.f )
	 * @param pOutputArray New values of the output layer
	 * @param iSize Number of values in pInputArray
	 * @param iLayerID Index of the layer in m_lLayers
	 */
	virtual Type SetOutput(Type *pOutputArray, const unsigned int &iSize, const unsigned int &iLayerID);

	/**
	 *  @brief Sets training data of the net.
	 */
	virtual void SetTrainingSet(const TrainingSet<Type> *pData);
	
	/**
	 *  @brief Sets training data of the net.
	 */
	virtual void SetTrainingSet(const TrainingSet<Type> &Data);
	
	/**
	 *  @brief Training data of the net.
	 *  @return Returns the current training set of the net or NULL if nothing was set.
	 */
	virtual TrainingSet<Type> *GetTrainingSet() const;

	/**
	 * @brief Returns layer at index iLayerID.
	 * @return Pointer to Layer at iLayerID.
	 */
	virtual AbsLayer<Type>* GetLayer(const unsigned int &iLayerID) const;

	/**
	 * @brief Pointer to the input layer (If input layer was already defined).
	 * @return Returns a pointer to the input layer.
	 */
	virtual AbsLayer<Type> *GetIPLayer() const;
	/**
	 * @brief Pointer to the output layer (If output layer was already defined).
	 * @return Returns a pointer to the output layer.
	 */
	virtual AbsLayer<Type> *GetOPLayer() const;
	
	/**
	 * @brief Sets the input layer
	 * @param iID ID of the layer.
	 */
	virtual void SetIPLayer(const unsigned int iID);
	/**
	 * @brief Sets the output layer
	 * @param iID ID of the layer.
	 */
	virtual void SetOPLayer(const unsigned int iID);

	/**
	 * @brief Save net's content to filesystem
	 */
	virtual void ExpToFS(std::string path);
	/**
	 * @brief Load net's content to filesystem
	 * @return The connections table of this net.
	 */
	virtual void ImpFromFS(std::string path);

	/**
	 * @brief Only usable if input/output layer was already set.
	 * @return Returns the values of the output layer after propagating the net.
	 */
	virtual std::vector<Type> GetOutput();
	
#ifdef __AbsNet_ADDON
	#include __AbsNet_ADDON
#endif
};

#include "AbsNet.tpp"
}

template <class T>
std::ostream& operator << (std::ostream &os, ANN::AbsNet<T> *op) {
	assert( op->GetOPLayer() != NULL );
	if( op->GetTrainingSet() != NULL ) {
		for( unsigned int i = 0; i < op->GetTrainingSet()->GetNrElements(); i++ ) {
			op->SetInput( op->GetTrainingSet()->GetInput(i) );
			op->PropagateFW();

			for(unsigned int j = 0; j < op->GetOPLayer()->GetNeurons().size(); j++) {
				ANN::AbsNeuron<T> *pCurNeuron = op->GetOPLayer()->GetNeuron(j);
				std::cout << pCurNeuron;
			}
			std::cout << std::endl;
		}
	} else {
		for(unsigned int i = 0; i < op->GetOPLayer()->GetNeurons().size(); i++) {
			ANN::AbsNeuron<T> *pCurNeuron = op->GetOPLayer()->GetNeuron(i);
			std::cout << pCurNeuron;
		}
	}
	return os;
}
