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

#ifndef ANABSNET_H_
#define ANABSNET_H_

#ifndef SWIG
#include "AbsLayer.h"

#include <vector>
#include <string>
#include <bzlib.h>
#include <sstream>
#include <iostream>
#endif

namespace ANN {
 
// containers
class F2DArray;
class F3DArray;
class TrainingSet;
class ConTable;
// math
class TransfFunction;
// net
class AbsLayer;
class Layer;
class Neuron;
class AbsNeuron;
class Edge;

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
class AbsNet {
protected:
	NetTypeFlag m_fTypeFlag;

	TrainingSet *m_pTrainingData;			// list of training data
	float m_fLearningRate;				// global learning rate
	float m_fMomentum;
	float m_fWeightDecay;
	TransfFunction *m_pTransfFunction;

	/* list of all layers in this net; last should be output layer, first input layer */

	// TODO maybe USE MAP for index administration?!
	std::vector<AbsLayer*> m_lLayers;		// list of all layers, layer->GetID() must be identical with indices of this array!
	AbsLayer *m_pIPLayer;				// pointer to input layer
	AbsLayer *m_pOPLayer;				// pointer to output layer

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
	virtual void CreateNet(const ConTable &Net);

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
	virtual std::vector<float> TrainFromData(const unsigned int &iCycles, const float &fTolerance, const bool &bBreak, float &fProgress);

	/**
	 * @brief Adds a new layer to the network. New layer will get appended to m_lLayers.
	 * @param pLayer Pointer to the new layer.
	 */
	virtual void AddLayer(AbsLayer *pLayer);
	/**
	 * @brief List of all layers of the net.
	 * @return Returns an array with pointers to every layer.
	 */
	virtual std::vector<AbsLayer*> GetLayers() const;

	/**
	 * @brief Deletes the complete network (all connections and all values).
	 */
	virtual void EraseAll();

	/**
	 * @brief Set the value of neurons in the input layer to new values
	 * @param inputArray New values of the input layer
	 */
	virtual void SetInput(const std::vector<float> &inputArray);		// only usable if input or output layer was set
	/**
	 * @brief Set the value of neurons in the input layer to new values
	 * @param inputArray New values of the input layer
	 * @param iLayerID Index of the layer in m_lLayers
	 */
	virtual void SetInput(const std::vector<float> &inputArray, const unsigned int &iLayerID);
	/**
	 * @brief Set the value of neurons in the input layer to new values
	 * @param pInputArray New values of the input layer
	 * @param iLayerID Index of the layer in m_lLayers
	 * @param iSize Number of values in pInputArray
	 */
	virtual void SetInput(float *pInputArray, const unsigned int &iSize, const unsigned int &iLayerID);

	/**
	 * @brief Set the values of the neurons equal to the values of the outputArray. Also calcs the error delta of each neuron in the output layer.
	 * @return returns the total error of the output layer ( sum(pow(delta, 2)/2.f )
	 * @param outputArray New values of the output layer
	 */
	virtual float SetOutput(const std::vector<float> &outputArray); 	// only usable if input or output layer was set
	/**
	 * @brief Set the values of the neurons equal to the values of the outputArray. Also calcs the error delta of each neuron in the output layer.
	 * @return returns the total error of the output layer ( sum(pow(delta, 2)/2.f )
	 * @param outputArray New values of the output layer
	 * @param iLayerID Index of the layer in m_lLayers
	 */
	virtual float SetOutput(const std::vector<float> &outputArray, const unsigned int &iLayerID);
	/**
	 * @brief Set the values of the neurons equal to the values of the outputArray. Also calcs the error delta of each neuron in the output layer.
	 * @return returns the total error of the output layer ( sum(pow(delta, 2)/2.f )
	 * @param pOutputArray New values of the output layer
	 * @param iSize Number of values in pInputArray
	 * @param iLayerID Index of the layer in m_lLayers
	 */
	virtual float SetOutput(float *pOutputArray, const unsigned int &iSize, const unsigned int &iLayerID);

	/**
	 *  @brief Sets training data of the net.
	 */
	virtual void SetTrainingSet(const TrainingSet *pData);
	/**
	 *  @brief Sets training data of the net.
	 */
	virtual void SetTrainingSet(const TrainingSet &Data);
	/**
	 *  @brief Training data of the net.
	 *  @return Returns the current training set of the net or NULL if nothing was set.
	 */
	virtual TrainingSet *GetTrainingSet() const;

	/**
	 * @brief Returns layer at index iLayerID.
	 * @return Pointer to Layer at iLayerID.
	 */
	virtual AbsLayer* GetLayer(const unsigned int &iLayerID) const;

	/**
	 * @brief Pointer to the input layer (If input layer was already defined).
	 * @return Returns a pointer to the input layer.
	 */
	//virtual const AbsLayer *GetIPLayer() const;
	/**
	 * @brief Pointer to the output layer (If output layer was already defined).
	 * @return Returns a pointer to the output layer.
	 */
	//virtual const AbsLayer *GetOPLayer() const;

	/**
	 * @brief Pointer to the input layer (If input layer was already defined).
	 * @return Returns a pointer to the input layer.
	 */
	virtual AbsLayer *GetIPLayer() const;
	/**
	 * @brief Pointer to the output layer (If output layer was already defined).
	 * @return Returns a pointer to the output layer.
	 */
	virtual AbsLayer *GetOPLayer() const;
	
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
	 * @brief Defines the type of "activation" function the net has to use for back-/propagation.
	 * @param pFunction New "activation" function
	 */
	virtual void SetTransfFunction(const TransfFunction *pFunction);

	/**
	 * @brief Defines the type of "activation" function the net has to use for back-/propagation.
	 * @param pFunction New "activation" function
	 */
	virtual void SetTransfFunction(const TransfFunction &pFunction);

	/**
	 * @brief Access of the currently used transfer function of the network.
	 * @return Returns the current net (activation) function.
	 */
	virtual TransfFunction *GetTransfFunction();

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
	virtual std::vector<float> GetOutput();
	/**
	 * @brief standard output of the net. Only usable if input/output layer was already set.
	 */
	friend std::ostream& operator << (std::ostream &os, AbsNet &op);
};

}
#endif /* ANABSNET_H_ */
