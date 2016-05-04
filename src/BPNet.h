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

#ifndef SIMPLENET_H_
#define SIMPLENET_H_

#ifndef SWIG
#include "AbsNet.h"

#include <vector>
#include <string>
#endif

namespace ANN {

class BPLayer;

/**
 * \brief Implementation of a back propagation network.
 *
 * @author Daniel "dgrat" Frenzel
 */
class BPNet : public AbsNet
{
protected:
	/**
	 * Adds a layer to the network.
	 * @param iSize Number of neurons of the layer.
	 * @param flType Flag describing the type of the net.
	 */
	virtual void AddLayer(const unsigned int &iSize, const LayerTypeFlag &flType);

public:
	/**
	 * Standard constructor
	 */
	BPNet();
	/**
	 * Copy constructor for copying the complete network:
	 * @param pNet
	 */
	BPNet(ANN::BPNet *pNet);

	virtual ~BPNet();

	/*
	 *
	 */
	virtual void CreateNet(const ConTable &Net);

	/**
	 * Adds a new layer to the network. New layer will get appended to m_lLayers.
	 * @param pLayer Pointer to the new layer.
	 */
	virtual void AddLayer(BPLayer *pLayer);

	/**
	 * Cycles the input from m_pTrainingData
	 * Checks total error of the output returned from SetExpectedOutputData()
	 * @return Returns the total error of the net after every training step.
	 * @param iCycles Maximum number of training cycles
	 * @param fTolerance Maximum error value (working as a break condition for early break-off)
	 */
	virtual std::vector<float> TrainFromData(const unsigned int &iCycles, const float &fTolerance, const bool &bBreak, float &fProgress);

	/**
	 * Propagates through all neurons of the net beginning from the input layer.
	 * Updates all neuron values of the network.
	 *
	 * The neuron output is defined as:
	 * \f$
	 * o_{j}=\varphi(\mbox{net}_{j})
	 * \f$
	 * , whereas the neuron input is defined as:
	 * \f$
	 * \mbox{net}_{j}=\sum\limits_{i=1}^{n} x_{i} w_{ij}.
	 * \f$
	 * \n
	 * \f$
	 * 	\\	\varphi\ \text{ is a differentiable activation function,}
	 * 	\\	n \text{ is the number of inputs,}
	 * 	\\	x_{i} \text{ is the input } i \text{ and}
	 * 	\\	w_{ij} \text{ is the weight between neuron } i \text{ and neuron } j
	 * \f$.
	 */
	virtual void PropagateFW();
	/**
	 * Propagates through all neurons of the net beginning from the output layer. \n
	 * Calculates error deltas of neurons from current learning output and training output data. \n
	 * Also updates all weights of the net beginning from the output layer. \n
	 * The backpropagation works as described below: \n \n
	 * \f$
	 * 	\\	\mbox{1. Is the neuron in the output layer, it takes part of the output,}
	 * 	\\	\mbox{2. is the neuron in the hidden layer, the weight adaption could get calculated.}
	 * 	\\	\mbox{	concrete:}
	 *
	 * 	\\	\Delta w_{ij}(t+1)= \eta \delta_{j} x_{i} + \alpha \Delta w_{ij}(t)
	 *
	 * 	\\	\mbox{	with}
	 *
	 * 	\\	\delta_{j}=\begin{cases}
	 * 		\varphi'(\mbox{net}_{j})(t_{j}-o_{j}) & \mbox{if } j \mbox{ is an output neuron,}\\
	 * 		\varphi'(\mbox{net}_{j}) \sum_{k} \delta_{k} w_{jk} & \mbox{if } j \mbox{ is an hidden neuron.}
	 * 		\end{cases}
	 *
	 *	\\ 	\mbox{	and}
	 *
	 * 	\\	\Delta w_{ij} \mbox{ is the change of the weight } w_{ij} \mbox{ of the connection }i\mbox{ to neuron }j\mbox{,}
	 * 	\\	\eta \mbox{ is the learning rate, which regulates to amount of the weight change,}
	 * 	\\	\delta_{j} \mbox{ is the error signal of the neuron } j mbox{,}
	 * 	\\	x_{i} \mbox{ is the output of the neuron } i \mbox{,}
	 * 	\\	t_{j} \mbox{ is the debit output of the output neuron } j \mbox{,}
	 * 	\\	o_{j} \mbox{ is the actual output of the output neuron } j \mbox{ und}
	 * 	\\	k \mbox{ is the index of the subsequent neurons of } j \mbox{.}
	 * 	\\ 	\Delta w_{ij}(t+1) \mbox{ is the change of the weight } w_{ij}(t+1) \mbox{ of the connection of neuron } i \mbox{ to neuron } j \mbox{ at the time point (t+1),}
	 * 	\\ 	\alpha \mbox{ is the influence of the momentum term } \Delta w_{ij}(t) \mbox{. Correlates with the weight change of the prior time point.}
	 * \f$
	 */
	virtual void PropagateBW();

	/**
	 * Will create a sub-network from layer "iStartID" to layer "iStopID".
	 * This network will have all the properties of the network it is derivated from,
	 * however without the layer of edges between "iStartID" and "iStopID".
	 * Also the first and last layers of the new sub-net will automatically get a new flag as input or output layer.
	 *
	 * @param iStartID
	 * @param iStopID
	 * @return Returns a pointer to the new sub-network.
	 */
	BPNet *GetSubNet(const unsigned int &iStartID, const unsigned int &iStopID);

	/**
	 * Sets learning rate scalar of the network.
	 * @param fVal New value of the learning rate. Recommended: 0.005f - 1.0f
	 */
	void SetLearningRate 	(const float &fVal);
	/**
	 * @return Return the learning rate of the net.
	 */
	float GetLearningRate() const;
	
	/**
	 * Sets momentum scalar of the network.
	 * @param fVal New value of the momentum. Recommended: 0.3f - 0.9f
	 */
	void SetMomentum 		(const float &fVal);
	/**
	 * @return Returns the momentum scalar of the net.
	 */
	float GetMomentum() const;
	
	/**
	 * Sets weight decay of the network.
	 * @param fVal New value of the weight decay. Recommended: 0.f
	 */
	void SetWeightDecay 	(const float &fVal);
	/**
	 * @return Return the weight decay scalar of the net.
	 */
	float GetWeightDecay() const;
};

}

#endif /* NET_H_ */
