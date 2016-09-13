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
#include "AbsNet.h"

#include "BPNeuron.h"
#include "BPLayer.h"

#include "Common.h"
#include "containers/ConTable.h"
#include "containers/TrainingSet.h"
#include "containers/ConTable.h"

#include "Edge.h"

#include "math/Functions.h"
#include "math/Random.h"

#include <vector>
#include <string>

#include <iostream>
#include <cassert>
#include <algorithm>
#include <omp.h>
#endif

namespace ANN {

class Function;
template <class T> class ConTable;
template <class T, class F> class BPNeuron;
template <class T, class F> class BPLayer;

/**
 * \brief Implementation of a back propagation network.
 *
 * @author Daniel "dgrat" Frenzel
 */
template <class Type, class Functor>
class BPNet : public AbsNet<Type>
{
protected:	
	HebbianConf<Type> m_Setup;

	/**
	 * Adds a new layer to the network. New layer will get appended to m_lLayers.
	 * @param pLayer Pointer to the new layer.
	 */
	virtual void AddLayer(BPLayer<Type, Functor> *pLayer);

	
public:
	/**
	 * Standard constructor
	 */
	BPNet();
	/**
	 * Copy constructor for copying the complete network:
	 * @param pNet
	 */
	BPNet(BPNet<Type, Functor> *pNet);

	/*
	 *
	 */
	virtual void CreateNet(const ConTable<Type> &Net);

	/**
	 * Adds a layer to the network.
	 * @param iSize Number of neurons of the layer.
	 * @param flType Flag describing the type of the net.
	 */
	virtual BPLayer<Type, Functor> *AddLayer(const unsigned int &iSize, const LayerTypeFlag &flType);
	
	/**
	 * Cycles the input from m_pTrainingData
	 * Checks total error of the output returned from SetExpectedOutputData()
	 * @return Returns the total error of the net after every training step.
	 * @param iCycles Maximum number of training cycles
	 * @param fTolerance Maximum error value (working as a break condition for early break-off)
	 */
	virtual std::vector<Type> TrainFromData(const unsigned int &iCycles, const Type &fTolerance, const bool &bBreak, Type &fProgress);

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
	 * \\	\varphi\ \text{ is a differentiable activation function,}
	 * \\	n \text{ is the number of inputs,}
	 * \\	x_{i} \text{ is the input } i \text{ and}
	 * \\	w_{ij} \text{ is the weight between neuron } i \text{ and neuron } j
	 * \f$.
	 */
	virtual void PropagateFW();
	/**
	 * Propagates through all neurons of the net beginning from the output layer. \n
	 * Calculates error deltas of neurons from current learning output and training output data. \n
	 * Also updates all weights of the net beginning from the output layer. \n
	 * The backpropagation works as described below: \n \n
	 * \f$
	 * \\	\mbox{1. Is the neuron in the output layer, it takes part of the output,}
	 * \\	\mbox{2. is the neuron in the hidden layer, the weight adaption could get calculated.}
	 * \\	\mbox{	concrete:}
	 *
	 * \\	\Delta w_{ij}(t+1)= \eta \delta_{j} x_{i} + \alpha \Delta w_{ij}(t)
	 *
	 * \\	\mbox{	with}
	 *
	 * \\	\delta_{j}=\begin{cases}
	 * 	\varphi'(\mbox{net}_{j})(t_{j}-o_{j}) & \mbox{if } j \mbox{ is an output neuron,}\\
	 * 	\varphi'(\mbox{net}_{j}) \sum_{k} \delta_{k} w_{jk} & \mbox{if } j \mbox{ is an hidden neuron.}
	 * 	\end{cases}
	 *
	 * \\ 	\mbox{	and}
	 *
	 * \\	\Delta w_{ij} \mbox{ is the change of the weight } w_{ij} \mbox{ of the connection }i\mbox{ to neuron }j\mbox{,}
	 * \\	\eta \mbox{ is the learning rate, which regulates to amount of the weight change,}
	 * \\	\delta_{j} \mbox{ is the error signal of the neuron } j mbox{,}
	 * \\	x_{i} \mbox{ is the output of the neuron } i \mbox{,}
	 * \\	t_{j} \mbox{ is the debit output of the output neuron } j \mbox{,}
	 * \\	o_{j} \mbox{ is the actual output of the output neuron } j \mbox{ und}
	 * \\	k \mbox{ is the index of the subsequent neurons of } j \mbox{.}
	 * \\ 	\Delta w_{ij}(t+1) \mbox{ is the change of the weight } w_{ij}(t+1) \mbox{ of the connection of neuron } i \mbox{ to neuron } j \mbox{ at the time point (t+1),}
	 * \\ 	\alpha \mbox{ is the influence of the momentum term } \Delta w_{ij}(t) \mbox{. Correlates with the weight change of the prior time point.}
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
	BPNet<Type, Functor> *GetSubNet(const unsigned int &iStartID, const unsigned int &iStopID);

	/**
	 * Define the learning rate, the weight decay and the momentum term.
	 */
	void Setup(const HebbianConf<Type> &config);
	
#ifdef __BPNet_ADDON
	#include __BPNet_ADDON
#endif
};

#include "BPNet.tpp"

}
