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

#ifndef ANHFLAYER_H_
#define ANHFLAYER_H_

#ifndef SWIG
#include "AbsLayer.h"
#endif

namespace ANN {

class HFNeuron;


class HFLayer : public AbsLayer {
private:
	unsigned int m_iWidth;
	unsigned int m_iHeight;

public:
	HFLayer();
	HFLayer(const unsigned int &iWidth, const unsigned int &iHeight);
	virtual ~HFLayer();

	/**
	 * Resizes the layer. Deletes old neurons and adds new ones (initialized with random values).
	 * @param iSize New number of neurons.
	 * @param iShiftID When called each neuron created gets an ID defined in this function plus the value of iShiftID. Used for example in ANHFLayer, when creating 2d matrix.
	 */
	virtual void Resize(const unsigned int &iSize);
	/**
	 *
	 */
	virtual void AddNeurons(const unsigned int &iSize);

	/**
	 * @return Returns the width of the layer.
	 */
	unsigned int GetWidth();

	/**
	 * @return Returns the height of the layer.
	 */
	unsigned int GetHeight();

	/**
	 * Resizes the layer. Deletes old neurons and adds new ones (initialized with random values).
	 * @param iWidth Width of map.
	 * @param iHeight Height of map.
	 * @param iShiftID Value which has to get added to the ID of each neuron.
	 */
	void Resize(const unsigned int &iWidth, const unsigned int &iHeight);

	/**
	 * Pointer to the neuron at index.
	 * @return Returns the pointer of the neuron at point in layer
	 * @param iX Column in the layer
	 * @param iY Line in the layer
	 */
	HFNeuron *GetNeuron(const unsigned int &iX, const unsigned int &iY) const;

	/**
	 * A hopfield net only consists of one layer.
	 * This functions connects each neuron of this layer with all the other ones.
	 * @param bAllowAdapt allows the change of the weights between both layers.
	 */
	void ConnectLayer(bool bAllowAdapt = true);

	/**
	 * A hopfield net only consists of one layer.
	 * This functions connects each neuron of this layer with all the other ones
	 * and sets the connection to a value specified in pEdges.
	 * @param pEdges is a pointer to a one dimensional array saving the values of the connections between all neurons.
	 * @param bAllowAdapt allows the change of the weights between both layers.
	 */
	void ConnectLayer(const float *pEdges, bool bAllowAdapt = true);

	/**
	 * This function is running through all connections between all neurons and sets them to zero.
	 */
	void ClearWeights();
};

}

#endif /* ANHFLAYER_H_ */
