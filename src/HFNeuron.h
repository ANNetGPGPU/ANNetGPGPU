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

#ifndef ANHFNEURON_H_
#define ANHFNEURON_H_

#ifndef SWIG
#include "AbsNeuron.h"
#endif

namespace ANN {


class HFNeuron : public AbsNeuron {
public:
	HFNeuron(AbsLayer *parentLayer = NULL);
	virtual ~HFNeuron();

	/**
	 * Defines how to calculate the values of each neuron.
	 */
	virtual void CalcValue();

	/**
	 * Unused function in this hopfield net.
	 */
	virtual void AdaptEdges();
};

}

#endif /* ANHFNEURON_H_ */
