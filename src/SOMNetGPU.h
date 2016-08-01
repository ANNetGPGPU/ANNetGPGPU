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
#include "SOMExport.h"
#include "SOMNet.h"

#include <cfloat>
#include <cmath>
#include <algorithm>
#include <omp.h>
#include <vector>
#endif


namespace ANNGPGPU {

template <class Type, class Functor>
class SOMNetGPU : public ANN::SOMNet<Type, Functor> {
private:
	int m_iDeviceCount = GetCudaDeviceCount();
	
	std::vector<SOMExport<Type>*> SplitDeviceData() const;
	void CombineDeviceData(std::vector<SOMExport<Type>*> &SExp);

	/**
	 * Returns the number of cuda capable devices as integer.
	 * @return Number of cuda capable devices
	 */
	int GetCudaDeviceCount() const;
	BMUExport<Type> hostSOMFindBMNeuronID(std::vector<SOMExport<Type>*> &SExp);
	void hostSOMPropagateBW( std::vector<SOMExport<Type>*> &SExp, const unsigned int &iPatternID);

public:
	SOMNetGPU();
	virtual ~SOMNetGPU();
	
	/**
	 * Trains the network with given input until iCycles is reached.
	 * @param iCycles Maximum number of training cycles.
	 * @param eMode 
	 * Value: ANRandomMode is faster, because one random input pattern is presented and a new cycle starts.\n
	 * Value: ANSerialMode means, that all input patterns are presented in order. Then a new cycle starts.
	 */
	virtual void Training(const unsigned int &iCycles = 1000, const ANN::TrainingMode &eMode = ANN::ANRandomMode);
	
#ifdef __SOMNetGPU_ADDON
	#include __SOMNetGPU_ADDON
#endif
};

//#include "SOMNetGPU.cu"

};

