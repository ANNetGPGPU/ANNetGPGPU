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
#include "SOMExport.h"
#include "SOMNet.h"

#include <cfloat>
#include <map>
#include <cmath>
#include <algorithm>
#include <omp.h>
#include <vector>
#endif


namespace ANNGPGPU {

template <class T> class Centroid;
	
template <class Type, class Functor>
class SOMNetGPU : public ANN::SOMNet<Type, Functor> {
private:
	int32_t m_iDeviceCount = GetCudaDeviceCount();
	std::vector<SOMExport<Type>> SplitDeviceData() const;
	void CombineDeviceData(std::vector<SOMExport<Type>> &SExp);

	/**
	 * Returns the number of cuda capable devices as integer.
	 * @return Number of cuda capable devices
	 */
	int32_t GetCudaDeviceCount() const;
	Centroid<Type> hostSOMFindBMNeuronID(std::vector<SOMExport<Type>> &SExp, const thrust::device_vector<Type> &dvInput);
	void hostSOMPropagateBW( std::vector<SOMExport<Type>> &SExp, const int32_t &iPatternID);

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
	void Training(const uint32_t &iCycles = 1000, const ANN::TrainingMode &eMode = ANN::ANRandomMode) override;
	
	/**
	 * @brief Clustering results of the network.
	 * @return std::vector<Centroid> Iterates through the input list and calcs the euclidean distance based on the BMU.
	 */
	virtual std::vector<Centroid<Type>> FindCentroidsGPU();
	
#ifdef __SOMNetGPU_ADDON
	#include __SOMNetGPU_ADDON
#endif
};

//#include "SOMNetGPU.cu"

};

