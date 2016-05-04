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

#ifndef ANSOMNETGPU_H_
#define ANSOMNETGPU_H_

#ifndef SWIG
#include "containers/TrainingSet.h"
#include "containers/2DArray.h"

#include "gpgpu/helper_cuda.h"

#include "SOMExport.h"
#include "SOMNet.h"

#include "math/Functions.h"

#include <cassert>
#include <vector>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#endif


/*
 * SOM kernels
 */
void hostSOMTraining
( std::vector<ANNGPGPU::SplittedNetExport*> &SExp,
  const ANN::TrainingSet &InputSet,
  const unsigned int &iCycles,
  const float &fSigma0,
  const float &fLearningRate0,
  const float &fConscienceRate,
  const ANN::DistFunction &pDistFunc,
  const ANN::TrainingMode &eMode
);

namespace ANNGPGPU {
    
class SOMNetGPU : public ANN::SOMNet {
private:
	std::vector<SplittedNetExport*> SplitDeviceData() const;
	void CombineDeviceData(std::vector<SplittedNetExport*> &SExp);

	/**
	 * Returns the number of cuda capable devices as integer.
	 * @return Number of cuda capable devices
	 */
	int GetCudaDeviceCount() const;

	/**
	 * Assigns a function pointer for the distance function: "GetDistFunction()->distance"
	 * This assignment is (due to CUDA related restrictions) only working, if a preimplemented distance function is used.
	 * The assignment is just for the distance, not the decay function. 
	 * Overloading of the distance function could be done at compile time manually. 
	 * Preimplemented functions: fcn_bubble_nhood, fcn_gaussian_nhood, fcn_cutgaussian_nhood, fcn_mexican_nhood, fcn_epanechicov_nhood.
	 */
	bool AssignDistanceFunction();
	
	/**
	 * Free device memory after assignment.
	 */
	bool DeassignDistanceFunction();

public:
	SOMNetGPU();
	SOMNetGPU(ANN::AbsNet *pNet);
	virtual ~SOMNetGPU();
	
	/**
	 * Trains the network with given input until iCycles is reached.
	 * @param iCycles Maximum number of training cycles.
	 * @param eMode 
	 * Value: ANRandomMode is faster, because one random input pattern is presented and a new cycle starts.\n
	 * Value: ANSerialMode means, that all input patterns are presented in order. Then a new cycle starts.
	 */
	virtual void Training(const unsigned int &iCycles = 1000, const ANN::TrainingMode &eMode = ANN::ANRandomMode);
	
	/**
	 * @brief Sets the neighborhood and decay function of the network together.
	 * @param pFCN Kind of function the net has to use while back-/propagating.
	 */
	void SetDistFunction (const ANN::DistFunction *pFCN);

	/**
	 * @brief Sets the neighborhood and decay function of the network together.
	 * @param pFCN Kind of function the net has to use while back-/propagating.
	 */
	void SetDistFunction (const ANN::DistFunction &FCN);
};

}

#endif /* ANSOMNETGPU_H_ */
