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

#include <cfloat>
#include <cmath>
#include <algorithm>
#include <omp.h>
#include <gpgpu/timer.h>

#include <thrust/extrema.h>
#include <thrust/distance.h>
#include <thrust/device_vector.h>

#include "gpgpu/helper_cuda.h"

#include "SOMNetGPU.h"
#include "math/Functors.h"


////////////////////////////////////////////////////////////////////////////////
// Data configuration
////////////////////////////////////////////////////////////////////////////////
const int MAX_GPU_COUNT = 32;


namespace ANNGPGPU {


// new reference implementation
template<class Type>
inline ANNGPGPU::BMUExport<Type> hostGetMin(std::vector<ANNGPGPU::BMUExport<Type>> &vec) {
	assert(vec.size() > 0);
	if(vec.size() > 1) {
		std::sort(vec.begin(), vec.end() );
	}
	return *vec.begin();
}

template<class Type>
inline std::pair<Type, unsigned int> devGetMin(const thrust::device_vector<Type> &vec) {
	const_iterator<Type> d_min = thrust::min_element(vec.begin(), vec.end() );
	unsigned int iID = thrust::distance(vec.begin(), d_min);
	return std::pair<Type, unsigned int>(*d_min, iID);
}

template<class Type, class Functor>
int SOMNetGPU<Type, Functor>::GetCudaDeviceCount() const {
	int iGPU_N 	= 0; 	// device number
	int iSM20_N 	= 0; 	// number of devices with SM >= 2.0 

	ANN::printf("Check for CUDA-capable devices: ");
	checkCudaErrors(cudaGetDeviceCount(&iGPU_N) );
	if (iGPU_N > MAX_GPU_COUNT) {
	    iGPU_N = MAX_GPU_COUNT;
	}
	// Check hardware level
	for(int i = 0; i < iGPU_N; i++) {
		cudaDeviceProp props;
		checkCudaErrors(cudaGetDeviceProperties(&props, i) );
		if(props.major >= 2) {
			iSM20_N++;
		}
	}
	iGPU_N = iSM20_N;
	
	if(!iGPU_N) {
		ANN::printf("FAIL\nProgram will be terminated because no compatible hardware found\n");
		exit(-1); // No GPU found
	}

	ANN::printf("SUCCESS\nCUDA SM 2.0 capable device count: %i\n", iGPU_N);
	return iGPU_N;
}

template<class Type, class Functor>
std::vector<SOMExport<Type>*> SOMNetGPU<Type, Functor>::SplitDeviceData() const {
	unsigned int iStart = 0;
	unsigned int iStop = 0;
	ANN::SOMLayer<Type> *pLayer = (ANN::SOMLayer<Type> *)(this->GetOPLayer());
	unsigned int iSizeOfLayer = pLayer->GetNeurons().size();
	unsigned int iDeviceCount = m_iDeviceCount;

	// To make things easy ..
	if(iSizeOfLayer%iDeviceCount != 0) {
		iDeviceCount = 1;
	}
	
	std::vector<SOMExport<Type>*> vRes(iDeviceCount);
	for(int i = 0; i < iDeviceCount; i++) { 
		checkCudaErrors(cudaSetDevice(i) );
	  
		iStart = i*(iSizeOfLayer/iDeviceCount);
		iStop = (i+1)*(iSizeOfLayer/iDeviceCount)-1;

		// Copy conscience information
		thrust::host_vector<Type> hvConscience(iStop-iStart+1);
		thrust::host_vector<Type> hvSigma0(iStop-iStart+1);
		thrust::host_vector<Type> hvLearningRates(iStop-iStart+1);
		
		for(unsigned int j = 0; j <= iStop-iStart; j++) {
			hvConscience[j] = ((ANN::SOMNeuron<Type>*)pLayer->GetNeuron(j+iStart))->GetConscience();
			hvSigma0[j]     = ((ANN::SOMNeuron<Type>*)pLayer->GetNeuron(j+iStart))->GetSigma0();
			hvLearningRates[j] = ((ANN::SOMNeuron<Type>*)pLayer->GetNeuron(j+iStart))->GetLearningRate();
		}
		ANN::printf(".. Copy edges from host to device: %d/%d\n", i+1, iDeviceCount);

		ANN::F2DArray<Type> h2dEdges = pLayer->ExpEdgesIn(iStart, iStop);
		F2DArray<Type> d2dEdges(h2dEdges.GetW(), h2dEdges.GetH(), h2dEdges.ToDevice() );

		ANN::printf(".. Copy positions from host to device: %d/%d\n", i+1, iDeviceCount);
		ANN::F2DArray<Type> h2dPositions = pLayer->ExpPositions(iStart, iStop);
		F2DArray<Type> d2dPositions(h2dPositions.GetW(), h2dPositions.GetH(), h2dPositions.ToDevice() );
		
		// Create network export container
		vRes[i] = new SOMExport<Type>(d2dEdges, d2dPositions, hvConscience, hvSigma0, hvLearningRates);
	}
	return vRes;
}

template<class Type, class Functor>
void SOMNetGPU<Type, Functor>::CombineDeviceData(std::vector<SOMExport<Type>*> &SExp) {
	unsigned int iStart = 0;
	unsigned int iStop = 0;
	ANN::SOMLayer<Type> *pLayer = (ANN::SOMLayer<Type> *)(this->GetOPLayer());
	unsigned int iSizeOfLayer = pLayer->GetNeurons().size();
	unsigned int iDeviceCount = m_iDeviceCount;
        
	if(!iDeviceCount) {
		return;
	}
        
	// To make things easy ..
	if(iSizeOfLayer%iDeviceCount != 0) {
		iDeviceCount = 1;
	}

	for(int i = 0; i < iDeviceCount; i++) {
		checkCudaErrors(cudaSetDevice(i) );

		iStart = i*(iSizeOfLayer/iDeviceCount);
		iStop = (i+1)*(iSizeOfLayer/iDeviceCount)-1;

		// Copy back conscience
		for(unsigned int j = 0; j <= iStop-iStart; j++) {
			this->m_pOPLayer->GetNeuron(j+iStart)->SetValue((SExp.at(i)->_dvConscience)[j]);
		}

		ANN::printf(".. Copy edges from device to host: %d/%d\n", i+1, iDeviceCount);
		// Copy weights between neurons of the input and output layer
		pLayer->ImpEdgesIn(SExp.at(i)->_f2dEdges, iStart, iStop);
		
		// delete old network export container
		delete SExp.at(i);
	}
	// delete old network export container
	SExp.clear();
}

template<class Type, class Functor>
SOMNetGPU<Type, Functor>::SOMNetGPU() {
	this->m_pIPLayer = NULL;
	this->m_pOPLayer = NULL;
	this->m_pBMNeuron = NULL;

	this->m_iCycle = 0;

	this->m_iWidthI = 0.f;
	this->m_iHeightI = 0.f;
	this->m_iWidthO = 0.f;
	this->m_iHeightO = 0.f;

	// Conscience mechanism
	this->m_fTypeFlag = ANN::ANNetSOM;
}

template<class Type, class Functor>
SOMNetGPU<Type, Functor>::~SOMNetGPU() {
	unsigned int iDeviceCount = m_iDeviceCount;
        if(iDeviceCount) {
		checkCudaErrors(cudaDeviceReset() );
        }
}

template<class Type, class Functor>
void SOMNetGPU<Type, Functor>::Training(const unsigned int &iCycles, const ANN::TrainingMode &eMode) {
	assert(iCycles > 0);
	this->m_iCycles = iCycles;
        
	unsigned int iDeviceCount = m_iDeviceCount;
	if(!iDeviceCount) {
		return;
	}
        
	if(this->GetTrainingSet() == NULL) {
		std::cout<<"No training set available!"<<std::endl;
		return;
	}

	ANN::printf("Copy memory from host to device ..\n");
	std::vector<SOMExport<Type>*> SOMExp = this->SplitDeviceData();

	StartTimer();
	ANN::printf("Calculate SOM ..\n");
	int iMin = 0;
	int iMax = this->GetTrainingSet()->GetNrElements()-1;
	int iProgCount = 1;

	for(this->m_iCycle = 0; this->m_iCycle < static_cast<int>(this->m_iCycles); this->m_iCycle++) {
		if(this->m_iCycles >= 10) {
			if(((this->m_iCycle+1) / (this->m_iCycles/10)) == iProgCount && (this->m_iCycle+1) % (this->m_iCycles/10) == 0) {
				ANN::printf("Current training progress calculated by the GPU: %f%%/Step: %d/%d\n", 
							this->m_iCycle+1, 
							this->m_iCycle+1, 
							this->m_iCycles);
				iProgCount++;
			}
		} 
		else {
			ANN::printf("Current training progress calculated by the GPU: %f%%/Step: %d/%d\n", 
						(float)(this->m_iCycle+1.f)/(float)this->m_iCycles*100.f, 
						this->m_iCycle+1, 
						this->m_iCycles);
		}
		
		if(eMode == ANN::ANRandomMode) {
			unsigned int iRandID = ANN::GetRandInt(iMin, iMax);
			this->hostSOMPropagateBW(SOMExp, iRandID);
		}
		// The input vectors are presented to the network in serial order
		else if(eMode == ANN::ANSerialMode) {
			for(unsigned int j = 0; j < this->GetTrainingSet()->GetNrElements(); j++) {
				this->hostSOMPropagateBW(SOMExp, j);
			}
		}
	}
	ANN::printf("GPU Processing time: %f (ms)\n", GetTimer() );

	// Write edge matrix back
	ANN::printf("Copy memory from device to host..");
	this->CombineDeviceData(SOMExp);
	
	// Clean up after memory allocation
	for(int i = 0; i < SOMExp.size(); i++) {
		SOMExp.at(i)->Clear();
	}
	SOMExp.clear();
	
	// End with an output
	ANN::printf(" finished\n");
}

/*
 * Layout of SOMEdgeF2DArray:
 * 		COL1	COL2	COL3	COL(n+1)
 * ROW1		toNeur1	toNeur1	toNeur1	..
 * ROW2		toNeur2	toNeur2	toNeur2	..
 * ROW3		toNeur3	toNeur3	toNeur3	..
 * ROW(n+1)	..		..		..
 */
template<class Type, class Functor>
BMUExport<Type> SOMNetGPU<Type, Functor>::hostSOMFindBMNeuronID(std::vector<SOMExport<Type>*> &SExp, const thrust::device_vector<Type> &dvInput) {
	BMUExport<Type> resBMU;
	std::vector<ANNGPGPU::BMUExport<Type>> vBMUExp(SExp.size() );

	assert(SExp.size() > 0);
	assert(vBMUExp.size() == SExp.size() );

	omp_set_num_threads(SExp.size() );  	// create as many CPU threads as there are CUDA devices
	#pragma omp parallel 			// for(int iDevID = 0; iDevID < static_cast<int>(SExp.size() ); iDevID++) {
	{
		unsigned int iDevID 	= omp_get_thread_num();
		checkCudaErrors(cudaSetDevice(iDevID) );
		
		unsigned int iWidth 	= SExp.at(iDevID)->_f2dEdges.GetW();
		unsigned int iHeight 	= SExp.at(iDevID)->_f2dEdges.GetH();

		assert(iWidth 	> 0);
		assert(iHeight 	> 0);

		thrust::device_vector<Type> dvRes(iWidth, 0.f);

		for(int y = 0; y < static_cast<int>(iHeight); y++) {               
			thrust::transform(SExp.at(iDevID)->_f2dEdges.GetRowBegin(y),
				SExp.at(iDevID)->_f2dEdges.GetRowEnd(y),
				dvRes.begin(),
				dvRes.begin(),
				spowAmXpY_functor<Type>(dvInput[y]) );
		}

		if(this->m_fConscienceRate > 0.f) { 								// Implementation of conscience mechanism
			thrust::transform(dvRes.begin(),					// input
				dvRes.end(),							// input
				SExp.at(iDevID)->_dvConscience.begin(),				// input
				dvRes.begin(),							// result
				sXmAmY_functor<Type>(1.f/(Type)iWidth) );				// functor

			thrust::transform(dvRes.begin(),					// input
				dvRes.end(),							// input
				SExp.at(iDevID)->_dvConscience.begin(),				// input
				SExp.at(iDevID)->_dvConscience.begin(),				// result
				sAXmY_functor<Type>(this->m_fConscienceRate) );					// functor
		}

		std::pair<Type, unsigned int> pCurBMUVal = devGetMin(dvRes);
		BMUExport<Type> BMU(pCurBMUVal.first, pCurBMUVal.second, iDevID);
		vBMUExp[iDevID] = BMU;
	}

	resBMU = hostGetMin(vBMUExp);
	checkCudaErrors(cudaSetDevice(resBMU.iDeviceID) );
	resBMU.dvBMUPos = SExp.at(resBMU.iDeviceID)->_f2dPositions.GetSubArrayY(resBMU.iBMUID);

	return resBMU;
}

/*
 * Layout of SOMPositionF2DArray:
 * 		COL1	COL2	COL3	COL(n+1)
 * ROW1		Xpos	Xpos	Xpos	..
 * ROW2		Ypos	Ypos	Ypos	..
 * ROW3		Zpos	Zpos	Zpos	..
 * ROW(n+1)	..		..		..		..
 */
template<class Type, class Functor>
void SOMNetGPU<Type, Functor>::hostSOMPropagateBW( std::vector<SOMExport<Type>*> &SExp, const unsigned int &iPatternID) {
	assert(iPatternID < this->GetTrainingSet()->GetNrElements() );

	// Set Input
	std::vector<Type> vCurInput = this->GetTrainingSet()->GetInput(iPatternID);
	thrust::device_vector<Type> dvInput(vCurInput.size() );
	thrust::copy(vCurInput.begin(), vCurInput.end(), dvInput.begin() );

	// Find BMNeuron 
	BMUExport<Type> BMUExp = this->hostSOMFindBMNeuronID(SExp, dvInput);

	// Propagate BW SM 2.0
	omp_set_num_threads(SExp.size() );  	// create as many CPU threads as there are CUDA devices
	#pragma omp parallel 			// for(int iDev = 0; iDev < static_cast<int>(SExp.size() ); iDev++) {
	{
		unsigned int iDevID 	= omp_get_thread_num();
		checkCudaErrors(cudaSetDevice(iDevID) );
		
		unsigned int iWidth 	= SExp.at(iDevID)->_f2dPositions.GetW();
		unsigned int iHeight 	= SExp.at(iDevID)->_f2dPositions.GetH();

		thrust::device_vector<Type> dvTmp 		(iWidth, 0.f); 			// temporary
		thrust::device_vector<Type> dvLearningRate	(iWidth, 0.f);
		thrust::device_vector<Type> dvInfl		(iWidth, 0.f);
		thrust::device_vector<Type> dvDist		(iWidth, 0.f);

		// 1a. Calc distances for all neurons to BMNeuron: Distance = sqrt(pow(x,2)+pow(y,2)+pow(z,2)+pow(n+1,2) )
		for(int y = 0; y < static_cast<int>(iHeight); y++) { 				// for each coordinate position of the neuron
			thrust::transform(
				SExp.at(iDevID)->_f2dPositions.GetRowBegin(y),
				SExp.at(iDevID)->_f2dPositions.GetRowEnd(y),
				dvDist.begin(),
				dvDist.begin(),
				spowAmXpY_functor<Type>(BMUExp.dvBMUPos[y]) );
		}
		
		// 1b. Calc the square root
		thrust::transform(dvDist.begin(), dvDist.end(), dvDist.begin(), square_root<Type>());

		// 2. calc learning rate
		thrust::device_vector<Type> *dvLRate = &(SExp.at(iDevID)->_dvLearningRate);
		thrust::transform( dvLRate->begin(),						// input
			dvLRate->end(), 							// input
			dvLearningRate.begin(), 						// result
			lrate_decay_functor<Type, Functor>(this->m_iCycle, this->m_iCycles) );	// functor

		// 3. Calc SigmaT^2 (already squared)
		thrust::device_vector<Type> *dvSigma0 = &SExp.at(iDevID)->_dvSigma0;
		thrust::transform( dvSigma0->begin(),						// input
			dvSigma0->end(), 							// input
			dvTmp.begin(), 								// result
			rad_decay_functor<Type, Functor>(this->m_iCycle, this->m_iCycles) );	// functor

		// 4a. Calculate the influence for each neuron
		thrust::transform( dvTmp.begin(),						// input
			dvTmp.end(), 								// input
			dvDist.begin(), 							// input 2
			dvInfl.begin(), 							// result
			distance_functor<Type, Functor>() );					// functor

		// 4b. Multiply with learning rate
		thrust::transform( dvInfl.begin(),						// input
			dvInfl.end(), 								// input
			dvLearningRate.begin(), 						// input 2
			dvInfl.begin(), 							// result
			thrust::multiplies<Type>() );						// functor

		// 5. Only handle neurons in radius:
		// 5a. Make stencil
		thrust::transform( dvDist.begin(), 						// input
			dvDist.end(),								// input
			dvTmp.begin(),								// input 2
			dvTmp.begin(), 								// result
			thrust::less<Type>() 							// functor
		);
		// 5b. Use stencil to modify only neurons inside the radius
		iWidth 	= SExp.at(iDevID)->_f2dEdges.GetW();
		iHeight = SExp.at(iDevID)->_f2dEdges.GetH();
		for(int y = 0; y < static_cast<int>(iHeight); y++) {				// for each edge of the neuron
			thrust::transform_if( SExp.at(iDevID)->_f2dEdges.GetRowBegin(y),		// input 1
				SExp.at(iDevID)->_f2dEdges.GetRowEnd(y), 			// input 1
				dvInfl.begin(),							// input 2
				dvTmp.begin(),							// stencil
				SExp.at(iDevID)->_f2dEdges.GetRowBegin(y), 			// result
				som_hebbian_functor<Type, Functor>(dvInput[y]), // functor
				thrust::identity<int>() ); 					// predicate
		}
	}
}

template SOMNetGPU<float, ANN::functor_bubble<float>>::SOMNetGPU();
template SOMNetGPU<float, ANN::functor_gaussian<float>>::SOMNetGPU();
template SOMNetGPU<float, ANN::functor_cutgaussian<float>>::SOMNetGPU();
template SOMNetGPU<float, ANN::functor_epanechicov<float>>::SOMNetGPU();
template SOMNetGPU<float, ANN::functor_mexican<float>>::SOMNetGPU();

template SOMNetGPU<double, ANN::functor_bubble<double>>::SOMNetGPU();
template SOMNetGPU<double, ANN::functor_gaussian<double>>::SOMNetGPU();
template SOMNetGPU<double, ANN::functor_cutgaussian<double>>::SOMNetGPU();
template SOMNetGPU<double, ANN::functor_epanechicov<double>>::SOMNetGPU();
template SOMNetGPU<double, ANN::functor_mexican<double>>::SOMNetGPU();
};

#ifdef __SOMNetGPU_INSTANCES
	#include __SOMNetGPU_INSTANCES
#endif
