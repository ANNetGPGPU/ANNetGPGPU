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
#include "Functors.h"


using namespace ANNGPGPU;


// new reference implementation
inline ANNGPGPU::BMUExport hostGetMin(std::vector<ANNGPGPU::BMUExport> &vec) {
	assert(vec.size() > 0);
	if(vec.size() > 1) {
		std::sort(vec.begin(), vec.end() );
	}
	return *vec.begin();
}

// fast when maps are big
inline std::pair<float, unsigned int> devGetMin(const thrust::device_vector<float> &vec) {
	thrust::device_vector<float>::const_iterator d_min = thrust::min_element(vec.begin(), vec.end() );
	unsigned int iID = thrust::distance(vec.begin(), d_min);
	return std::pair<float, unsigned int>(*d_min, iID);
}

////////////////////////////////////////////////////////////////////////////////
// Data configuration
////////////////////////////////////////////////////////////////////////////////
const int MAX_GPU_COUNT = 32;
////////////////////////////////////////////////////////////////////////////////

template<class F>
int SOMNetGPU<F>::GetCudaDeviceCount() const {
	int iGPU_N 	= 0; 	// device number
	int iSM20_N 	= 0; 	// number of devices with SM >= 2.0 

	printf("Check for CUDA-capable devices\n");
	checkCudaErrors(cudaGetDeviceCount(&iGPU_N) );
	if (iGPU_N > MAX_GPU_COUNT) {
	    iGPU_N = MAX_GPU_COUNT;
	}
	printf("CUDA-capable device count: %i\n", iGPU_N);

	for(int i = 0; i < iGPU_N; i++) {
		cudaDeviceProp props;
		checkCudaErrors(cudaGetDeviceProperties(&props, i) );
		if(props.major >= 2) {
			iSM20_N++;
		}
	}
	iGPU_N = iSM20_N;
	printf("CUDA SM 2.0 capable device count: %i\n", iSM20_N);

	return iGPU_N;
}

template<class F>
std::vector<SOMExport*> SOMNetGPU<F>::SplitDeviceData() const {
	unsigned int iStart 		= 0;
	unsigned int iStop 		= 0;
	
	ANN::SOMLayer *pLayer 		= (ANN::SOMLayer *)(this->GetOPLayer());
	unsigned int iSizeOfLayer 	= pLayer->GetNeurons().size();
	unsigned int iDeviceCount 	= GetCudaDeviceCount();
	// To make things easy ..
	if(iSizeOfLayer%iDeviceCount != 0) {
		iDeviceCount = 1;
	}
	
	std::vector<SOMExport*> vRes(iDeviceCount);
	printf("Computing with %d GPUs ..\n", iDeviceCount);
	for(int i = 0; i < iDeviceCount; i++) { 
		checkCudaErrors(cudaSetDevice(i) );
	  
		iStart = i*(iSizeOfLayer/iDeviceCount);
		iStop = (i+1)*(iSizeOfLayer/iDeviceCount)-1;

		// Copy conscience information
		thrust::host_vector<float> hvConscience(iStop-iStart+1);
		thrust::host_vector<float> hvSigma0(iStop-iStart+1);
		thrust::host_vector<float> hvLearningRates(iStop-iStart+1);
		
		for(unsigned int j = 0; j <= iStop-iStart; j++) {
			hvConscience[j] = ((ANN::SOMNeuron*)pLayer->GetNeuron(j+iStart))->GetConscience();
			hvSigma0[j]     = ((ANN::SOMNeuron*)pLayer->GetNeuron(j+iStart))->GetSigma0();
			hvLearningRates[j] = ((ANN::SOMNeuron*)pLayer->GetNeuron(j+iStart))->GetLearningRate();
		}

		printf(".. Copy edges: %d/%d\n", i+1, iDeviceCount);
		F2DArray f2dEdges 	= pLayer->ExpEdgesIn(iStart, iStop);
		printf(".. Copy positions: %d/%d\n", i+1, iDeviceCount);
		F2DArray f2dPositions 	= pLayer->ExpPositions(iStart, iStop);

		// Create network export container
		vRes[i] = new SOMExport(f2dEdges, f2dPositions, hvConscience, hvSigma0, hvLearningRates);
	}
	return vRes;
}

template<class F>
void SOMNetGPU<F>::CombineDeviceData(std::vector<SOMExport*> &SExp) {
	unsigned int iStart 		= 0;
	unsigned int iStop 		= 0;
	ANN::SOMLayer *pLayer 		= (ANN::SOMLayer *)(this->GetOPLayer());
	unsigned int iSizeOfLayer 	= pLayer->GetNeurons().size();
	unsigned int iDeviceCount 	= GetCudaDeviceCount();
        
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
			this->m_pOPLayer->GetNeuron(j+iStart)->SetValue((*SExp.at(i)->dvConscience)[j]);
		}
		
		printf(".. Copy back edges: %d/%d\n", i+1, iDeviceCount);
		// Copy weights between neurons of the input and output layer
		pLayer->ImpEdgesIn(SExp.at(i)->f2dEdges, iStart, iStop);
		
		// delete old network export container
		delete SExp.at(i);
	}
	// delete old network export container
	SExp.clear();
}

template<class F>
SOMNetGPU<F>::SOMNetGPU() {
	this->m_pIPLayer 		= NULL;
	this->m_pOPLayer 		= NULL;
	this->m_pBMNeuron 		= NULL;

	this->m_iCycle 		= 0;
	this->m_fLearningRate 	= 0.5f;
	this->SetLearningRate(this->m_fLearningRate);

	this->m_iWidthI 		= 0.f;
	this->m_iHeightI 		= 0.f;
	this->m_iWidthO 		= 0.f;
	this->m_iHeightO 		= 0.f;

	// Conscience mechanism
	this->m_fConscienceRate 	= 0.f;

	this->m_fTypeFlag 	= ANN::ANNetSOM;
}

template<class F>
SOMNetGPU<F>::~SOMNetGPU() {
	unsigned int iDeviceCount = GetCudaDeviceCount();
        if(iDeviceCount) {
		checkCudaErrors(cudaDeviceReset() );
        }
}

template<class F>
void SOMNetGPU<F>::Training(const unsigned int &iCycles, const ANN::TrainingMode &eMode) {
	assert(iCycles > 0);
	this->m_iCycles = iCycles;
        
	unsigned int iDeviceCount = GetCudaDeviceCount();
	if(!iDeviceCount) {
		return;
	}
        
	if(this->GetTrainingSet() == NULL) {
		std::cout<<"No training set available!"<<std::endl;
		return;
	}

	printf("Copy memory from host to device ..\n");
	std::vector<SOMExport*> SOMExp = this->SplitDeviceData();

	StartTimer();

	printf("Calculate SOM ..\n");
	int iMin 		= 0;
	int iMax 		= this->GetTrainingSet()->GetNrElements()-1;
	int iProgCount 		= 1;

	for(this->m_iCycle = 0; this->m_iCycle < static_cast<int>(this->m_iCycles); this->m_iCycle++) {
		if(this->m_iCycles >= 10) {
			if(((this->m_iCycle+1) / (this->m_iCycles/10)) == iProgCount && (this->m_iCycle+1) % (this->m_iCycles/10) == 0) {
				std::cout<<"Current training progress calculated by the GPU is: "<<iProgCount*10.f<<"%/Step="<<this->m_iCycle+1<<std::endl;
				iProgCount++;
			}
		} 
		else {
			std::cout<<"Current training progress calculated by the CPU is: "<<(float)(this->m_iCycle+1.f)/(float)this->m_iCycles*100.f<<"%/Step="<<this->m_iCycle+1<<std::endl;
		}
		
		if(eMode == ANN::ANRandomMode) {
			unsigned int iRandID = ANN::RandInt(iMin, iMax);
			this->hostSOMPropagateBW(SOMExp, iRandID);
		}
		// The input vectors are presented to the network in serial order
		else if(eMode == ANN::ANSerialMode) {
			for(unsigned int j = 0; j < this->GetTrainingSet()->GetNrElements(); j++) {
				this->hostSOMPropagateBW(SOMExp, j);
			}
		}
	}

	printf("GPU Processing time: %f (ms)\n", GetTimer() );

	// Write edge matrix back
	std::cout<<"Copy memory from device to host .."<<std::endl;
	// Copy data from device to host
	this->CombineDeviceData(SOMExp);	
	std::cout<<".. Finished"<<std::endl;
}

/*
 * Layout of SOMEdgeF2DArray:
 * 		COL1	COL2	COL3	COL(n+1)
 * ROW1		toNeur1	toNeur1	toNeur1	..
 * ROW2		toNeur2	toNeur2	toNeur2	..
 * ROW3		toNeur3	toNeur3	toNeur3	..
 * ROW(n+1)	..		..		..
 */
template<class F>
BMUExport SOMNetGPU<F>::hostSOMFindBMNeuronID(std::vector<SOMExport*> &SExp) {
	BMUExport resBMU;
	std::vector<ANNGPGPU::BMUExport> vBMUExp(SExp.size() );

	assert(SExp.size() > 0);
	assert(vBMUExp.size() == SExp.size() );

	omp_set_num_threads(SExp.size() );  							// create as many CPU threads as there are CUDA devices
	#pragma omp parallel 									// for(int iDevID = 0; iDevID < static_cast<int>(SExp.size() ); iDevID++) {
	{
		unsigned int iDevID 	= omp_get_thread_num();
		checkCudaErrors(cudaSetDevice(iDevID) );
		
		unsigned int iWidth 	= SExp.at(iDevID)->f2dEdges.GetW();
		unsigned int iHeight 	= SExp.at(iDevID)->f2dEdges.GetH();

		assert(iWidth 	> 0);
		assert(iHeight 	> 0);

		thrust::device_vector<float> dvRes(iWidth, 0.f);

		for(int y = 0; y < static_cast<int>(iHeight); y++) {               
			thrust::transform(SExp.at(iDevID)->f2dEdges.GetRowBegin(y),
				SExp.at(iDevID)->f2dEdges.GetRowEnd(y),
				dvRes.begin(),
				dvRes.begin(),
				spowAmXpY_functor((*SExp.at(iDevID)->dvInput)[y]) );
		}

		if(this->m_fConscienceRate > 0.f) { 								// Implementation of conscience mechanism
			thrust::transform(dvRes.begin(),					// input
				dvRes.end(),							// input
				SExp.at(iDevID)->dvConscience->begin(),				// input
				dvRes.begin(),							// result
				sXmAmY_functor(1.f/(float)iWidth) );				// functor

			thrust::transform(dvRes.begin(),					// input
				dvRes.end(),							// input
				SExp.at(iDevID)->dvConscience->begin(),				// input
				SExp.at(iDevID)->dvConscience->begin(),				// result
				sAXmY_functor(this->m_fConscienceRate) );					// functor
		}

		std::pair<float, unsigned int> pCurBMUVal = devGetMin(dvRes);
		BMUExport BMU(pCurBMUVal.first, pCurBMUVal.second, iDevID);
		vBMUExp[iDevID] = BMU;
	}

	resBMU = hostGetMin(vBMUExp);
	checkCudaErrors(cudaSetDevice(resBMU.iDeviceID) );
	resBMU.dvBMUPos = SExp.at(resBMU.iDeviceID)->f2dPositions.GetSubArrayY(resBMU.iBMUID);

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
template<class F>
void SOMNetGPU<F>::hostSOMPropagateBW( std::vector<SOMExport*> &SExp, const unsigned int &iPatternID) {
	assert(iPatternID < this->GetTrainingSet()->GetNrElements() );

	// Set Input
	std::vector<float> vCurInput = this->GetTrainingSet()->GetInput(iPatternID);
	for(int iDevID = 0; iDevID < static_cast<int>(SExp.size() ); iDevID++) {
		checkCudaErrors(cudaSetDevice(iDevID) );

		thrust::device_vector<float> *p_dvInputVector = new thrust::device_vector<float>(vCurInput.size() );
		thrust::copy(vCurInput.begin(), vCurInput.end(), p_dvInputVector->begin() );
		SExp[iDevID]->dvInput = p_dvInputVector;
	}

	// Find BMNeuron 
	BMUExport BMUExp = this->hostSOMFindBMNeuronID(SExp);

	// Propagate BW SM 2.0
	omp_set_num_threads(SExp.size() );  							// create as many CPU threads as there are CUDA devices
	#pragma omp parallel 									// for(int iDev = 0; iDev < static_cast<int>(SExp.size() ); iDev++) {
	{
		unsigned int iDevID 	= omp_get_thread_num();
		checkCudaErrors(cudaSetDevice(iDevID) );
		
		unsigned int iWidth 	= SExp.at(iDevID)->f2dPositions.GetW();
		unsigned int iHeight 	= SExp.at(iDevID)->f2dPositions.GetH();

		thrust::device_vector<float> dvTmp 		(iWidth, 0.f); 			// temporary
		thrust::device_vector<float> dvLearningRate	(iWidth, 0.f);
		thrust::device_vector<float> dvInfl		(iWidth, 0.f);
		thrust::device_vector<float> dvDist		(iWidth, 0.f);

		// 1. Calc distances for all neurons to BMNeuron: Distance = sqrt(pow(x,2)+pow(y,2)+pow(z,2)+pow(n+1,2) )
		for(int y = 0; y < static_cast<int>(iHeight); y++) { 				// for each coordinate position of the neuron
			thrust::transform(
				SExp.at(iDevID)->f2dPositions.GetRowBegin(y),
				SExp.at(iDevID)->f2dPositions.GetRowEnd(y),
				dvDist.begin(),
				dvDist.begin(),
				spowAmXpY_functor(BMUExp.dvBMUPos[y]) );
		}

		// 2. calc learning rate
		thrust::device_vector<float> *dvLRate = SExp.at(iDevID)->dvLearningRate;
		thrust::transform( dvLRate->begin(),						// input
			dvLRate->end(), 							// input
			dvLearningRate.begin(), 						// result
			sm20lrate_decay_functor<F>(this->m_iCycle, this->m_iCycles) );	// functor

		// 3. Calc SigmaT^2 (already squared)
		thrust::device_vector<float> *dvSigma0 = SExp.at(iDevID)->dvSigma0;
		thrust::transform( dvSigma0->begin(),						// input
			dvSigma0->end(), 							// input
			dvTmp.begin(), 								// result
			sm20rad_decay_functor<F>(this->m_iCycle, this->m_iCycles) );		// functor

		// 4a. Calculate the influence for each neuron
		thrust::transform( dvTmp.begin(),						// input
			dvTmp.end(), 								// input
			dvDist.begin(), 							// input 2
			dvInfl.begin(), 							// result
			sm20distance_functor<F>() );			// functor

		// 4b. Multiply with learning rate
		thrust::transform( dvInfl.begin(),						// input
			dvInfl.end(), 								// input
			dvLearningRate.begin(), 						// input 2
			dvInfl.begin(), 							// result
			thrust::multiplies<float>() );						// functor

		// 5. Only handle neurons in radius:
		// 5a. Make stencil
		thrust::transform( dvDist.begin(), 						// input
			dvDist.end(),								// input
			dvTmp.begin(),								// input 2
			dvTmp.begin(), 								// result
			thrust::less<float>() 							// functor
		);
		// 5b. Use stencil to modify only neurons inside the radius
		iWidth 	= SExp.at(iDevID)->f2dEdges.GetW();
		iHeight = SExp.at(iDevID)->f2dEdges.GetH();
		for(int y = 0; y < static_cast<int>(iHeight); y++) {				// for each edge of the neuron
			thrust::transform_if( SExp.at(iDevID)->f2dEdges.GetRowBegin(y),		// input 1
				SExp.at(iDevID)->f2dEdges.GetRowEnd(y), 			// input 1
				dvInfl.begin(),							// input 2
				dvTmp.begin(),							// stencil
				SExp.at(iDevID)->f2dEdges.GetRowBegin(y), 			// result
				hebbian_functor((*SExp.at(iDevID)->dvInput)[y]), // functor
				thrust::identity<int>() ); 					// predicate
		}
	}
}

template SOMNetGPU<ANN::functor_bubble>::SOMNetGPU();
template SOMNetGPU<ANN::functor_gaussian>::SOMNetGPU();
template SOMNetGPU<ANN::functor_cutgaussian>::SOMNetGPU();
template SOMNetGPU<ANN::functor_epanechicov>::SOMNetGPU();
template SOMNetGPU<ANN::functor_mexican>::SOMNetGPU();
// Include user implementations
#ifdef SOMNETGPU_EXTENSIONS
#include SOMNETGPU_EXTENSIONS
#endif
