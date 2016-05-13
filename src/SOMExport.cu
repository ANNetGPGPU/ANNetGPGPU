#include "SOMExport.h"

using namespace ANNGPGPU;

 
BMUExport::BMUExport() {
	fDistance 		= 0.f;
	iBMUID 			= -1;
	iDeviceID 		= -1;
	dvBMUPos 		= thrust::host_vector<float>(0);
}

BMUExport::BMUExport(float fDist, int iUID, int iDID) {
	fDistance 		= fDist;
	iBMUID 			= iUID;
	iDeviceID 		= iDID;
}

BMUExport::BMUExport(int iUID, int iDID, const thrust::host_vector<float> &vPos) {
	fDistance 		= 0.f;
	iBMUID 			= iUID;
	iDeviceID 		= iDID;
	dvBMUPos 		= vPos;
}
///////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////
SOMExport::SOMExport(const ANNGPGPU::F2DArray &mEdgeMat, 
		     const ANNGPGPU::F2DArray &mPosMat, 
		     const thrust::host_vector<float> &vConscience, 
		     const thrust::host_vector<float> &vSigma0,
		     const thrust::host_vector<float> &vLearningRate) 
{
	f2dEdges 	= mEdgeMat;
	f2dPositions 	= mPosMat;
	
	dvInput 	= NULL;
	dvConscience 	= NULL;
	dvSigma0 	= NULL;
	dvLearningRate  = NULL;
	
	dvConscience 	= new thrust::device_vector<float>;
	*dvConscience 	= vConscience;
	
	dvSigma0    	= new thrust::device_vector<float>;
	*dvSigma0 	= vSigma0;
	
	dvLearningRate  = new thrust::device_vector<float>;
	*dvLearningRate = vLearningRate;
}

void SOMExport::SetInput(thrust::device_vector<float> *p_dvInput) {
	assert(p_dvInput != NULL);

	if(dvInput != NULL) {
		delete dvInput;
		dvInput = NULL;
	}
	if(dvInput == NULL && p_dvInput != NULL) {
		dvInput = p_dvInput;
	}
}

void SOMExport::SetConscience(thrust::device_vector<float> *p_dvConscience) {
	assert(p_dvConscience != NULL);

	if(dvConscience != NULL) {
		delete dvConscience;
		dvConscience = NULL;
	}
	if(dvConscience == NULL && p_dvConscience != NULL) {
		dvConscience = p_dvConscience;
	}
}

void SOMExport::SetSigma0(thrust::device_vector<float> *p_dvSigma0) {
	assert(p_dvSigma0 != NULL);

	if(dvSigma0 != NULL) {
		delete dvSigma0;
		dvSigma0 = NULL;
	}
	if(dvSigma0 == NULL && p_dvSigma0 != NULL) {
		dvSigma0 = p_dvSigma0;
	}
}
