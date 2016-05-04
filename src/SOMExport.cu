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
SplittedNetExport::SplittedNetExport(const ANNGPGPU::F2DArray &mEdgeMat, const ANNGPGPU::F2DArray &mPosMat, const thrust::host_vector<float> &vConscience) {
	f2dEdges 	= mEdgeMat;
	f2dPositions 	= mPosMat;
	
	dvInput 	= NULL;
	dvConscience 	= NULL;
	
	dvConscience 	= new thrust::device_vector<float>;
	*dvConscience 	= vConscience;
}

void SplittedNetExport::SetInput(thrust::device_vector<float> *p_dvInput) {
	assert(p_dvInput != NULL);

	if(dvInput != NULL) {
		delete dvInput;
		dvInput = NULL;
	}
	if(dvInput == NULL && p_dvInput != NULL) {
		dvInput = p_dvInput;
	}
}

void SplittedNetExport::SetConscience(thrust::device_vector<float> *p_dvConscience) {
	assert(p_dvConscience != NULL);

	if(dvConscience != NULL) {
		delete dvConscience;
		dvConscience = NULL;
	}
	if(dvConscience == NULL && p_dvConscience != NULL) {
		dvConscience = p_dvConscience;
	}
}
