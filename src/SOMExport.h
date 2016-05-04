#ifndef ANSOMEXP_H_
#define ANSOMEXP_H_

#ifndef SWIG
#include <vector>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include "containers/2DArrayGPU.h"
#endif


namespace ANNGPGPU {

class BMUExport {
public:
// VARIABLES
	float fDistance;
	int iBMUID;
	int iDeviceID;
	thrust::host_vector<float> dvBMUPos;
//FUNCTIONS
	BMUExport();
	BMUExport(int iUID, int iDID, const thrust::host_vector<float> &vPos);
	BMUExport(float fDist, int iUID, int iDID);
	
	bool operator<(const BMUExport &rhs) const {
		return fDistance < rhs.fDistance;
	}
	bool operator>(const BMUExport &rhs) const {
		return fDistance > rhs.fDistance;
	}
	bool operator<=(const BMUExport &rhs) const {
		return fDistance <= rhs.fDistance;
	}
	bool operator>=(const BMUExport &rhs) const {
		return fDistance >= rhs.fDistance;
	}
	bool operator==(const BMUExport &rhs) const {
		return fDistance == rhs.fDistance;
	}
	bool operator!=(const BMUExport &rhs) const {
		return fDistance != rhs.fDistance;
	}
};

class SplittedNetExport {
public:
// VARIABLES
	ANNGPGPU::F2DArray f2dEdges;
	ANNGPGPU::F2DArray f2dPositions;
	thrust::device_vector<float> *dvConscience;
	thrust::device_vector<float> *dvInput;
//FUNCTIONS
	SplittedNetExport(const ANNGPGPU::F2DArray &mEdgeMat, const ANNGPGPU::F2DArray &mPosMat, const thrust::host_vector<float> &vConscience);
	void SetInput(thrust::device_vector<float> *p_dvInput);
	void SetConscience(thrust::device_vector<float> *p_dvConscience);
};

}

#endif
