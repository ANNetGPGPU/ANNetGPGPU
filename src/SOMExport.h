#pragma once

#ifndef SWIG
#include <vector>
#include "containers/2DArrayGPU.h"

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#endif


namespace ANNGPGPU {

template <class Type>
struct BMUExport {
// VARIABLES
	Type fDistance;
	int iBMUID;
	int iDeviceID;
	thrust::host_vector<Type> dvBMUPos;
//FUNCTIONS
	BMUExport();
	BMUExport(int iUID, int iDID, const thrust::host_vector<Type> &vPos);
	BMUExport(Type fDist, int iUID, int iDID);
	
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

template <class Type>
struct SOMExport {
// VARIABLES
	ANNGPGPU::F2DArray<Type> _f2dEdges;
	ANNGPGPU::F2DArray<Type> _f2dPositions;
	thrust::device_vector<Type> _dvConscience;
        thrust::device_vector<Type> _dvSigma0;
	thrust::device_vector<Type> _dvLearningRate;
	
//FUNCTIONS
	SOMExport(const ANNGPGPU::F2DArray<Type> &mEdgeMat, 
		  const ANNGPGPU::F2DArray<Type> &mPosMat, 
		  const thrust::host_vector<Type> &vConscience, 
		  const thrust::host_vector<Type> &vSigma0,
		  const thrust::host_vector<Type> &vLearningRate);
        
	void SetConscience(thrust::device_vector<Type> &dvConscience);
	void SetSigma0(thrust::device_vector<Type> &dvSigma0);
	
	void Clear();
	
#ifdef __SOMExport_ADDON
	#include __SOMExport_ADDON
#endif
};

#include "SOMExport.tpp"

}
