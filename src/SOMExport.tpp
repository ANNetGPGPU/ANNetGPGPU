template <class Type>
BMUExport<Type>::BMUExport() {
	fDistance 	= 0.f;
	iBMUID 		= -1;
	iDeviceID 	= -1;
	dvBMUPos 	= thrust::host_vector<Type>(0);
}

template <class Type>
BMUExport<Type>::BMUExport(Type fDist, int iUID, int iDID) {
	fDistance 	= fDist;
	iBMUID 		= iUID;
	iDeviceID 	= iDID;
}

template <class Type>
BMUExport<Type>::BMUExport(int iUID, int iDID, const thrust::host_vector<Type> &vPos) {
	fDistance 	= 0.f;
	iBMUID 		= iUID;
	iDeviceID 	= iDID;
	dvBMUPos 	= vPos;
}
///////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////
template <class Type>
SOMExport<Type>::SOMExport(const ANNGPGPU::F2DArray<Type> &mEdgeMat, 
		     const ANNGPGPU::F2DArray<Type> &mPosMat, 
		     const thrust::host_vector<Type> &vConscience, 
		     const thrust::host_vector<Type> &vSigma0,
		     const thrust::host_vector<Type> &vLearningRate) 
{
	_f2dEdges 	= mEdgeMat;
	_f2dPositions 	= mPosMat;
	_dvConscience 	= vConscience;
	_dvSigma0 	= vSigma0;
	_dvLearningRate = vLearningRate;
}

template <class Type>
void SOMExport<Type>::Clear() {
	_dvConscience.clear();
	_dvSigma0.clear();
	_dvLearningRate.clear();
}

template <class Type>
void SOMExport<Type>::SetConscience(thrust::device_vector<Type> &dvConscience) {
	_dvConscience = dvConscience;
}

template <class Type>
void SOMExport<Type>::SetSigma0(thrust::device_vector<Type> &dvSigma0) {
	_dvSigma0 = dvSigma0;
}
