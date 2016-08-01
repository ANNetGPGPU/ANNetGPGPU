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
	f2dEdges 	= mEdgeMat;
	f2dPositions 	= mPosMat;
	
	dvInput 	= NULL;
	dvConscience 	= NULL;
	dvSigma0 	= NULL;
	dvLearningRate  = NULL;
	
	dvConscience 	= new thrust::device_vector<Type>;
	*dvConscience 	= vConscience;
	
	dvSigma0    	= new thrust::device_vector<Type>;
	*dvSigma0 	= vSigma0;
	
	dvLearningRate  = new thrust::device_vector<Type>;
	*dvLearningRate = vLearningRate;
}

template <class Type>
void SOMExport<Type>::SetInput(thrust::device_vector<Type> *p_dvInput) {
	assert(p_dvInput != NULL);

	if(dvInput != NULL) {
		delete dvInput;
		dvInput = NULL;
	}
	if(dvInput == NULL && p_dvInput != NULL) {
		dvInput = p_dvInput;
	}
}

template <class Type>
void SOMExport<Type>::SetConscience(thrust::device_vector<Type> *p_dvConscience) {
	assert(p_dvConscience != NULL);

	if(dvConscience != NULL) {
		delete dvConscience;
		dvConscience = NULL;
	}
	if(dvConscience == NULL && p_dvConscience != NULL) {
		dvConscience = p_dvConscience;
	}
}

template <class Type>
void SOMExport<Type>::SetSigma0(thrust::device_vector<Type> *p_dvSigma0) {
	assert(p_dvSigma0 != NULL);

	if(dvSigma0 != NULL) {
		delete dvSigma0;
		dvSigma0 = NULL;
	}
	if(dvSigma0 == NULL && p_dvSigma0 != NULL) {
		dvSigma0 = p_dvSigma0;
	}
}
