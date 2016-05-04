#include "SetFcn.h"


typedef float (*pDistanceFu) (float, float);

// nonsense function to show it
__device__ static float foo(float r, float t) {
	return t*pow(r, 2);
}

__device__ pDistanceFu pOwn = foo; 

void SetFcn(ANN::DistFunction *fcn) {
	pDistanceFu hOwn;
	cudaMemcpyFromSymbol(&hOwn, pOwn, sizeof(pDistanceFu) );
	fcn->distance = hOwn;
}
