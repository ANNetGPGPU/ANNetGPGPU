#ifndef ANFUNCTORS_H_
#define ANFUNCTORS_H_

#ifndef SWIG
#include "../math/Functions.h"
#endif


struct sAXpY_functor { // Y <- A * X + Y
    float a;

    sAXpY_functor(float _a) : a(_a) {}

    __host__ __device__
	float operator()(float x, float y) {
		return a * x + y;
	}
};

struct sAX_functor { // Y <- A * X
    float a;

    sAX_functor(float _a) : a(_a) {}

    __host__ __device__
	float operator()(float x) {
		return a * x;
	}
};

struct sAXmY_functor { // Y <- A * (X - Y)
	float a;

	sAXmY_functor(float _a) : a(_a) {}

	__host__ __device__
	float operator()(float x, float y) { 
		return a * (x - y);
	}
};

struct sXmAmY_functor { // Y <- X - (A - Y)
	float a;

	sXmAmY_functor(float _a) : a(_a) {}

	__host__ __device__
	float operator()(float x, float y) { 
		return x - (a - y);
	}
};

struct spowAmXpY_functor { // Y <- (A-X)^2 + Y
	float a;

	spowAmXpY_functor(float _a) : a(_a) {}

	__host__ __device__
	float operator()(float x, float y) { 
		return pow(a-x, 2) + y;
	}
};
//////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////
struct sm13bubble_functor {
	float fSigmaT;
	sm13bubble_functor(float sigmaT) : fSigmaT(sigmaT)	{}

	__host__ __device__
	float operator()(float dist) {
		return ANN::fcn_bubble_nhood(sqrt(dist), fSigmaT);
	}
};

struct sm13gaussian_functor {
	float fSigmaT;
	sm13gaussian_functor(float sigmaT) : fSigmaT(sigmaT)	{}

	__host__ __device__
	float operator()(float dist) {
		return ANN::fcn_gaussian_nhood(sqrt(dist), fSigmaT);
	}
};

struct sm13cut_gaussian_functor {
	float fSigmaT;
	sm13cut_gaussian_functor(float sigmaT) : fSigmaT(sigmaT)	{}

	__host__ __device__
	float operator()(float dist) {
		return ANN::fcn_cutgaussian_nhood(sqrt(dist), fSigmaT);
	}
};

struct sm13mexican_functor {
	float fSigmaT;
	sm13mexican_functor(float sigmaT) : fSigmaT(sigmaT)	{}

	__host__ __device__
	float operator()(float dist) {
		return ANN::fcn_mexican_nhood(sqrt(dist), fSigmaT);
	}
};

struct sm13epanechicov_functor {
	float fSigmaT;
	sm13epanechicov_functor(float sigmaT) : fSigmaT(sigmaT)	{}

	__host__ __device__
	float operator()(float dist) {
		return ANN::fcn_epanechicov_nhood(sqrt(dist), fSigmaT);
	}
};
//////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////
struct hebbian_functor {
	float fLearningRate;
	float fInput;

	hebbian_functor(float learning_rate, float input) :
		fLearningRate(learning_rate), fInput(input) {}

	__host__ __device__
	float operator()(float fWeight, float fInfluence) {
		return fWeight + (fInfluence*fLearningRate*(fInput-fWeight) );
	}
};
//////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////
#if defined(__CUDA_CAB__) && (__CUDA_CAB__ >= 20)
typedef float (*pDistanceFu) (float, float);
struct sm20distance_functor {
	float fSigmaT;
	pDistanceFu m_pfunc;
	sm20distance_functor(float sigmaT, pDistanceFu pfunc) : 
		fSigmaT(sigmaT), m_pfunc(pfunc) {}

	__host__ __device__
	float operator()(float dist) {
		return (*m_pfunc)(sqrt(dist), fSigmaT);
	}
};
#endif

#endif