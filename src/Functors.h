#ifndef ANFUNCTORS_H_
#define ANFUNCTORS_H_

#ifndef SWIG
#include "math/Functions.h"
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
	float fInput;
	hebbian_functor(float input) : fInput(input) {}

	__host__ __device__
	float operator()(float fWeight, float fInfluence) {
		return fWeight + (fInfluence*(fInput-fWeight) );
	}
};
//////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////
template <class F>
struct sm20distance_functor {
	__host__ __device__
	float operator()(float sigmaT, float dist) {
		return F::distance(sqrt(dist), sigmaT);
	}
};
//////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////
template <class F>
struct sm20rad_decay_functor {
	float fCycle;
	float fCycles;
	sm20rad_decay_functor(float cycle, float cycles) : fCycle(cycle), fCycles(cycles) {}

	__host__ __device__
	float operator()(float sigma0) {
		float fLambda = fCycles / log(sigma0);
		return pow(F::rad_decay(sigma0, fCycle, fLambda), 2);
	}
};
//////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////
template <class F>
struct sm20lrate_decay_functor {
	float fCycle;
	float fCycles;
	sm20lrate_decay_functor(float cycle, float cycles) : fCycle(cycle), fCycles(cycles) {}

	__host__ __device__
	float operator()(float lrate) {
		return F::lrate_decay(lrate, fCycle, fCycles);
	}
};

#endif
