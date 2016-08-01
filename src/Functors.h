#pragma once

#ifndef SWIG
#include "math/Functions.h"
#endif


template <class Type>
struct sAXpY_functor { // Y <- A * X + Y
	Type a;
	sAXpY_functor(Type _a) : a(_a) {}

	__cudacc_attribute
	Type operator()(Type x, Type y) const {
		return a * x + y;
	}
};

template <class Type>
struct sAX_functor { // Y <- A * X
	Type a;
	sAX_functor(Type _a) : a(_a) {}
	
	__cudacc_attribute
	Type operator()(Type x) const {
		return a * x;
	}
};

template <class Type>
struct sAXmY_functor { // Y <- A * (X - Y)
	Type a;
	sAXmY_functor(Type _a) : a(_a) {}

	__cudacc_attribute
	Type operator()(Type x, Type y) const { 
		return a * (x - y);
	}
};

template <class Type>
struct sXmAmY_functor { // Y <- X - (A - Y)
	Type a;
	sXmAmY_functor(Type _a) : a(_a) {}

	__cudacc_attribute
	Type operator()(Type x, Type y) const { 
		return x - (a - y);
	}
};

template <class Type>
struct spowAmXpY_functor { // Y <- (A-X)^2 + Y
	Type a;
	spowAmXpY_functor(Type _a) : a(_a) {}

	__cudacc_attribute
	Type operator()(Type x, Type y) const { 
		return std::pow(a-x, 2) + y;
	}
};

template <class Type>
struct square_root {
	__cudacc_attribute
	Type operator()(Type x) const {
		return std::sqrt(x);
	}
};


template <class Type, class F>
struct hebbian_functor {
	Type fInput;
	hebbian_functor(Type input) : fInput(input) {}

	__cudacc_attribute
	Type operator()(Type fWeight, Type fInfluence) const {
		return F::learn(fWeight, fInfluence, fInput);
	}
};

template <class Type, class F>
struct distance_functor {
	__cudacc_attribute
	Type operator()(Type sigmaT, Type dist) const {
		return F::distance(dist, sigmaT);
	}
};

template <class Type, class F>
struct rad_decay_functor {
	Type fCycle;
	Type fCycles;
	rad_decay_functor(Type cycle, Type cycles) : fCycle(cycle), fCycles(cycles) {}

	__cudacc_attribute
	Type operator()(Type sigma0) const {
		Type fLambda = fCycles / log(sigma0);
		return F::rad_decay(sigma0, fCycle, fLambda);
	}
};

template <class Type, class F>
struct lrate_decay_functor {
	Type fCycle;
	Type fCycles;
	lrate_decay_functor(Type cycle, Type cycles) : fCycle(cycle), fCycles(cycles) {}

	__cudacc_attribute
	Type operator()(Type lrate) const {
		return F::lrate_decay(lrate, fCycle, fCycles);
	}
};

