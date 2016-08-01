/*
#-------------------------------------------------------------------------------
# Copyright (c) 2012 Daniel <dgrat> Frenzel.
# All rights reserved. This program and the accompanying materials
# are made available under the terms of the GNU Lesser Public License v2.1
# which accompanies this distribution, and is available at
# http://www.gnu.org/licenses/old-licenses/gpl-2.0.html
# 
# Contributors:
#     Daniel <dgrat> Frenzel - initial API and implementation
#-------------------------------------------------------------------------------
*/
#pragma once

#define STRINGIFY(s) XSTRINGIFY(s)
#define XSTRINGIFY(s) #s

#ifndef SWIG
#include <cmath>
#include <stdio.h>
#include <string.h>
#endif

#define PI 3.14159265358979323846f 

#ifdef __CUDACC__
	#define __cudacc_attribute __host__ __device__
#else
	#define __cudacc_attribute
#endif


namespace ANN {

template <class T>
struct HebbianConf {
	T learning_rate;
	T momentum_rate;
	T weight_decay;
};

//////////////////////////////////////////////////////////////////////////////////////////////
/* Some basic functions of the neuronal net
 * All of the functions could get used with CUDA
 * and there the declaration must be in the same file as the implementation
 */
//////////////////////////////////////////////////////////////////////////////////////////////
/*
 * Transfer functions for backpropagation networks
 */
//////////////////////////////////////////////////////////////////////////////////////////////
template <class T>
inline T __cudacc_attribute fcn_tanh_normal (T in, T theta) {
	return (tanh (in - theta));
}

template <class T>
inline T __cudacc_attribute fcn_tanh_derivate (T in, T theta) {
	return (1.f - pow (tanh (in - theta), 2.f));
}
//////////////////////////////////////////////////////////////////////////////////////////////
template <class T>
inline T __cudacc_attribute fcn_log_normal (T in, T theta) {
	return (1.f / (1.f + exp (theta - in)));
}

template <class T>
inline T __cudacc_attribute fcn_log_derivate (T in, T theta) {
	T e_val;
	e_val = exp (theta - in);
	return (e_val / pow (e_val + 1.f, 2.f));
}
//////////////////////////////////////////////////////////////////////////////////////////////
template <class T>
inline T __cudacc_attribute fcn_linear_normal (T in, T theta) {
	return (in - theta);
}

template <class T>
inline T __cudacc_attribute fcn_linear_derivate (T in, T theta) {
	return (1.f);
}
//////////////////////////////////////////////////////////////////////////////////////////////
template <class T>
inline T __cudacc_attribute fcn_binary_normal (T in, T theta) {
	if (in >= theta) {
		return (1.f);
	}
	return (-1.f);
}

template <class T>
inline T __cudacc_attribute fcn_binary_derivate (T in, T theta) {
	return (1.f);
}

//////////////////////////////////////////////////////////////////////////////////////////////
/*
 * Distance functions for self organizing maps
 */
//////////////////////////////////////////////////////////////////////////////////////////////
template <class T>
inline T __cudacc_attribute fcn_bubble_nhood (T dist, T sigmaT) {
	if(dist < sigmaT)
		return 1.f;
	else return 0.f;
}

//////////////////////////////////////////////////////////////////////////////////////////////
template <class T>
inline T __cudacc_attribute fcn_gaussian_nhood (T dist, T sigmaT) {
	return exp(-pow(dist, 2.f)/(2.f*pow(sigmaT, 2.f)));
}

//////////////////////////////////////////////////////////////////////////////////////////////
template <class T>
inline T __cudacc_attribute fcn_cutgaussian_nhood (T dist, T sigmaT) {
	if(dist < sigmaT)
		return exp(-pow(dist, 2.f)/(2.f*pow(sigmaT, 2.f)));
	else return 0.f;
}

//////////////////////////////////////////////////////////////////////////////////////////////
template <class T>
inline T __cudacc_attribute fcn_mexican_nhood (T dist, T sigmaT) {
	return 	2.f/(sqrt(3.f * sigmaT) * pow(PI, 0.25f) ) * 
		(1.f-pow(dist, 2.f) / pow(sigmaT, 2.f) ) * 
		fcn_gaussian_nhood(dist, sigmaT);
}

//////////////////////////////////////////////////////////////////////////////////////////////
template <class T>
inline T __cudacc_attribute fcn_epanechicov_nhood (T dist, T sigmaT) {
	T fVal = 1 - pow(dist/sigmaT, 2.f);
	if(fVal > 0)
		return fVal;
	else return 0.f;
}

//////////////////////////////////////////////////////////////////////////////////////////////
template <class T>
inline T __cudacc_attribute fcn_rad_decay (T sigma0, T t, T lambda) {
	return std::floor(sigma0*exp(-t/lambda) + 0.5f);
}

//////////////////////////////////////////////////////////////////////////////////////////////
template <class T>
inline T __cudacc_attribute fcn_lrate_decay (T sigma0, T t, T lambda) {
	return sigma0*exp(-t/lambda);
}

//////////////////////////////////////////////////////////////////////////////////////////////
template <class T>
inline T __cudacc_attribute hebbian_learn (T neuron_value, T edge_value, T edge_momentum, T targ_neuron_error_delta, const HebbianConf<T> &heb) {
	return heb.learning_rate * neuron_value * targ_neuron_error_delta
	- heb.weight_decay * edge_value
	+ heb.momentum_rate * edge_momentum;
}

template <class T>
inline T __cudacc_attribute som_learn(T fWeight, T fInfluence, T fInput) {
	return fWeight + (fInfluence*(fInput-fWeight) );
}

/** 
 * @class TransfFunction
 * @brief Represents an activation function.
 * Complete definition of the function and it's derivate.
 */
template<class T> using pLearn = T (*)(T, T, T, T, const HebbianConf<T> &);
template<class T> using p2Parm = T (*)(T, T);
template<class T> using p3Parm = T (*)(T, T, T);

template <class Type, pLearn<Type> LearnCB, p2Parm<Type> NormCB, p2Parm<Type> DerivCB>
struct TransfFunction {
	/** 
	 * @brief learning function (e.g. Hebbian). 
	 */
	static Type learn(Type a, Type b, Type c, Type d, const HebbianConf<Type> &setup) { 
		return LearnCB(a, b, c, d, setup); 
	}

	/** 
	 * @brief Transfer function for backpropagation networks.
	 * The first parameter gives the x-value,
	 * the second one is the theta value, taken from the neuron.
	 */
	static Type transfer(Type a, Type b) { 
		return NormCB(a, b); 
	}

	/** 
	 * @brief The derivative of the transfre function for backpropagation networks.
	 * Used for the backpropagation algorithm.
	 */
	static Type derivate(Type a, Type b) { 
		return DerivCB(a, b); 
	}
};

template<class T> using fcn_log = TransfFunction<T, hebbian_learn<T>, fcn_log_normal<T>, fcn_log_derivate<T> >;
template<class T> using fcn_tan = TransfFunction<T, hebbian_learn<T>, fcn_tanh_normal<T>, fcn_tanh_derivate<T> >;
template<class T> using fcn_lin = TransfFunction<T, hebbian_learn<T>, fcn_linear_normal<T>, fcn_linear_derivate<T> >;
template<class T> using fcn_bin = TransfFunction<T, hebbian_learn<T>, fcn_binary_normal<T>, fcn_binary_derivate<T> >;

/** 
 * @class DistFunction
 * @brief Represents a neighborhood and decay function.
 * Consists of a distance and a decay function. 
 * Normally just the neighborhood function is free to be changed. 
 */
template <class Type, p3Parm<Type> LearnCB, p2Parm<Type> DistCB, p3Parm<Type> RadCB, p3Parm<Type> LRateCB>
class DistFunction {
public:
	/** 
	 * @brief learning function (e.g. Hebbian). 
	 */
	static __cudacc_attribute Type learn(Type a, Type b, Type c) { 
		return LearnCB(a, b, c); 
	}
	
	/** 
	 * @brief The distance (or neighborhood) function for SOMs
	 * Used for the determination of the excitation of a neuron.
	 */
	static __cudacc_attribute Type distance(Type a, Type b) { return DistCB(a,b); };
	
	/**  
	 * @brief The decay function for SOMs
	 * Calculates the decay after each epoch.\n
	 * \f$
	 * \\ \sigma(t) = floor(\sigma_0e^{-\frac{t}{\lambda}})+0.5
	 * \\
	 * \\ \mbox{The Greek letter sigma (} \sigma_0 \mbox{) denotes the width of the lattice at time t(0) }
	 * \\ \mbox{and the Greek letter lambda (} \lambda \mbox{) denotes a time constant. }
	 * \\ \mbox{t is the current time-step (iteration of the loop). }
	 * \f$
	 */
	static __cudacc_attribute Type rad_decay(Type a, Type b, Type c) { return RadCB(a,b,c); };
	
	/**  
	 * @brief The decay function for SOMs
	 * Calculates the decay after each epoch.\n
	 * \f$
	 * \\ \sigma(t) = \sigma_0e^{-\frac{t}{\lambda}}
	 * \\
	 * \\ \mbox{The Greek letter sigma (} \sigma_0 \mbox{) denotes the width of the lattice at time t(0) }
	 * \\ \mbox{and the Greek letter lambda (} \lambda \mbox{) denotes a time constant. }
	 * \\ \mbox{t is the current time-step (iteration of the loop). }
	 * \f$
	 */
	static __cudacc_attribute Type lrate_decay(Type a, Type b, Type c) { return LRateCB(a,b,c); };
};

template<class T> using functor_gaussian = DistFunction<T, som_learn<T>, fcn_gaussian_nhood<T>, fcn_rad_decay<T>, fcn_lrate_decay<T> >;
template<class T> using functor_bubble = DistFunction<T, som_learn<T>, fcn_bubble_nhood<T>, fcn_rad_decay<T>, fcn_lrate_decay<T> >;
template<class T> using functor_cutgaussian = DistFunction<T, som_learn<T>, fcn_cutgaussian_nhood<T>, fcn_rad_decay<T>, fcn_lrate_decay<T> >;
template<class T> using functor_epanechicov = DistFunction<T, som_learn<T>, fcn_epanechicov_nhood<T>, fcn_rad_decay<T>, fcn_lrate_decay<T> >;
template<class T> using functor_mexican = DistFunction<T, som_learn<T>, fcn_mexican_nhood<T>, fcn_rad_decay<T>, fcn_lrate_decay<T> >;
};

#ifdef __Functions_ADDONS
	#include __Functions_ADDONS
#endif
