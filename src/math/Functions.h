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

#ifndef TRANSFERFUNCTIONS_H_
#define TRANSFERFUNCTIONS_H_

#ifndef SWIG
#include <cmath>
#include <stdio.h>
#include <string.h>
#endif

#define PI    3.14159265358979323846f 


namespace ANN {

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
#ifdef __CUDACC__
	__host__ __device__
#endif
inline static float
fcn_tanh_normal (float in, float theta) {
	return (tanh (in - theta));
}

#ifdef __CUDACC__
	__host__ __device__
#endif
inline static float
fcn_tanh_derivate (float in, float theta) {
	return (1.f - pow (tanh (in - theta), 2.f));
}
//////////////////////////////////////////////////////////////////////////////////////////////
#ifdef __CUDACC__
	__host__ __device__
#endif
inline static float
fcn_log_normal (float in, float theta) {
	return (1.f / (1.f + exp (theta - in)));
}

#ifdef __CUDACC__
	__host__ __device__
#endif
inline static float
fcn_log_derivate (float in, float theta) {
	float e_val;
	e_val = exp (theta - in);
	return (e_val / pow (e_val + 1.f, 2.f));
}
//////////////////////////////////////////////////////////////////////////////////////////////
#ifdef __CUDACC__
	__host__ __device__
#endif
inline static float
fcn_linear_normal (float in, float theta) {
	return (in - theta);
}

#ifdef __CUDACC__
	__host__ __device__
#endif
inline static float
fcn_linear_derivate (float in, float theta) {
	return (1.f);
}
//////////////////////////////////////////////////////////////////////////////////////////////
#ifdef __CUDACC__
	__host__ __device__
#endif
inline static float
fcn_binary_normal (float in, float theta) {
	if (in >= theta) {
		return (1.f);
	}
	return (-1.f);
}

#ifdef __CUDACC__
	__host__ __device__
#endif
inline static float
fcn_binary_derivate (float in, float theta) {
	return (1.f);
}

//////////////////////////////////////////////////////////////////////////////////////////////
#ifdef __CUDACC__
struct tanTransferFcn {
	__host__ __device__
	float operator()(float fVal, float fBias) {
		return ANN::fcn_tanh_normal(fVal, fBias);
	}
};

struct devTanTransferFcn {
	__host__ __device__
	float operator()(float fVal) {
		return ANN::fcn_tanh_derivate(fVal, 0);
	}
};

//////////////////////////////////////////////////////////////////////////////////////////////
struct binTransferFcn {
	__host__ __device__
	float operator()(float fVal, float fBias) {
		return ANN::fcn_binary_normal(fVal, fBias);
	}
};

struct devBinTransferFcn {
	__host__ __device__
	float operator()(float fVal) {
		return ANN::fcn_binary_derivate(fVal, 0);
	}
};

//////////////////////////////////////////////////////////////////////////////////////////////
struct linTransferFcn {
	__host__ __device__
	float operator()(float fVal, float fBias) {
		return ANN::fcn_linear_normal(fVal, fBias);
	}
};

struct devLinTransferFcn {
	__host__ __device__
	float operator()(float fVal) {
		return ANN::fcn_linear_derivate(fVal, 0);
	}
};

//////////////////////////////////////////////////////////////////////////////////////////////
struct logTransferFcn {
	__host__ __device__
	float operator()(float fVal, float fBias) {
		return ANN::fcn_log_normal(fVal, fBias);
	}
};

struct devLogTransferFcn {
	__host__ __device__
	float operator()(float fVal) {
		return ANN::fcn_log_derivate(fVal, 0);
	}
};
#endif

//////////////////////////////////////////////////////////////////////////////////////////////
/*
 * Distance functions for self organizing maps
 */
//////////////////////////////////////////////////////////////////////////////////////////////
#ifdef __CUDACC__
	__host__ __device__
#endif
inline static float
fcn_bubble_nhood (float dist, float sigmaT) {
	if(dist < sigmaT)
		return 1.f;
	else return 0.f;
}

//////////////////////////////////////////////////////////////////////////////////////////////
#ifdef __CUDACC__
	__host__ __device__
#endif
inline static float
fcn_gaussian_nhood (float dist, float sigmaT) {
	return exp(-pow(dist, 2.f)/(2.f*pow(sigmaT, 2.f)));
}

//////////////////////////////////////////////////////////////////////////////////////////////
#ifdef __CUDACC__
	__host__ __device__
#endif
inline static float
fcn_cutgaussian_nhood (float dist, float sigmaT) {
	if(dist < sigmaT)
		return exp(-pow(dist, 2.f)/(2.f*pow(sigmaT, 2.f)));
	else return 0.f;
}

//////////////////////////////////////////////////////////////////////////////////////////////
#ifdef __CUDACC__
	__host__ __device__
#endif
inline static float
fcn_mexican_nhood (float dist, float sigmaT) {
	return 	2.f/(sqrt(3.f * sigmaT) * pow(PI, 0.25f) ) * 
		(1.f-pow(dist, 2.f) / pow(sigmaT, 2.f) ) * 
		fcn_gaussian_nhood(dist, sigmaT);
}

//////////////////////////////////////////////////////////////////////////////////////////////
#ifdef __CUDACC__
	__host__ __device__
#endif
inline static float
fcn_epanechicov_nhood (float dist, float sigmaT) {
	float fVal = 1 - pow(dist/sigmaT, 2.f);
	if(fVal > 0)
		return fVal;
	else return 0.f;
}

//////////////////////////////////////////////////////////////////////////////////////////////
#ifdef __CUDACC__
	__host__ __device__
#endif
inline static float
fcn_rad_decay (float sigma0, float T, float lambda) {
	return std::floor(sigma0*exp(-T/lambda) + 0.5f);
}

//////////////////////////////////////////////////////////////////////////////////////////////
#ifdef __CUDACC__
	__host__ __device__
#endif
inline static float
fcn_lrate_decay (float sigma0, float T, float lambda) {
	return sigma0*exp(-T/lambda);
}

/** 
 * @class TransfFunction
 * @brief Represents an activation function.
 * Complete definition of the function and it's derivate.
 */
class TransfFunction {
public:
	/** 
	 * @brief The symbolic name of the function. 
	 */
	char *name;

	/** 
	 * @brief Transfer function for backpropagation networks.
	 * The first parameter gives the x-value,
	 * the second one is the theta value, taken from the neuron.
	 */
	float (* normal)(float, float);

	/** 
	 * @brief The derivative of the transfre function for backpropagation networks.
	 * Used for the backpropagation algorithm.
	 */
	float (* derivate)(float, float);
};

/** 
 * @class DistFunction
 * @brief Represents a neighborhood and decay function.
 * Consists of a distance and a decay function. 
 * Normally just the neighborhood function is free to be changed. 
 */
class DistFunction {
public:
	/** 
	 * @brief The symbolic name of the function.
	 * Determined normally by the name of the used neighborhood function.
	 */
	char *name;

	/** 
	 * @brief The distance (or neighborhood) function for SOMs
	 * Used for the determination of the excitation of a neuron.
	 */
	float (* distance)(float, float);
	
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
	float (* rad_decay)(float sigma, float t, float lambda);
	
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
	float (* lrate_decay)(float sigma, float t, float lambda);
};

/** 
 * @class Functions
 * @brief List of activation functions that are available to the Network.
 */
class Functions {
public:
	/** 
	 * @brief Resolve activation function by symbolic name.
	 * @param  name The function name, as given in the function structure.
	 * @return NULL on failure, pointer to structure on success.
	 */
	static TransfFunction* ResolveTransfFByName (const char *name);
	
	/** 
	 * @brief Resolve neighborhood function by symbolic name.
	 * @param  name The function name, as given in the function structure.
	 * @return NULL on failure, pointer to structure on success.
	 */
	static DistFunction* ResolveDistFByName (const char *name);

	 /**
	  * @brief The sigmoid tanh function.
	  * \f$f_{act} (x, \Theta) = tanh (x - \Theta)\f$
	  */
	static TransfFunction fcn_tanh;

	 /**
	  * @brief The sigmoid log function.
	  * \f$f_{act} (x, \Theta) = \frac{1}{1 + e^{-(x - \Theta)}}\f$
	  */
	static TransfFunction fcn_log;
	
	 /**
	  * @brief A linear activation function.
	  * \f$f_{act} (x, \Theta) = x - \Theta\f$
	  */
	static TransfFunction fcn_linear;

	 /**
	  * @brief A binary activation function.
	  * \f$f_{act} (x, \Theta) = \left\{\begin{array}{cl}1.0  x \geq
	  * \Theta\\-1.0  x < \Theta\end{array}\right.\f$
	  */
	static TransfFunction fcn_binary;

	/**
	 * @brief A binary neighborhood function.
	 * \f$
	 * \\ f_{act} (dist, \sigma(t)) = \left\{\begin{array}{cl}1.0  dist < \sigma(t)
	 * \\0  dist \geq \sigma(t)\end{array}\right.\f$
	 */
	static DistFunction fcn_bubble;
	
	/**
	 * @brief The gaussian neighborhood function.
	 * \f$
	 *  \\ \sigma(t) = \sigma_0e^{-\frac{t}{\lambda}}
	 *  \\ h(t) = {e^{-\frac{dist^2}{2\sigma(t)^2}}}
	 *  \\
	 *  \\ \mbox{The Greek letter sigma (} \sigma_0 \mbox{) denotes the width of the lattice at time t(0)}
	 *  \\ \mbox{and the Greek letter lambda (} \lambda \mbox{) denotes a time constant.}
	 *  \\ \mbox{t is the current time-step (iteration of the loop).}
	 * \f$
	 */
	static DistFunction fcn_gaussian;
	
	/**
	 * @brief The cut gaussian neighborhood function.
	 * \f$
	 *  \\ {act} (dist, \sigma(t)) = \left\{\begin{array}{cl} {e^{-\frac{dist^2}{2\sigma(t)^2}}}  dist < \sigma(t)
	 *  \\ 0  dist \geq \sigma(t)\end{array}\right.
	 *  \\ \sigma(t) = \sigma_0e^{-\frac{t}{\lambda}}
	 *  \\
	 *  \\ \mbox{The Greek letter sigma (} \sigma_0 \mbox{) denotes the width of the lattice at time t(0)}
	 *  \\ \mbox{and the Greek letter lambda (} \lambda \mbox{) denotes a time constant.}
	 *  \\ \mbox{t is the current time-step (iteration of the loop).}
	 * \f$
	 */
	static DistFunction fcn_cutgaussian;
	
	/**
	 * @brief Epanechicov neighborhood function.
	 * \f$
	 *  \\ {act} (dist, \sigma(t)) = \left\{\begin{array}{cl} {1-{\frac{dist}{\sigma(t)}^2}}  {act} > 0
	 *  \\ 0  {act} \leq 0\end{array}\right.
	 *  \\ \sigma(t) = \sigma_0e^{-\frac{t}{\lambda}}
	 *  \\
	 *  \\ \mbox{The Greek letter sigma (} \sigma_0 \mbox{) denotes the width of the lattice at time t(0)}
	 *  \\ \mbox{and the Greek letter lambda (} \lambda \mbox{) denotes a time constant.}
	 *  \\ \mbox{t is the current time-step (iteration of the loop).}
	 * \f$
	 */
	static DistFunction fcn_epanechicov;
	
	/**
	 * @brief A wavelet gaussian neighborhood function (shape of mexican hat).
	 * \f$
	 * \\ \sigma(t) = \sigma_0e^{-\frac{t}{\lambda}}
	 * \\ h(t) = \frac{2}{\sqrt{3*sigma(t)}\pi^{-\frac{1}{4}}}(1-\frac{dist^2}{sigma(t)^2})({e^{-\frac{dist^2}{2\sigma(t)^2}}})
	 * \\
	 * \\ \mbox{The Greek letter sigma (} \sigma_0 \mbox{) denotes the width of the lattice at time t(0)}
	 * \\ \mbox{and the Greek letter lambda (} \lambda \mbox{) denotes a time constant.}
	 * \\ \mbox{t is the current time-step (iteration of the loop).}
	 * \\ \mbox{dist is the distance of a node to the BMU}
	 * \\ \mbox{and } \sigma \mbox{, is the width of the neighborhood function.}
	 * \f$
	 */
	static DistFunction fcn_mexican;
};

};

#endif /* TRANSFERFUNCTIONS_H_ */
