/// -*- tab-width: 8; Mode: C++; c-basic-offset: 8; indent-tabs-mode: t -*-
/*
   This program is free software: you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.
   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.
   You should have received a copy of the GNU General Public License
   along with this program.  If not, see <http://www.gnu.org/licenses/>.
   
   Author: Daniel Frenzel (dgdanielf@gmail.com)
*/

#pragma once

#ifndef SWIG
#include "math/Functions.h"
#endif


/*
 * General
 */
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

/*
 * For SOMs
 */
template <class Type, class F>
struct som_hebbian_functor {
	Type fInput;
	som_hebbian_functor(Type input) : fInput(input) {}

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
	Type _fCycle;
	Type _fCycles;
	rad_decay_functor(Type cycle, Type cycles) : _fCycle(cycle), _fCycles(cycles) {}

	__cudacc_attribute
	Type operator()(Type sigma0) const {
		Type fLambda = _fCycles / log(sigma0);
		return F::rad_decay(sigma0, _fCycle, fLambda);
	}
};

template <class Type, class F>
struct lrate_decay_functor {
	Type _fCycle;
	Type _fCycles;
	lrate_decay_functor(Type cycle, Type cycles) : _fCycle(cycle), _fCycles(cycles) {}

	__cudacc_attribute
	Type operator()(Type lrate) const {
		return F::lrate_decay(lrate, _fCycle, _fCycles);
	}
};

/*
 * For Back Propagation Networks
 */
template <class Type, class F>
struct bp_transfer_functor {
	Type _theta;
	bp_transfer_functor(Type theta) : _theta(theta) {}
	
	__cudacc_attribute
	Type operator()(Type in) const {
		return F::transfer(in, _theta);
	}
};

template <class Type, class F>
struct bp_derived_transfer_functor {
	Type _theta;
	bp_derived_transfer_functor(Type theta) : _theta(theta) {}
	
	__cudacc_attribute
	Type operator()(Type in) const {
		return F::derivate(in, _theta);
	}
};

template <class Type, class F>
struct bp_hebbian_functor {
	const ANN::HebbianConf<Type> _setup;
	bp_hebbian_functor(const ANN::HebbianConf<Type> &setup) : _setup(setup) {}
	
	__cudacc_attribute
	Type operator()(Type a, Type b, Type c, Type d) const {
		return F::learn(a, b, c, d, _setup);
	}
};
