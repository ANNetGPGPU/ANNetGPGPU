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

#ifndef SWIG
#include <random>
#include <ctime>
#endif
/*
#ifdef __linux__
	#include <sys/times.h>
	inline long getTickCount() {
		struct tms tm;
		return times(&tm);
	}
#endif //__linux__

#ifdef WIN32
	typedef unsigned long DWORD;
	typedef unsigned short WORD;
	typedef unsigned int UNINT32;

	#include <windows.h>
#endif //WIN32
*/

namespace ANN {

/**
 * @brief A random number generator for floats
 * @param min Minimum value of random number 
 * @param max Maximum value of random number
 * @return Returns a random min >= number <= max
 */
static std::default_random_engine __RAND_GEN(std::time(0));

template <class T>
inline T GetRandReal(T min, T max) {
	static std::uniform_real_distribution<T> __RAND_DIST(min, max);
	return __RAND_DIST(__RAND_GEN);
}

inline int GetRandInt(int min, int max) {
	auto val = GetRandReal((double)min, (double)max);
	return static_cast<int>(round(val));
}

}
