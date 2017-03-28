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
	std::uniform_real_distribution<T> __RAND_DIST(min, max);
	return __RAND_DIST(__RAND_GEN);
}

inline int GetRandInt(int min, int max) {
	auto val = GetRandReal((double)min, (double)max);
	return static_cast<int>(round(val));
}

}
