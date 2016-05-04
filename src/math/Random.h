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

#ifndef RANDOMIZER_H_
#define RANDOMIZER_H_

#ifndef SWIG
#include <cstdlib>
#include <cmath>
#include <ctime>
#endif

#ifdef __linux__
	#include <sys/times.h>
	/*
	 * not defined in unix os but windows
	 */
	inline long getTickCount() {
		struct tms tm;
		return times(&tm);
	}
#endif /*__linux__*/

namespace ANN {

/*
 * predeclaration of some functions
 */
inline float RandFloat(float fMin, float fMax);
inline int RandInt(int iMin, int iMax);
inline void InitTime();

#define INIT_TIME InitTime();

#ifdef WIN32
	/*
	 * for getTickCount()
	 */
	typedef unsigned long 	DWORD;
	typedef unsigned short 	WORD;
	typedef unsigned int 		UNINT32;

	#include <windows.h>
#endif /*WIN32*/

/**
 * @brief Initialises the system clock
 */
void InitTime() {
	time_t t;
	time(&t);
	srand((unsigned int)t);
}

/**
 * @brief A random number generator for floats
 * @param fMin Minimum value of random number 
 * @param fMax Maximum value of random number
 * @return Returns a random fMin >= number <= fMax
 */
float RandFloat(float fMin, float fMax) {
	float temp;
	/* swap low & high around if the user makes no sense */
	if (fMin > fMax) {
		temp = fMin;
		fMin = fMax;
		fMax = temp;
	}
	/* calculate the random number & return it */
	return rand() / (RAND_MAX + 1.f) * (fMax - fMin) + fMin;
}

/**
 * @brief A random number generator for integers
 * @param fMin Minimum value of random number 
 * @param fMax Maximum value of random number
 * @return Returns a random fMin >= number <= fMax
 */
int RandInt(int iMin, int iMax) {
	int temp;
	/* swap low & high around if the user makes no sense */
	if (iMin > iMax) {
		temp = iMin;
		iMin = iMax;
		iMax = temp;
	}
	return rand()%(iMax-iMin+1)+iMin;
}

}

#endif /* RANDOMIZER_H_ */
