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

#ifndef NEURONARRAY_H_
#define NEURONARRAY_H_

#ifndef SWIG
#include <iostream>
#include <vector>
#ifdef __CUDACC__
#include "2DArrayGPU.h"
#endif
#endif

namespace ANN {

class F3DArray;

/**
 * \brief Pseudo 2D-array.
 * @author Daniel "dgrat" Frenzel
 */
class F2DArray {
	friend class F3DArray;

private:
	bool m_bAllocated;
	unsigned int m_iX;	// nr. of neurons in layer m_iY
	unsigned int m_iY;	// nr. of layer in net
	float *m_pArray;	// value of neuron

protected:
	void GetOutput();
  
	void SetArray(const unsigned int &iX, const unsigned int &iY, const float &fVal);
	void SetArray(const unsigned int &iX, const unsigned int &iY, float *pArray);
	float *GetArray() const;

public:
	// Standard C++ "conventions"
	F2DArray();
	F2DArray(const unsigned int &iSizeX, const unsigned int &iSizeY, const float &fVal);
	F2DArray(const unsigned int &iSizeX, const unsigned int &iSizeY, float *pArray);
	virtual ~F2DArray();

	void Alloc(const unsigned int &iSize);
	void Alloc(const unsigned int &iX, const unsigned int &iY);

	unsigned int GetW() const;
	unsigned int GetH() const;
	
	unsigned int GetTotalSize() const;

	void SetSubArrayX(const unsigned int &iY, const std::vector<float> &vRow);
	void SetSubArrayY(const unsigned int &iX, const std::vector<float> &vCol);
	
	std::vector<float> GetSubArrayX(const unsigned int &iY) const;
	std::vector<float> GetSubArrayY(const unsigned int &iX) const;

	void SetValue(const unsigned int &iX, const unsigned int &iY, const float &fVal);
	float GetValue(const unsigned int &iX, const unsigned int &iY) const;

	/* Operators */
	operator float*();
	operator const float*() const;
	
	float *operator[] (int iY);
	const float *operator[] (int iY) const;
	
	#ifdef __CUDACC__
	F2DArray(const ANNGPGPU::F2DArray &);
	operator ANNGPGPU::F2DArray ();
	#endif
};

}

#endif /* NEURONARRAY_H_ */
