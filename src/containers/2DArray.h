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
#include <cassert>
#include <vector>
#include <iostream>
#include <cstring>
#ifdef __CUDA_ARCH__ 
#include "2DArrayGPU.h"
#endif
#endif

namespace ANN {

template<class T> class F3DArray;

/**
 * \brief Pseudo 2D-array.
 * @author Daniel "dgrat" Frenzel
 */
template <class Type>
class F2DArray {
	friend class F3DArray<Type>;

private:
	unsigned int m_iX;	// nr. of neurons in layer m_iY
	unsigned int m_iY;	// nr. of layer in net
	Type *m_pArray;	// value of neuron

protected:
	void GetOutput();
  
	void SetArray(const unsigned int &iX, const unsigned int &iY, const Type &fVal);
	void SetArray(const unsigned int &iX, const unsigned int &iY, Type *pArray);
	Type *GetArray() const;

public:
	// Standard C++ "conventions"
	F2DArray();
	F2DArray(const unsigned int &iSizeX, const unsigned int &iSizeY, const Type &fVal);
	F2DArray(const unsigned int &iSizeX, const unsigned int &iSizeY, Type *pArray);
#ifdef __CUDACC__ 
	F2DArray(const ANNGPGPU::F2DArray<Type> &mat);
#endif
	virtual ~F2DArray();

	void Alloc(const unsigned int &iSize);
	void Alloc(const unsigned int &iX, const unsigned int &iY);

	unsigned int GetW() const;
	unsigned int GetH() const;
	
	unsigned int GetTotalSize() const;

	void SetSubArrayX(const unsigned int &iY, const std::vector<Type> &vRow);
	void SetSubArrayY(const unsigned int &iX, const std::vector<Type> &vCol);
	
	std::vector<Type> GetSubArrayX(const unsigned int &iY) const;
	std::vector<Type> GetSubArrayY(const unsigned int &iX) const;

	void SetValue(const unsigned int &iX, const unsigned int &iY, const Type &fVal);
	Type GetValue(const unsigned int &iX, const unsigned int &iY) const;

	/* Operators */
	operator Type*();
	operator const Type*() const;
	
	Type *operator[] (int iY);
	const Type *operator[] (int iY) const;
	
#ifdef __CUDACC__
	ANNGPGPU::F2DArray<Type> ToDevice() const;
#endif
	
#ifdef __F2DArray_ADDON
	#include __F2DArray_ADDON
#endif
};

#include "2DArray.tpp"

}

