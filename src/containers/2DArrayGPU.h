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
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/extrema.h>
#include <thrust/distance.h>

#include <cassert>
#endif


namespace ANNGPGPU {
  
template<class T> using iterator = thrust::detail::normal_iterator<thrust::device_ptr<T>>;
template<class T> using const_iterator = thrust::detail::normal_iterator<thrust::device_ptr<const T>>;

/*
 * Host classes
 */
template <class Type>
class F2DArray : public thrust::device_vector<Type> {
private:
	unsigned int iWidth;
	unsigned int iHeight;

public:
	F2DArray();
	F2DArray(const unsigned int &width, const unsigned int &height, const Type &fVal);
	F2DArray(const unsigned int &width, const unsigned int &height, thrust::host_vector<Type> vArray);

	thrust::device_vector<Type> GetSubArrayX(const unsigned int &iY) const;
	thrust::device_vector<Type> GetSubArrayY(const unsigned int &iX) const;

	unsigned int GetW() const {
		return iWidth;
	}
	unsigned int GetH() const {
		return iHeight;
	}
	
	unsigned int GetTotalSize() const {
		return this->size();
	}

	void SetValue(const unsigned int &iX, const unsigned int &iY, const Type &fVal) {
		assert(iX < iWidth);
		assert(iY < iHeight);
		
		(*this)[iY*iWidth+iX] = fVal;
	}
	Type GetValue(const unsigned int &iX, const unsigned int &iY) const {
	  	assert(iX < iWidth);
		assert(iY < iHeight);
		
		return (*this)[iY*iWidth+iX];
	}
	
	iterator<Type> GetRowBegin(const unsigned int &y) {
		assert(y < iHeight);
		return this->begin()+y*iWidth;
	}
	
	iterator<Type> GetRowEnd(const unsigned int &y) {
		assert(y < iHeight);
		return this->begin()+y*iWidth+iWidth;
	}
	
	const_iterator<Type> GetRowBegin(const unsigned int &y) const {
		assert(y < iHeight);
		return this->begin()+y*iWidth;
	}
	
	const_iterator<Type> GetRowEnd(const unsigned int &y) const {
		assert(y < iHeight);
		return this->begin()+y*iWidth+iWidth;
	}
	
#ifdef __F2DArrayGPU_ADDON
	#include __F2DArrayGPU_ADDON
#endif
};

#include "2DArrayGPU.tpp"

}

