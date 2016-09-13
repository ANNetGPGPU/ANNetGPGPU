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

