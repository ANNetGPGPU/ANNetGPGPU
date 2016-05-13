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

#ifndef MATRIX_H_
#define MATRIX_H_

#ifndef SWIG
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/extrema.h>
#include <thrust/distance.h>

#include <cassert>
#endif


namespace ANNGPGPU {
  
/*
 * Host classes
 */
class F2DArray : public thrust::device_vector<float> {
private:
	unsigned int iWidth;
	unsigned int iHeight;

public:
	F2DArray();
	F2DArray(const unsigned int &width, const unsigned int &height, const float &fVal);
	F2DArray(const unsigned int &width, const unsigned int &height, thrust::host_vector<float> vArray);

	thrust::device_vector<float> GetSubArrayX(const unsigned int &iY) const;
	thrust::device_vector<float> GetSubArrayY(const unsigned int &iX) const;

	unsigned int GetW() const {
		return iWidth;
	}
	unsigned int GetH() const {
		return iHeight;
	}
	
	unsigned int GetTotalSize() const {
		return this->size();
	}

	void SetValue(const unsigned int &iX, const unsigned int &iY, const float &fVal) {
		assert(iX < iWidth);
		assert(iY < iHeight);
		
		(*this)[iY*iWidth+iX] = fVal;
	}
	float GetValue(const unsigned int &iX, const unsigned int &iY) const {
	  	assert(iX < iWidth);
		assert(iY < iHeight);
		
		return (*this)[iY*iWidth+iX];
	}

	iterator GetRowBegin(const unsigned int &y) {
		assert(y < iHeight);
		return begin()+y*iWidth;
	}
	iterator GetRowEnd(const unsigned int &y) {
		assert(y < iHeight);
		return begin()+y*iWidth+iWidth;
	}

	const_iterator GetRowBegin(const unsigned int &y) const {
		assert(y < iHeight);
		return begin()+y*iWidth;
	}
	const_iterator GetRowEnd(const unsigned int &y) const {
		assert(y < iHeight);
		return begin()+y*iWidth+iWidth;
	}
/*
	F2DArray Rotate90() {
		F2DArray mat(iHeight, iWidth, 0);
		for(unsigned int x = 0; x < iWidth; x++) {
			mat.GetRowBegin(x);
			thrust::device_vector<float> col = GetSubArrayY(x);
			thrust::copy(col.begin(), col.end(), mat.GetRowBegin(x) );
		}
		return mat;
	}
*/
};

}

#endif /* MATRIX_H_ */
