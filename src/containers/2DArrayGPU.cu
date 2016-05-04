/*
 * Matrix.cpp
 *
 *  Created on: 05.04.2012
 *      Author: dgrat
 */

#include "2DArrayGPU.h"


namespace ANNGPGPU {

/**
 * Matrix class implementation needs to be done by NVCC!
 */
F2DArray::F2DArray() : thrust::device_vector<float>(), iWidth(0), iHeight(0) {

}

F2DArray::F2DArray(const unsigned int &width, const unsigned int &height, const float &fVal) : thrust::device_vector<float>(width*height, fVal), iWidth(width), iHeight(height) {

}

F2DArray::F2DArray(const unsigned int &width, const unsigned int &height, thrust::host_vector<float> vec) : thrust::device_vector<float>(vec), iWidth(width), iHeight(height) {

}

thrust::device_vector<float> F2DArray::GetSubArrayY(const unsigned int &iX) const {
	assert(iX < iWidth);
	thrust::device_vector<float> dvTmp(iHeight);
	for(unsigned int y = 0; y < iHeight; y++) {
		float fVal = (*this)[y*iWidth+iX];
		dvTmp[y] = fVal;
	}
	return dvTmp;
}

thrust::device_vector<float> F2DArray::GetSubArrayX(const unsigned int &iY) const {
	assert(iY < iHeight);
	thrust::device_vector<float> dvTmp(iHeight);
	for(unsigned int x = 0; x < iWidth; x++) {
		float fVal = (*this)[iY*iWidth+x];
		dvTmp[x] = fVal;
	}
	return dvTmp;
}
/*
F2DArray::F2DArray(const ANN::F2DArray &mat) {
	unsigned int iHeight 	= mat.GetH();
	unsigned int iWidth 	= mat.GetW();

	this->clear();

	for(unsigned int y = 0; y < iHeight; y++) {
		for(unsigned int x = 0; x < iWidth; x++) {
			this->push_back(mat->GetValue(x, y) );
		}
	}
}
F2DArray::operator ANN::F2DArray () {
	ANN::F2DArray dmRes(GetW(), GetH(), 0.f);

	for(unsigned int y = 0; y < GetH(); y++) {
		for(unsigned int x = 0; x < GetW(); x++) {
			dmRes.SetValue(x, y, GetValue(x, y) );
		}
	}

	return dmRes;
}
*/
}
