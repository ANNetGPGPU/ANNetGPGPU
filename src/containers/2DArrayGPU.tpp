/*
 * F2DArray.cu
 *
 *  Created on: 05.04.2012
 *      Author: dgrat
 */

template <class Type>
F2DArray<Type>::F2DArray() : thrust::device_vector<Type>(), iWidth(0), iHeight(0) {

}

template <class Type>
F2DArray<Type>::F2DArray(const unsigned int &width, const unsigned int &height, const Type &fVal) : thrust::device_vector<Type>(width*height, fVal), iWidth(width), iHeight(height) {

}

template <class Type>
F2DArray<Type>::F2DArray(const unsigned int &width, const unsigned int &height, thrust::host_vector<Type> vec) : thrust::device_vector<Type>(vec), iWidth(width), iHeight(height) {

}

template <class Type>
thrust::device_vector<Type> F2DArray<Type>::GetSubArrayY(const unsigned int &iX) const {
	assert(iX < iWidth);
	thrust::device_vector<Type> dvTmp(iHeight);
	for(unsigned int y = 0; y < iHeight; y++) {
		Type fVal = (*this)[y*iWidth+iX];
		dvTmp[y] = fVal;
	}
	return dvTmp;
}

template <class Type>
thrust::device_vector<Type> F2DArray<Type>::GetSubArrayX(const unsigned int &iY) const {
	assert(iY < iHeight);
	thrust::device_vector<Type> dvTmp(iHeight);
	for(unsigned int x = 0; x < iWidth; x++) {
		Type fVal = (*this)[iY*iWidth+x];
		dvTmp[x] = fVal;
	}
	return dvTmp;
}
