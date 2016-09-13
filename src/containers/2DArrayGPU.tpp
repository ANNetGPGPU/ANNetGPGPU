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
