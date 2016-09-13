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
#include <cassert>
#include <cstring>
#include <iostream>
#include <vector>
#endif 

namespace ANN {

template<class T> class F2DArray;

/**
 * \brief Pseudo 3D-array.
 * @author Daniel "dgrat" Frenzel
 */
template <class Type>
class F3DArray {
	friend class F2DArray<Type>;

private:
	unsigned int m_iX;	// nr. of neuron in layer m_iY
	unsigned int m_iY;	// nr. of layer in net
	unsigned int m_iZ;	// nr. of axon/weight of neuron m:iX in layer m_iY
	Type *m_pArray;

public:
	// Standard C++ "conventions"
	F3DArray();
	F3DArray(const unsigned int &iSizeX, const unsigned int &iSizeY, const unsigned int &iSizeZ, const Type &fVal);
	F3DArray(const unsigned int &iSizeX, const unsigned int &iSizeY, const unsigned int &iSizeZ, Type *pArray);
	virtual ~F3DArray();

	void Alloc(const unsigned int &iX, const unsigned int &iY, const unsigned int &iZ);

	unsigned int GetW() const;	// X
	unsigned int GetH() const;	// Y
	unsigned int GetD() const;	// Z

	unsigned int GetTotalSize() const; 	// X*Y*Z

	void SetSubArrayYZ(const unsigned int &iX, const F2DArray<Type> &mYZ);
	void SetSubArrayXZ(const unsigned int &iY, const F2DArray<Type> &mXZ);
	void SetSubArrayXY(const unsigned int &iZ, const F2DArray<Type> &mXY);
	
	/* return a pointer to the subarray at: Y,X */
	F2DArray<Type> GetSubArrayYZ(const unsigned int &iX) const;
	F2DArray<Type> GetSubArrayXZ(const unsigned int &iY) const;
	F2DArray<Type> GetSubArrayXY(const unsigned int &iZ) const;

	void SetValue(const int &iX, const int &iY, const int &iZ, const Type &fVal);
	Type GetValue(const int &iX, const int &iY, const int &iZ) const;

//OPERATORS
	operator Type*();
	F2DArray<Type> operator[] (const int &iX) const;
	
#ifdef __F3DArray_ADDON
	#include __F3DArray_ADDON
#endif
};

#include "3DArray.tpp"

}

