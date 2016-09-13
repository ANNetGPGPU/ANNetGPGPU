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
F3DArray<Type>::F3DArray() {
	m_iX = 0;
	m_iY = 0;
	m_iZ = 0;
}

template <class Type>
F3DArray<Type>::F3DArray(const unsigned int &iSizeX, const unsigned int &iSizeY, const unsigned int &iSizeZ, const Type &fVal) {
	Alloc(iSizeX, iSizeY, iSizeZ);
	for(unsigned int i = 0; i < iSizeX*iSizeY*iSizeZ; i++) {
		m_pArray[i] = fVal;
	}
}

template <class Type>
F3DArray<Type>::F3DArray(const unsigned int &iSizeX, const unsigned int &iSizeY, const unsigned int &iSizeZ, Type *pArray) {
	assert(pArray != NULL);
	assert(iSizeX*iSizeY*iSizeZ >= 0);
  
	Alloc(iSizeX, iSizeY, iSizeZ);
	std::memcpy( m_pArray, pArray, m_iX*m_iY*m_iZ*sizeof(Type) );
}

template <class Type>
F3DArray<Type>::~F3DArray() {
/*
	if(m_iX*m_iY*m_iZ > 0) {
		delete [] m_pArray;
	}
*/
}

template <class Type>
void F3DArray<Type>::Alloc(const unsigned int &iX, const unsigned int &iY, const unsigned int &iZ) {
	assert( iY > 0 );
	assert( iX > 0 );
	assert( iZ > 0 );

	if(m_iX*m_iY*m_iZ > 0) {
		delete [] m_pArray;
	}

	m_iX = iX;
	m_iY = iY;
	m_iZ = iZ;
	int iSize = iX*iY*iZ;
	m_pArray = new Type[iSize];
	std::memset( m_pArray, 0, iSize*sizeof(Type) );
}

template <class Type>
unsigned int F3DArray<Type>::GetW() const {
	return m_iX;
}

template <class Type>
unsigned int F3DArray<Type>::GetH() const {
	return m_iY;
}

template <class Type>
unsigned int F3DArray<Type>::GetD() const {
	return m_iZ;
}

template <class Type>
unsigned int F3DArray<Type>::GetTotalSize() const {
	return (m_iX*m_iY*m_iZ);
}

template <class Type>
void F3DArray<Type>::SetSubArrayYZ(const unsigned int &iX, const F2DArray<Type> &mYZ) {
	assert( iX < m_iX );

	for(unsigned int y = 0; y < m_iY; y++) {
		for(int z = 0; z < m_iZ; z++) {
			SetValue(iX, y, z, mYZ.GetValue(y, z) );
		}
	}
}

template <class Type>
void F3DArray<Type>::SetSubArrayXZ(const unsigned int &iY, const F2DArray<Type> &mXZ) {
	assert( iY < m_iY );

	for(unsigned int x = 0; x < m_iX; x++) {
		for(int z = 0; z < m_iZ; z++) {
			SetValue(x, iY, z, mXZ.GetValue(x, z) );
		}
	}
}

template <class Type>
void F3DArray<Type>::SetSubArrayXY(const unsigned int &iZ, const F2DArray<Type> &mXY) {
	assert( iZ < m_iZ );

	for(unsigned int x = 0; x < m_iX; x++) {
		for(int y = 0; y < m_iY; y++) {
			SetValue(x, y, iZ, mXY.GetValue(x, y) );
		}
	}
}

template <class Type>
F2DArray<Type> F3DArray<Type>::GetSubArrayYZ(const unsigned int &iX) const {
	assert( iX < m_iX );

	F2DArray<Type> f2dYZ;
	f2dYZ.Alloc(m_iY, m_iZ);
	for(unsigned int y = 0; y < m_iY; y++) {
		for(int z = 0; z < m_iZ; z++) {
			f2dYZ.SetValue(y, z, GetValue(iX, y, z) );
		}
	}
	return f2dYZ;
}

template <class Type>
F2DArray<Type> F3DArray<Type>::GetSubArrayXZ(const unsigned int &iY) const {
	assert( iY < m_iY );

	F2DArray<Type> f2dXZ;
	f2dXZ.Alloc(m_iX, m_iZ);
	for(unsigned int x = 0; x < m_iX; x++) {
		for(int z = 0; z < m_iZ; z++) {
			f2dXZ.SetValue(x, z, GetValue(x, iY, z) );
		}
	}
	return f2dXZ;
}

template <class Type>
F2DArray<Type> F3DArray<Type>::GetSubArrayXY(const unsigned int &iZ) const {
	assert( iZ < m_iZ );

	Type *pSubArray = &m_pArray[iZ*m_iX*m_iY];
	return F2DArray<Type>(m_iX, m_iY, pSubArray);
}

template <class Type>
void F3DArray<Type>::SetValue(const int &iX, const int &iY, const int &iZ, 
			const Type &fVal)
{
	assert( iY < m_iY );
	assert( iX < m_iX );
	assert( iZ < m_iZ );
	// x + y Dx + z Dx Dy
	m_pArray[iX + iY*m_iX + iZ*m_iX*m_iY] = fVal;
}

template <class Type>
Type F3DArray<Type>::GetValue(const int &iX, const int &iY, const int &iZ) const {
	assert( iY < m_iY );
	assert( iX < m_iX );
	assert( iZ < m_iZ );
	// x + y Dx + z Dx Dy
	return m_pArray[iX + iY*m_iX + iZ*m_iX*m_iY];
}

//OPERATORS
/**
 * Returns the X*Y matrix
 * by splicing via z-axis
 */
template <class Type>
F3DArray<Type>::operator Type*() {
	return m_pArray;
}

template <class Type>
F2DArray<Type> F3DArray<Type>::operator[] (const int &iX) const {
	// z Dx Dy
	return GetSubArrayYZ(iX);
}
