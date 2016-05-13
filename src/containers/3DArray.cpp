/*
 * F3DArray.cpp
 *
 *  Created on: 22.01.2011
 *      Author: dgrat
 */


#include <iostream>
#include <cassert>
#include <vector>
#include <string.h>
//own classes
#include "2DArray.h"
#include "3DArray.h"

using namespace ANN;


F3DArray::F3DArray() {
	m_iX = 0;
	m_iY = 0;
	m_iZ = 0;
}

F3DArray::F3DArray(const unsigned int &iSizeX, const unsigned int &iSizeY, const unsigned int &iSizeZ, const float &fVal) {
	Alloc(iSizeX, iSizeY, iSizeZ);
	for(unsigned int i = 0; i < iSizeX*iSizeY*iSizeZ; i++) {
		m_pArray[i] = fVal;
	}
}

F3DArray::F3DArray(const unsigned int &iSizeX, const unsigned int &iSizeY, const unsigned int &iSizeZ, float *pArray) {
	assert(pArray != NULL);
	assert(iSizeX*iSizeY*iSizeZ >= 0);
  
	Alloc(iSizeX, iSizeY, iSizeZ);
	memcpy( m_pArray, pArray, m_iX*m_iY*m_iZ*sizeof(float) );
}

F3DArray::~F3DArray() {
/*
	if(m_iX*m_iY*m_iZ > 0) {
		delete [] m_pArray;
	}
*/
}

void F3DArray::Alloc(const unsigned int &iX, const unsigned int &iY, const unsigned int &iZ) {
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
	m_pArray = new float[iSize];
	memset( m_pArray, 0, iSize*sizeof(float) );
}

unsigned int F3DArray::GetW() const {
	return m_iX;
}

unsigned int F3DArray::GetH() const {
	return m_iY;
}

unsigned int F3DArray::GetD() const {
	return m_iZ;
}

unsigned int F3DArray::GetTotalSize() const {
	return (m_iX*m_iY*m_iZ);
}

void F3DArray::SetSubArrayYZ(const unsigned int &iX, const F2DArray &mYZ) {
	assert( iX < m_iX );

	for(unsigned int y = 0; y < m_iY; y++) {
		for(int z = 0; z < m_iZ; z++) {
			SetValue(iX, y, z, mYZ.GetValue(y, z) );
		}
	}
}

void F3DArray::SetSubArrayXZ(const unsigned int &iY, const F2DArray &mXZ) {
	assert( iY < m_iY );

	for(unsigned int x = 0; x < m_iX; x++) {
		for(int z = 0; z < m_iZ; z++) {
			SetValue(x, iY, z, mXZ.GetValue(x, z) );
		}
	}
}

void F3DArray::SetSubArrayXY(const unsigned int &iZ, const F2DArray &mXY) {
	assert( iZ < m_iZ );

	for(unsigned int x = 0; x < m_iX; x++) {
		for(int y = 0; y < m_iY; y++) {
			SetValue(x, y, iZ, mXY.GetValue(x, y) );
		}
	}
}

F2DArray F3DArray::GetSubArrayYZ(const unsigned int &iX) const {
	assert( iX < m_iX );

	F2DArray f2dYZ;
	f2dYZ.Alloc(m_iY, m_iZ);
	for(unsigned int y = 0; y < m_iY; y++) {
		for(int z = 0; z < m_iZ; z++) {
			f2dYZ.SetValue(y, z, GetValue(iX, y, z) );
		}
	}
	return f2dYZ;
}

F2DArray F3DArray::GetSubArrayXZ(const unsigned int &iY) const {
	assert( iY < m_iY );

	F2DArray f2dXZ;
	f2dXZ.Alloc(m_iX, m_iZ);
	for(unsigned int x = 0; x < m_iX; x++) {
		for(int z = 0; z < m_iZ; z++) {
			f2dXZ.SetValue(x, z, GetValue(x, iY, z) );
		}
	}
	return f2dXZ;
}

F2DArray F3DArray::GetSubArrayXY(const unsigned int &iZ) const {
	assert( iZ < m_iZ );

	float *pSubArray = &m_pArray[iZ*m_iX*m_iY];
	return F2DArray(m_iX, m_iY, pSubArray);
}

void F3DArray::SetValue(const int &iX, const int &iY, const int &iZ, 
			const float &fVal)
{
	assert( iY < m_iY );
	assert( iX < m_iX );
	assert( iZ < m_iZ );
	// x + y Dx + z Dx Dy
	m_pArray[iX + iY*m_iX + iZ*m_iX*m_iY] = fVal;
}

float F3DArray::GetValue(const int &iX, const int &iY, const int &iZ) const {
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

F3DArray::operator float*() {
	return m_pArray;
}

F2DArray F3DArray::operator[] (const int &iX) const {
	// z Dx Dy
	return GetSubArrayYZ(iX);
}
