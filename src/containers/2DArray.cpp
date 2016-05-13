/*
 * F2DArray.cpp
 *
 *  Created on: 28.01.2011
 *      Author: dgrat
 */

#include <cassert>
#include <string.h>
// own classes
#include "2DArray.h"


using namespace ANN;


F2DArray::F2DArray() {
	m_iX 	= 0;
	m_iY 	= 0;
	m_pArray 	= NULL;
//	m_pSubArray = NULL;

	m_bAllocated = false;
}

#ifdef __CUDACC__
/**
  * CUDA THRUST compatibility
  * host_vector<float>: Contains one row of the matrix
  * host_vector< host_vector<float> >: Contains all rows  of the matrix
  */
F2DArray::F2DArray(const ANNGPGPU::F2DArray &mat) {
	unsigned int iHeight 	= mat.GetH();
	unsigned int iWidth 	= mat.GetW();

	Alloc(iWidth, iHeight);

	for(unsigned int y = 0; y < iHeight; y++) {
		for(unsigned int x = 0; x < iWidth; x++) {
			m_pArray[y*iWidth+x] = mat[y*iWidth+x];
		}
	}
}

F2DArray::operator ANNGPGPU::F2DArray () {
	ANNGPGPU::F2DArray dmRes(GetW(), GetH(), 0.f);

	for(unsigned int y = 0; y < GetH(); y++) {
		for(unsigned int x = 0; x < GetW(); x++) {
			dmRes[y*GetW()+x] = m_pArray[y*GetW()+x];
		}
	}

	return dmRes;
}
#endif

F2DArray::F2DArray(const unsigned int &iSizeX, const unsigned int &iSizeY, const float &fVal) {
	SetArray(iSizeX, iSizeY, fVal);
}

F2DArray::F2DArray(const unsigned int &iSizeX, const unsigned int &iSizeY, float *pArray) {
	SetArray(iSizeX, iSizeY, pArray);
}

F2DArray::~F2DArray() {
/*
	if(m_bAllocated) {
		if( m_pArray != NULL )
			delete [] m_pArray;
//		if(m_pSubArray != NULL)
//			delete [] m_pSubArray;
	}
*/
}

void F2DArray::Alloc(const unsigned int &iSize) {
	assert( iSize > 0 );
/*
	if(m_bAllocated) {
		if( m_pArray != NULL )
			delete [] m_pArray;
		if( m_pSubArray != NULL )
			delete [] m_pSubArray;
	}
*/
	m_iX 	= 0;
	m_iY 	= 0;
	m_pArray 	= new float[iSize];
//	m_pSubArray = NULL;
	memset( m_pArray, 0, iSize*sizeof(float) );
	m_bAllocated = true;
}

void F2DArray::Alloc(const unsigned int &iX, const unsigned int &iY) {
	assert( iY > 0 );
	assert( iX > 0 );
/*
	if( m_bAllocated ) {
		if( m_pArray != NULL )
			delete [] m_pArray;
		if( m_pSubArray != NULL )
			delete [] m_pSubArray;
	}
*/
	m_iX 	= iX;
	m_iY 	= iY;
	m_pArray 	= new float[iX*iY];
//	m_pSubArray = new float[iY];
	memset( m_pArray, 0, iX*iY*sizeof(float) );
	m_bAllocated = true;
}

unsigned int F2DArray::GetW() const {
	return m_iX;
}

unsigned int F2DArray::GetH() const {
	return m_iY;
}

unsigned int F2DArray::GetTotalSize() const {
	return m_iY * m_iX;
}

std::vector<float> F2DArray::GetSubArrayX(const unsigned int &iY) const {
	assert(iY < m_iY);

	std::vector<float> vSubArray(GetW() );
	std::copy(&m_pArray[iY*m_iX], &m_pArray[iY*m_iX]+GetW(), vSubArray.begin() );
	return vSubArray; //return &m_pArray[iY*m_iX];
}

std::vector<float> F2DArray::GetSubArrayY(const unsigned int &iX) const {
	assert(iX < m_iX);

	std::vector<float> vSubArray(GetH() );
	for(unsigned int y = 0; y < m_iY; y++) {
		vSubArray[y] = GetValue(iX, y);
	}
	return vSubArray;
}

void F2DArray::SetSubArrayX(const unsigned int &iY, const std::vector<float> &vRow) {
	assert(iY < m_iY);
	assert(vRow.size() == m_iX);

	memcpy(&m_pArray[iY*m_iX], &vRow.data()[0], m_iX*sizeof(float) );
}

void F2DArray::SetSubArrayY(const unsigned int &iX, const std::vector<float> &vCol) {
	assert(iX < m_iX);
	assert(vCol.size() == m_iY);

	for(unsigned int y = 0; y < m_iY; y++) {
		SetValue(iX, y, vCol.at(y) );
	}
}

void F2DArray::SetValue(const unsigned int &iX, const unsigned int &iY, const float &fVal) {
	assert(iY < m_iY);
	assert(iX < m_iX);

	m_pArray[iX + iY*m_iX] = fVal;
}

float F2DArray::GetValue(const unsigned int &iX, const unsigned int &iY) const {
	assert(iY < m_iY);
	assert(iX < m_iX);

	return m_pArray[iX + iY*m_iX];
}

void F2DArray::SetArray(const unsigned int &iSizeX, const unsigned int &iSizeY, const float &fVal) {
	Alloc(iSizeX, iSizeY);
	for(unsigned int y = 0; y < m_iY; y++) {
		for(unsigned int x = 0; x < m_iX; x++) {
			SetValue(x, y, fVal);
		}
	}
}

void F2DArray::SetArray(const unsigned int &iSizeX, const unsigned int &iSizeY, float *pArray) {
	assert( pArray != NULL );

	m_pArray = pArray;
	m_iX = iSizeX;
	m_iY = iSizeY;
}

float *F2DArray::GetArray() const {
	return m_pArray;
}


F2DArray::operator float*() {
	return m_pArray;
}

F2DArray::operator const float*() const {
	return m_pArray;
}

float *F2DArray::operator[] (int iY) {
	return &m_pArray[iY*m_iX];
}

const float *F2DArray::operator[] (int iY) const {
	return &m_pArray[iY*m_iX];
}

void F2DArray::GetOutput() {
	for(unsigned int y = 0; y < GetH(); y++) {
		for(unsigned int x = 0; x < GetW(); x++) {
			std::cout << "Array["<<x<<"]["<<y<<"]=" << GetValue(x, y) << std::endl;
		}
	}
}
