/*
 * F2DArray.cpp
 *
 *  Created on: 28.01.2011
 *      Author: dgrat
 */
template <class Type>
F2DArray<Type>::F2DArray() {
	m_iX 	= 0;
	m_iY 	= 0;
	m_pArray 	= NULL;
//	m_pSubArray = NULL;

	m_bAllocated = false;
}

template <class Type>
F2DArray<Type>::F2DArray(const unsigned int &iSizeX, const unsigned int &iSizeY, const Type &fVal) {
	SetArray(iSizeX, iSizeY, fVal);
}

template <class Type>
F2DArray<Type>::F2DArray(const unsigned int &iSizeX, const unsigned int &iSizeY, Type *pArray) {
	SetArray(iSizeX, iSizeY, pArray);
}

template <class Type>
F2DArray<Type>::~F2DArray() {
/*
	if(m_bAllocated) {
		if( m_pArray != NULL )
			delete [] m_pArray;
//		if(m_pSubArray != NULL)
//			delete [] m_pSubArray;
	}
*/
}

template <class Type>
void F2DArray<Type>::Alloc(const unsigned int &iSize) {
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
	m_pArray 	= new Type[iSize];
//	m_pSubArray = NULL;
	std::memset( m_pArray, 0, iSize*sizeof(Type) );
	m_bAllocated = true;
}

template <class Type>
void F2DArray<Type>::Alloc(const unsigned int &iX, const unsigned int &iY) {
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
	m_pArray 	= new Type[iX*iY];
//	m_pSubArray = new Type[iY];
	std::memset( m_pArray, 0, iX*iY*sizeof(Type) );
	m_bAllocated = true;
}

template <class Type>
unsigned int F2DArray<Type>::GetW() const {
	return m_iX;
}

template <class Type>
unsigned int F2DArray<Type>::GetH() const {
	return m_iY;
}

template <class Type>
unsigned int F2DArray<Type>::GetTotalSize() const {
	return m_iY * m_iX;
}

template <class Type>
std::vector<Type> F2DArray<Type>::GetSubArrayX(const unsigned int &iY) const {
	assert(iY < m_iY);

	std::vector<Type> vSubArray(GetW() );
	std::copy(&m_pArray[iY*m_iX], &m_pArray[iY*m_iX]+GetW(), vSubArray.begin() );
	return vSubArray; //return &m_pArray[iY*m_iX];
}

template <class Type>
std::vector<Type> F2DArray<Type>::GetSubArrayY(const unsigned int &iX) const {
	assert(iX < m_iX);

	std::vector<Type> vSubArray(GetH() );
	for(unsigned int y = 0; y < m_iY; y++) {
		vSubArray[y] = GetValue(iX, y);
	}
	return vSubArray;
}

template <class Type>
void F2DArray<Type>::SetSubArrayX(const unsigned int &iY, const std::vector<Type> &vRow) {
	assert(iY < m_iY);
	assert(vRow.size() == m_iX);

	memcpy(&m_pArray[iY*m_iX], &vRow.data()[0], m_iX*sizeof(Type) );
}

template <class Type>
void F2DArray<Type>::SetSubArrayY(const unsigned int &iX, const std::vector<Type> &vCol) {
	assert(iX < m_iX);
	assert(vCol.size() == m_iY);

	for(unsigned int y = 0; y < m_iY; y++) {
		SetValue(iX, y, vCol.at(y) );
	}
}

template <class Type>
void F2DArray<Type>::SetValue(const unsigned int &iX, const unsigned int &iY, const Type &fVal) {
	assert(iY < m_iY);
	assert(iX < m_iX);

	m_pArray[iX + iY*m_iX] = fVal;
}

template <class Type>
Type F2DArray<Type>::GetValue(const unsigned int &iX, const unsigned int &iY) const {
	assert(iY < m_iY);
	assert(iX < m_iX);

	return m_pArray[iX + iY*m_iX];
}

template <class Type>
void F2DArray<Type>::SetArray(const unsigned int &iSizeX, const unsigned int &iSizeY, const Type &fVal) {
	Alloc(iSizeX, iSizeY);
	for(unsigned int y = 0; y < m_iY; y++) {
		for(unsigned int x = 0; x < m_iX; x++) {
			SetValue(x, y, fVal);
		}
	}
}

template <class Type>
void F2DArray<Type>::SetArray(const unsigned int &iSizeX, const unsigned int &iSizeY, Type *pArray) {
	assert( pArray != NULL );

	m_pArray = pArray;
	m_iX = iSizeX;
	m_iY = iSizeY;
}

template <class Type>
Type *F2DArray<Type>::GetArray() const {
	return m_pArray;
}


template <class Type>
F2DArray<Type>::operator Type*() {
	return m_pArray;
}

template <class Type>
F2DArray<Type>::operator const Type*() const {
	return m_pArray;
}

template <class Type>
Type *F2DArray<Type>::operator[] (int iY) {
	return &m_pArray[iY*m_iX];
}

template <class Type>
const Type *F2DArray<Type>::operator[] (int iY) const {
	return &m_pArray[iY*m_iX];
}

template <class Type>
void F2DArray<Type>::GetOutput() {
	for(unsigned int y = 0; y < GetH(); y++) {
		for(unsigned int x = 0; x < GetW(); x++) {
			std::cout << "Array["<<x<<"]["<<y<<"]=" << GetValue(x, y) << std::endl;
		}
	}
}

#ifdef __CUDACC__
template <class Type>
F2DArray<Type>::F2DArray(const ANNGPGPU::F2DArray<Type> &mat) {
	unsigned int iHeight 	= mat.GetH();
	unsigned int iWidth 	= mat.GetW();

	Alloc(iWidth, iHeight);

	for(unsigned int y = 0; y < iHeight; y++) {
		for(unsigned int x = 0; x < iWidth; x++) {
			m_pArray[y*iWidth+x] = mat[y*iWidth+x];
		}
	}
}

template <class Type>
ANNGPGPU::F2DArray<Type> F2DArray<Type>::ToDevice() const {
	ANNGPGPU::F2DArray<Type> dmRes(GetW(), GetH(), 0.f);

	for(unsigned int y = 0; y < GetH(); y++) {
		for(unsigned int x = 0; x < GetW(); x++) {
			dmRes[y*GetW()+x] = m_pArray[y*GetW()+x];
		}
	}

	return dmRes;
}
#endif
