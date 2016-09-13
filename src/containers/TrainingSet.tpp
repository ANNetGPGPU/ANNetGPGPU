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
TrainingSet<Type>::TrainingSet() {
	m_vInputList.resize(0);
	m_vOutputList.resize(0);
}

template <class Type>
TrainingSet<Type>::~TrainingSet() {
	Clear();
}

template <class Type>
void TrainingSet<Type>::AddInput(const std::vector<Type> &vIn) {
	m_vInputList.push_back(vIn);
}

template <class Type>
void TrainingSet<Type>::AddOutput(const std::vector<Type> &vOut) {
	m_vOutputList.push_back(vOut);
}

template <class Type>
void TrainingSet<Type>::AddInput(Type *pIn, const unsigned int &iSize) {
	std::vector<Type> vIn;
	for(unsigned int i = 0; i < iSize; i++)
		vIn.push_back(pIn[i]);
	AddInput(vIn);
}

template <class Type>
void TrainingSet<Type>::AddOutput(Type *pOut, const unsigned int &iSize) {
	std::vector<Type> vOut;
	for(unsigned int i = 0; i < iSize; i++)
		vOut.push_back(pOut[i]);
	AddOutput(vOut);
}

template <class Type>
unsigned int TrainingSet<Type>::GetNrElements() const {
	return m_vInputList.size();
}

template <class Type>
std::vector<Type> TrainingSet<Type>::GetInput(const unsigned int &iID) const {
	assert(iID < GetNrElements() );

	return m_vInputList.at(iID);
}

template <class Type>
std::vector<Type> TrainingSet<Type>::GetOutput(const unsigned int &iID) const {
	assert(iID < GetNrElements() );

	return m_vOutputList.at(iID);
}

template <class Type>
void TrainingSet<Type>::Clear() {
	m_vInputList.clear();
	m_vOutputList.clear();
}

template <class Type>
void TrainingSet<Type>::ExpToFS(BZFILE* bz2out, int iBZ2Error) {
	unsigned int iNrInpE = m_vInputList.size();
	unsigned int iNrOutE = m_vOutputList.size();

	BZ2_bzWrite( &iBZ2Error, bz2out, &iNrInpE, sizeof(unsigned int) );
	BZ2_bzWrite( &iBZ2Error, bz2out, &iNrOutE, sizeof(unsigned int) );

	for(unsigned int i = 0; i < iNrInpE; i++) {
		std::vector<Type> vInp = m_vInputList.at(i);
		unsigned int iSizeI = vInp.size();
		BZ2_bzWrite( &iBZ2Error, bz2out, &iSizeI, sizeof(unsigned int) );

		for(unsigned int j = 0; j < iSizeI; j++) {
			Type fVal = vInp.at(j);
			BZ2_bzWrite( &iBZ2Error, bz2out, &fVal, sizeof(Type) );
		}
	}
	for(unsigned int i = 0; i < iNrOutE; i++) {
		std::vector<Type> vOut = m_vOutputList.at(i);
		unsigned int iSizeO = vOut.size();
		BZ2_bzWrite( &iBZ2Error, bz2out, &iSizeO, sizeof(unsigned int) );

		for(unsigned int j = 0; j < iSizeO; j++) {
			Type fVal = vOut.at(j);
			BZ2_bzWrite( &iBZ2Error, bz2out, &fVal, sizeof(Type) );
		}
	}
}

template <class Type>
void TrainingSet<Type>::ImpFromFS(BZFILE* bz2in, int iBZ2Error) {
	Clear();

	unsigned int iNrInpE = 0;
	unsigned int iNrOutE = 0;

	BZ2_bzRead( &iBZ2Error, bz2in, &iNrInpE, sizeof(unsigned int) );
	BZ2_bzRead( &iBZ2Error, bz2in, &iNrOutE, sizeof(unsigned int) );

	for(unsigned int i = 0; i < iNrInpE; i++) {
		std::vector<Type> vInp;
		unsigned int iSizeI = 0;
		BZ2_bzRead( &iBZ2Error, bz2in, &iSizeI, sizeof(unsigned int) );

		for(unsigned int j = 0; j < iSizeI; j++) {
			Type fVal = 0.f;
			BZ2_bzRead( &iBZ2Error, bz2in, &fVal, sizeof(Type) );
			vInp.push_back(fVal);
		}
		m_vInputList.push_back(vInp);
	}
	for(unsigned int i = 0; i < iNrOutE; i++) {
		std::vector<Type> vOut;
		unsigned int iSizeO = 0;
		BZ2_bzRead( &iBZ2Error, bz2in, &iSizeO, sizeof(unsigned int) );

		for(unsigned int j = 0; j < iSizeO; j++) {
			Type fVal = 0.f;
			BZ2_bzRead( &iBZ2Error, bz2in, &fVal, sizeof(Type) );
			vOut.push_back(fVal);
		}
		m_vOutputList.push_back(vOut);
	}
}
