/*
 * TrainingData.cpp
 *
 *  Created on: 22.01.2011
 *      Author: dgrat
 */

#include <cassert>
#include <cstddef>
// own classes
#include "TrainingSet.h"

using namespace ANN;


TrainingSet::TrainingSet() {
	m_vInputList.resize(0);
	m_vOutputList.resize(0);
}

TrainingSet::~TrainingSet() {
	Clear();
}

void TrainingSet::AddInput(const std::vector<float> &vIn) {
	m_vInputList.push_back(vIn);
}

void TrainingSet::AddOutput(const std::vector<float> &vOut) {
	m_vOutputList.push_back(vOut);
}

void TrainingSet::AddInput(float *pIn, const unsigned int &iSize) {
	std::vector<float> vIn;
	for(unsigned int i = 0; i < iSize; i++)
		vIn.push_back(pIn[i]);
	AddInput(vIn);
}

void TrainingSet::AddOutput(float *pOut, const unsigned int &iSize) {
	std::vector<float> vOut;
	for(unsigned int i = 0; i < iSize; i++)
		vOut.push_back(pOut[i]);
	AddOutput(vOut);
}

unsigned int TrainingSet::GetNrElements() const {
	return m_vInputList.size();
}

std::vector<float> TrainingSet::GetInput(const unsigned int &iID) const {
	assert(iID < GetNrElements() );

	return m_vInputList.at(iID);
}

std::vector<float> TrainingSet::GetOutput(const unsigned int &iID) const {
	assert(iID < GetNrElements() );

	return m_vOutputList.at(iID);
}

void TrainingSet::Clear() {
	m_vInputList.clear();
	m_vOutputList.clear();
}

void TrainingSet::ExpToFS(BZFILE* bz2out, int iBZ2Error) {
	unsigned int iNrInpE = m_vInputList.size();
	unsigned int iNrOutE = m_vOutputList.size();

	BZ2_bzWrite( &iBZ2Error, bz2out, &iNrInpE, sizeof(unsigned int) );
	BZ2_bzWrite( &iBZ2Error, bz2out, &iNrOutE, sizeof(unsigned int) );

	for(unsigned int i = 0; i < iNrInpE; i++) {
		std::vector<float> vInp = m_vInputList.at(i);
		unsigned int iSizeI = vInp.size();
		BZ2_bzWrite( &iBZ2Error, bz2out, &iSizeI, sizeof(unsigned int) );

		for(unsigned int j = 0; j < iSizeI; j++) {
			float fVal = vInp.at(j);
			BZ2_bzWrite( &iBZ2Error, bz2out, &fVal, sizeof(float) );
		}
	}
	for(unsigned int i = 0; i < iNrOutE; i++) {
		std::vector<float> vOut = m_vOutputList.at(i);
		unsigned int iSizeO = vOut.size();
		BZ2_bzWrite( &iBZ2Error, bz2out, &iSizeO, sizeof(unsigned int) );

		for(unsigned int j = 0; j < iSizeO; j++) {
			float fVal = vOut.at(j);
			BZ2_bzWrite( &iBZ2Error, bz2out, &fVal, sizeof(float) );
		}
	}
}

void TrainingSet::ImpFromFS(BZFILE* bz2in, int iBZ2Error) {
	Clear();

	unsigned int iNrInpE = 0;
	unsigned int iNrOutE = 0;

	BZ2_bzRead( &iBZ2Error, bz2in, &iNrInpE, sizeof(unsigned int) );
	BZ2_bzRead( &iBZ2Error, bz2in, &iNrOutE, sizeof(unsigned int) );

	for(unsigned int i = 0; i < iNrInpE; i++) {
		std::vector<float> vInp;
		unsigned int iSizeI = 0;
		BZ2_bzRead( &iBZ2Error, bz2in, &iSizeI, sizeof(unsigned int) );

		for(unsigned int j = 0; j < iSizeI; j++) {
			float fVal = 0.f;
			BZ2_bzRead( &iBZ2Error, bz2in, &fVal, sizeof(float) );
			vInp.push_back(fVal);
		}
		m_vInputList.push_back(vInp);
	}
	for(unsigned int i = 0; i < iNrOutE; i++) {
		std::vector<float> vOut;
		unsigned int iSizeO = 0;
		BZ2_bzRead( &iBZ2Error, bz2in, &iSizeO, sizeof(unsigned int) );

		for(unsigned int j = 0; j < iSizeO; j++) {
			float fVal = 0.f;
			BZ2_bzRead( &iBZ2Error, bz2in, &fVal, sizeof(float) );
			vOut.push_back(fVal);
		}
		m_vOutputList.push_back(vOut);
	}

}
