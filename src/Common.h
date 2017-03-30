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

#include <iostream>
#include <vector>
#include <stdio.h>
#include <stdarg.h>


namespace ANN {
inline void printf(const char * format, ...) {
#ifdef VERBOSE
	va_list arglist;
	va_start( arglist, format );
	vprintf( format, arglist );
	va_end( arglist );
#endif
}
	
template <class T> class AbsLayer;
template <class T> class AbsNeuron;

template <typename T> void SetEdgesToValue(AbsLayer<T> *pSrcLayer, AbsLayer<T> *pDestLayer, const T &fVal, const bool &bAdaptState) {
	AbsNeuron<T> *pCurNeuron;
	Edge<T> *pCurEdge;
	for(unsigned int i = 0; i < pSrcLayer->GetNeurons().size(); i++) {
		pCurNeuron = pSrcLayer->GetNeurons().at(i);
		for(unsigned int j = 0; j < pCurNeuron->GetConsO().size(); j++) {
			pCurEdge = pCurNeuron->GetConO(j);
			// outgoing edge is connected with pDestLayer ..
			if(pCurEdge->GetDestination(pCurNeuron)->GetParent() == pDestLayer) {
				// .. d adapt only these edges
				pCurEdge->SetValue( fVal );
				pCurEdge->SetAdaptationState( bAdaptState );
			}
		}
	}
}

template <typename T> void Connect(AbsNeuron<T> *pSrcNeuron, AbsNeuron<T> *pDstNeuron, const bool &bAdaptState) {
	Edge<T> *pCurEdge = new Edge<T>(pSrcNeuron, pDstNeuron);
	pCurEdge->SetAdaptationState(bAdaptState);
	pSrcNeuron->AddConO(pCurEdge);
	pDstNeuron->AddConI(pCurEdge);
}

template <typename T> void Connect(AbsNeuron<T> *pSrcNeuron, AbsNeuron<T> *pDstNeuron, const T &fVal, const T &fMomentum, const bool &bAdaptState) {
	Edge<T> *pCurEdge = new Edge<T>(pSrcNeuron, pDstNeuron, fVal, fMomentum, bAdaptState);
	pSrcNeuron->AddConO(pCurEdge);
	pDstNeuron->AddConI(pCurEdge);
}

template <typename T> void Connect(AbsNeuron<T> *pSrcNeuron, AbsLayer<T> *pDestLayer, const std::vector<T> &vValues, const std::vector<T> &vMomentums, const bool &bAdaptState) {
	unsigned int iSize = pDestLayer->GetNeurons().size();
	unsigned int iProgCount = 1;

	for(int j = 0; j < static_cast<int>(iSize); j++) {
		if(iSize >= 10) {
			if(((j+1) / (iSize/10)) == iProgCount && (j+1) % (iSize/10) == 0) {
				ANN::printf("Building connections.. Progress: %f%%/Step:%d\n", iProgCount*10.f, j+1);
				iProgCount++;
			}
		} 
		else {
			ANN::printf("Building connections.. Progress: %f%%/Step:%d\n", (T)(j+1)/(T)iSize*100.f, j+1);
		}
		ANN::Connect(pSrcNeuron, pDestLayer->GetNeuron(j), vValues[j], vMomentums[j], bAdaptState);
	}
}

template <typename T> void Connect(AbsNeuron<T> *pSrcNeuron, AbsLayer<T> *pDestLayer, const bool &bAdaptState) {
	unsigned int iSize = pDestLayer->GetNeurons().size();
	unsigned int iProgCount = 1;

	for(int j = 0; j < static_cast<int>(iSize); j++) {
		if(iSize >= 10) {
			if(((j+1) / (iSize/10)) == iProgCount && (j+1) % (iSize/10) == 0) {
				ANN::printf("Building connections.. Progress: %f%%/Step:%d\n", iProgCount*10.f, j+1);
				iProgCount++;
			}
		} 
		else {
			ANN::printf("Building connections.. Progress: %f%%/Step:%d\n", (T)(j+1)/(T)iSize*100.f, j+1);
		}
		ANN::Connect(pSrcNeuron, pDestLayer->GetNeuron(j), bAdaptState);
	}
}

};
