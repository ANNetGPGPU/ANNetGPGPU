/*
 * HFLayer.cpp
 *
 *  Created on: 22.02.2011
 *      Author: dgrat
 */

//#include <math.h>
#include <cassert>
#include "HFLayer.h"
#include "HFNeuron.h"
#include "Edge.h"

using namespace ANN;


HFLayer::HFLayer() {
	m_iWidth 	= 0;
	m_iHeight 	= 0;

	SetFlag(ANLayerInput | ANLayerOutput);
}

HFLayer::HFLayer(const unsigned int &iWidth, const unsigned int &iHeight) {
	m_iWidth 	= iWidth;
	m_iHeight 	= iHeight;

	SetFlag(ANLayerInput | ANLayerOutput);

	Resize(iWidth, iHeight);
}

HFLayer::~HFLayer() {
	// TODO Auto-generated destructor stub
}

unsigned int HFLayer::GetWidth() {
	return m_iWidth;
}

unsigned int HFLayer::GetHeight() {
	return m_iHeight;
}

void HFLayer::Resize(const unsigned int &iSize) {
	EraseAll();
	AddNeurons(iSize);
}

void HFLayer::Resize(const unsigned int &iWidth, const unsigned int &iHeight) {
	EraseAll();

	m_iWidth 	= iWidth;
	m_iHeight 	= iHeight;

	for(unsigned int y = 0; y < iHeight; y++) {			// Height
		for(unsigned int x = 0; x < iWidth; x++) {			// Width
			HFNeuron *pNeuron = new HFNeuron(this);
			pNeuron->SetID(y*iWidth + x);
			m_lNeurons.push_back(pNeuron);
		}
	}
}

void HFLayer::AddNeurons(const unsigned int &iSize) {
	for(unsigned int x = 0; x < iSize; x++) {			// Width
		HFNeuron *pNeuron = new HFNeuron(this);
		m_lNeurons.push_back(pNeuron);
		pNeuron->SetID(m_lNeurons.size()-1);
	}
}

HFNeuron *HFLayer::GetNeuron(const unsigned int &iX, const unsigned int &iY) const {
	return (HFNeuron *)m_lNeurons.at(iY * m_iWidth + iX);
}

void HFLayer::ClearWeights() {
	AbsNeuron *pNeuron 	= NULL;
	Edge 		*pEdge 		= NULL;

	for(unsigned int i = 0; i < GetNeurons().size(); i++) {				// iLength
		pNeuron = AbsLayer::GetNeuron(i);
		for(unsigned int j = 0; j < pNeuron->GetConsI().size(); j++) {	//iLength
			if(i == j)													// skip neuron
				continue;

			pEdge = pNeuron->GetConI(j);
			pEdge->SetValue( 0.f );
		}
	}
}

void HFLayer::ConnectLayer(const float *pEdges, bool bAllowAdapt) {
	EraseAllEdges();

	AbsNeuron *pSrcNeuron;
	AbsNeuron *pDstNeuron;

	for(unsigned int y = 0; y < GetNeurons().size(); y++) {		// Height
		pSrcNeuron = AbsLayer::GetNeuron(y);
		if(pSrcNeuron == NULL)
			return;

		for(unsigned int x = 0; x < GetNeurons().size(); x++) {	// Width
			pDstNeuron = AbsLayer::GetNeuron(x);

			if(x == y)	{							// When reaching the same neuron
				continue;							// .. continue
			}
			else if(pDstNeuron == NULL)	{ 			// Break if fail somewhere
				return;
			}
			else {									// .. but if there is no fail continue
				// pSrcNeuron gets incoming connections from all other neurons
				Connect(pDstNeuron, pSrcNeuron, pEdges[y*GetNeurons().size()+x], 0.f, bAllowAdapt);
			}
		}
	}
}

void HFLayer::ConnectLayer(bool bAllowAdapt) {
	EraseAllEdges();

	AbsNeuron *pSrcNeuron;
	AbsNeuron *pDstNeuron;

	for(unsigned int y = 0; y < m_iHeight; y++) {		// Height
		for(unsigned int x = 0; x < m_iWidth; x++) {	// Width
			pSrcNeuron = GetNeuron(x, y);

			for(unsigned int Y = 0; Y < m_iHeight; Y++) {		// Height
				for(unsigned int X = 0; X < m_iWidth; X++) {	// Width
					// don't connect a neuron with itself
					if(y == Y && x == X)
						continue;

					pDstNeuron = GetNeuron(X, Y);
					// break when something went wrong
					assert(pSrcNeuron != NULL);
					assert(pDstNeuron != NULL);
					// pSrcNeuron gets incoming connections from all other neurons
					Connect(pDstNeuron, pSrcNeuron, 0.f, 0.f, bAllowAdapt);
				}
			}
		}
	}
}

