/*
 * AbsNeuron.cpp
 *
 *  Created on: 01.09.2010
 *      Author: dgrat
 */

#include <iostream>
#include <stdio.h>
#include <cassert>
//own classes
#include "math/Functions.h"
#include "math/Random.h"
#include "Edge.h"
#include "AbsNeuron.h"
#include "BPLayer.h"
#include "containers/TrainingSet.h"
#include "containers/ConTable.h"

using namespace ANN;


AbsNeuron::AbsNeuron(AbsLayer *parentLayer) : m_pParentLayer(parentLayer) {
	/*
	 * Weise dem Neuron eine Zufallszahl zwischen 0 und 1 zu
	 * Genauigkeit liegt bei 4 Nachkommastellen
	 */
	m_fValue = RandFloat(-0.5f, 0.5f);

	m_fErrorDelta = 0;
	m_pBias = NULL;
}

AbsNeuron::AbsNeuron(const AbsNeuron *pNeuron) {
	float fErrorDelta 	= pNeuron->GetErrorDelta();
	float fValue 		= pNeuron->GetValue();
	float iID 			= pNeuron->GetID();

	this->SetErrorDelta(fErrorDelta);
	this->SetValue(fValue);
	this->SetID(iID);
}

AbsNeuron::~AbsNeuron() {
	m_lIncomingConnections.clear();
	m_lOutgoingConnections.clear();
}

void AbsNeuron::EraseAllEdges() {
	// TODO
/*
	for(int i = 0; i < m_lIncomingConnections.size(); i++) {
		if(m_lIncomingConnections[i] == NULL)
			continue;
		else {
			delete m_lIncomingConnections[i];
			m_lIncomingConnections[i] = NULL;
		}
	}
	for(int i = 0; i < m_lOutgoingConnections.size(); i++) {
		if(m_lOutgoingConnections[i] == NULL)
			continue;
		else {
			delete m_lOutgoingConnections[i];
			m_lOutgoingConnections[i] = NULL;
		}
	}
*/
	m_lIncomingConnections.clear();
	m_lOutgoingConnections.clear();
}

void AbsNeuron::AddConO(Edge *Edge) {
	m_lOutgoingConnections.push_back(Edge);
}

void AbsNeuron::AddConI(Edge *Edge) {
	m_lIncomingConnections.push_back(Edge);
}

void AbsNeuron::SetConO(Edge *Edge, const unsigned int iID) {
	m_lOutgoingConnections[iID] = Edge;
}

void AbsNeuron::SetConI(Edge *Edge, const unsigned int iID) {
	m_lIncomingConnections[iID] = Edge;
}

/*
void AbsNeuron::SetConO(Edge *Edge, const unsigned int iID) {
	std::list<ANN::Edge*>::iterator it;
	it = m_lOutgoingConnections.begin();
	for(unsigned int i = 0; i < iID; i++) {
		it++;
	}
	*it = Edge;
}

void AbsNeuron::SetConI(Edge *Edge, const unsigned int iID) {
	std::list<ANN::Edge*>::iterator it;
	it = m_lIncomingConnections.begin();
	for(unsigned int i = 0; i < iID; i++) {
		it++;
	}
	*it = Edge;
}
*/

unsigned int AbsNeuron::GetID() const {
	return m_iNeuronID;
}

void AbsNeuron::SetID(const int ID) {
	m_iNeuronID = ID;
}

/*
ANN::list<Edge*> AbsNeuron::GetConsI() const{
	return m_lIncomingConnections;
}
ANN::list<Edge*> AbsNeuron::GetConsO() const{
	return m_lOutgoingConnections;
}
*/

std::vector<Edge*> AbsNeuron::GetConsI() const{
	return m_lIncomingConnections;
}
std::vector<Edge*> AbsNeuron::GetConsO() const{
	return m_lOutgoingConnections;
}

Edge* AbsNeuron::GetConI(const unsigned int &pos) const {
	return m_lIncomingConnections.at(pos);
}

Edge* AbsNeuron::GetConO(const unsigned int &pos) const {
	return m_lOutgoingConnections.at(pos);
}

/*
Edge* AbsNeuron::GetConI(const unsigned int &iID) {
	std::list<ANN::Edge*>::iterator it;
	it = m_lIncomingConnections.begin();
	for(unsigned int i = 0; i < iID; i++) {
		it++;
	}
	return *it;
}

Edge* AbsNeuron::GetConO(const unsigned int &iID) {
	std::list<ANN::Edge*>::iterator it;
	it = m_lOutgoingConnections.begin();
	for(unsigned int i = 0; i < iID; i++) {
		it++;
	}
	return *it;
}
*/

void AbsNeuron::SetValue(const float &value) {
	m_fValue = value;
}

void AbsNeuron::SetErrorDelta(const float &value)
{
	m_fErrorDelta = value;
}

void AbsNeuron::SetBiasEdge(Edge *Edge) {
	m_pBias = Edge;
}

const float &AbsNeuron::GetValue() const {
	return m_fValue;
}

const std::vector<float> AbsNeuron::GetPosition() const {
	return m_vPosition;
}

void AbsNeuron::SetPosition(const std::vector<float> &vPos) {
	m_vPosition = vPos;
}

const float &AbsNeuron::GetErrorDelta() const {
	return m_fErrorDelta;
}

Edge *AbsNeuron::GetBiasEdge() const {
	return m_pBias;
}

AbsLayer *AbsNeuron::GetParent() const {
	return m_pParentLayer;
}

void AbsNeuron::SetTransfFunction (const TransfFunction *pFCN) {
	this->m_ActFunction = const_cast<TransfFunction *>(pFCN);
}

const TransfFunction *AbsNeuron::GetTransfFunction() const {
	return (m_ActFunction);
}

AbsNeuron::operator float() const {
	return GetValue();
}

void AbsNeuron::ExpToFS(BZFILE* bz2out, int iBZ2Error) {
	unsigned int iNmbDims 		= GetPosition().size();
	unsigned int iNmbOfConnects = GetConsO().size();
	int iSrcNeurID 				= GetID();

	float fEdgeValue 	= 0.f;
	int iDstLayerID 	= -1;
	int iDstNeurID 		= -1;

	BZ2_bzWrite( &iBZ2Error, bz2out, &iSrcNeurID, sizeof(int) );
	/*
	 * Save positions of the neurons
	 * important for SOMs
	 */
	BZ2_bzWrite( &iBZ2Error, bz2out, &iNmbDims, sizeof(int) );
	for(unsigned int k = 0; k < iNmbDims; k++) {
		float fPos = GetPosition().at(k);
		BZ2_bzWrite( &iBZ2Error, bz2out, &fPos, sizeof(float) );
	}
	/*
	 * Save data of connections
	 */
	BZ2_bzWrite( &iBZ2Error, bz2out, &iNmbOfConnects, sizeof(int) );
	for(unsigned int k = 0; k < iNmbOfConnects; k++) {
		Edge *pCurEdge = GetConO(k);
		iDstLayerID = pCurEdge->GetDestination(this)->GetParent()->GetID();
		iDstNeurID 	= pCurEdge->GetDestinationID(this);
		fEdgeValue 	= pCurEdge->GetValue();
		BZ2_bzWrite( &iBZ2Error, bz2out, &iDstLayerID, sizeof(int) );
		BZ2_bzWrite( &iBZ2Error, bz2out, &iDstNeurID, sizeof(int) );
		BZ2_bzWrite( &iBZ2Error, bz2out, &fEdgeValue, sizeof(float) );
	}
}

void AbsNeuron::ImpFromFS(BZFILE* bz2in, int iBZ2Error, ConTable &Table) {
	unsigned int 	iNmbDims 		= 0;
	unsigned int 	iNmbOfConnects 	= 0;

	std::vector<float> vNeuronPos;

	float fEdgeValue 	= 0.0f;
	int iDstLayerID 	= -1;
	int iDstNeurID 		= -1;
	int iSrcNeurID 		= -1;

	BZ2_bzRead( &iBZ2Error, bz2in, &iSrcNeurID, sizeof(int) );
	/*
	 * Save positions of the neurons
	 * important for SOMs
	 */
	BZ2_bzRead( &iBZ2Error, bz2in, &iNmbDims, sizeof(int) );
	vNeuronPos.resize(iNmbDims);
	for(unsigned int k = 0; k < iNmbDims; k++) {
		BZ2_bzRead( &iBZ2Error, bz2in, &vNeuronPos[k], sizeof(float) );
	}
	Table.Neurons.back().m_iNeurID 		= iSrcNeurID;
	Table.Neurons.back().m_vPos 		= vNeuronPos;
	/*
	 * Save data of connections
	 */
	BZ2_bzRead( &iBZ2Error, bz2in, &iNmbOfConnects, sizeof(int) );
	for(unsigned int k = 0; k < iNmbOfConnects; k++) {
		BZ2_bzRead( &iBZ2Error, bz2in, &iDstLayerID, sizeof(int) );
		BZ2_bzRead( &iBZ2Error, bz2in, &iDstNeurID, sizeof(int) );
		BZ2_bzRead( &iBZ2Error, bz2in, &fEdgeValue, sizeof(float) );
		ConDescr cCurCon;
		cCurCon.m_fVal 			= fEdgeValue;
		cCurCon.m_iSrcNeurID 	= iSrcNeurID;
		cCurCon.m_iDstNeurID 	= iDstNeurID;
		cCurCon.m_iSrcLayerID 	= Table.Neurons.back().m_iLayerID;	// current array always equal to current index, so valid
		cCurCon.m_iDstLayerID 	= iDstLayerID;						// last chge
		Table.NeurCons.push_back(cCurCon);
	}
}

namespace ANN {
	/*
	 * AUSGABEOPERATOR
	 * OSTREAM
	 */
	std::ostream& operator << (std::ostream &os, AbsNeuron &op)
	{
	//	os << "Data of Neuron: " 									<< std::endl;
		os << "Value: \t" 		<< op.GetValue() 				<< std::endl;
	//    os << "Error delta: \t" << op.GetErrorDelta() 				<< std::endl;
	//    os << "Connections of Neuron:" 								<< std::endl;
	//    os << "Incoming: " 		<< op.GetConnectionsIn().size() 	<< std::endl;
	//    os << "Outgoing: " 		<< op.GetConnectionsOut().size() 	<< std::endl;
		return os;     // Ref. auf Stream
	}

	std::ostream& operator << (std::ostream &os, AbsNeuron *op)
	{
	//	os << "Data of Neuron: " 									<< std::endl;
		os << "Value: \t" 		<< op->GetValue() 			<< std::endl;
	//    os << "Delta: \t" << op->GetErrorDelta() 				<< std::endl;
	//    os << "Connections of Neuron:" 								<< std::endl;
	//    os << "Incoming: " 		<< op->GetConnectionsIn().size() 	<< std::endl;
	//    os << "Outgoing: " 		<< op->GetConnectionsOut().size() 	<< std::endl;
		return os;     // Ref. auf Stream
	}

	/*STATIC:*/
	void Connect(AbsNeuron *pSrcNeuron, AbsNeuron *pDstNeuron, const bool &bAdaptState) {
		Edge *pCurEdge = new Edge(pSrcNeuron, pDstNeuron);

		pCurEdge->SetAdaptationState(bAdaptState);
		pSrcNeuron->AddConO(pCurEdge);				// Edge beiden zuweisen
		pDstNeuron->AddConI(pCurEdge);
	}

	void Connect(AbsNeuron *pSrcNeuron, AbsNeuron *pDstNeuron, const float &fVal, const float &fMomentum, const bool &bAdaptState) {
		Edge *pCurEdge = new Edge(pSrcNeuron, pDstNeuron, fVal, fMomentum, bAdaptState);

		pSrcNeuron->AddConO(pCurEdge);				// Edge beiden zuweisen
		pDstNeuron->AddConI(pCurEdge);
	}

	void Connect(AbsNeuron *pSrcNeuron, AbsLayer *pDestLayer, const bool &bAdaptState) {
		unsigned int iSize 		= pDestLayer->GetNeurons().size();
		unsigned int iProgCount = 1;

		for(int j = 0; j < static_cast<int>(iSize); j++) {
			// Output
			if(iSize >= 10) {
				if(((j+1) / (iSize/10)) == iProgCount && (j+1) % (iSize/10) == 0) {
					std::cout<<"Building connections.. Progress: "<<iProgCount*10.f<<"%/Step="<<j+1<<std::endl;
					iProgCount++;
				}
			} else {
				std::cout<<"Building connections.. Progress: "<<(float)(j+1)/(float)iSize*100.f<<"%/Step="<<j+1<<std::endl;
			}
			// Work job
			Connect(pSrcNeuron, pDestLayer->GetNeuron(j), bAdaptState);
		}
	}

	void Connect(AbsNeuron *pSrcNeuron, AbsLayer *pDestLayer, const std::vector<float> &vValues, const std::vector<float> &vMomentums, const bool &bAdaptState) {
		unsigned int iSize 		= pDestLayer->GetNeurons().size();
		unsigned int iProgCount = 1;

		for(int j = 0; j < static_cast<int>(iSize); j++) {
			// Output
			if(iSize >= 10) {
				if(((j+1) / (iSize/10)) == iProgCount && (j+1) % (iSize/10) == 0) {
					std::cout<<"Building connections.. Progress: "<<iProgCount*10.f<<"%/Step="<<j+1<<std::endl;
					iProgCount++;
				}
			} else {
				std::cout<<"Building connections.. Progress: "<<(float)(j+1)/(float)iSize*100.f<<"%/Step="<<j+1<<std::endl;
			}
			// Work job
			Connect(pSrcNeuron, pDestLayer->GetNeuron(j), vValues[j], vMomentums[j], bAdaptState);
		}
	}
}
