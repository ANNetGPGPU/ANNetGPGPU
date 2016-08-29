/*
#include "AbsNeuron.h"
#include "AbsLayer.h"
#include "Edge.h"
#include "containers/TrainingSet.h"
#include "containers/ConTable.h"
#include "math/Random.h"

using namespace ANN;
*/

template <class Type>
AbsNeuron<Type>::AbsNeuron() {
	m_fValue = GetRandReal(-0.5f, 0.5f);
	m_fErrorDelta = 0;
	m_pParentLayer = NULL;
}

template <class Type>
AbsNeuron<Type>::AbsNeuron(AbsLayer<Type> *parentLayer) : m_pParentLayer(parentLayer) {
	m_fValue = GetRandReal(-0.5f, 0.5f);
	m_fErrorDelta = 0;
}

template <class Type>
AbsNeuron<Type>::AbsNeuron(const AbsNeuron<Type> *pNeuron) {
	Type fErrorDelta 	= pNeuron->GetErrorDelta();
	Type fValue 		= pNeuron->GetValue();
	int iID 		= pNeuron->GetID();

	this->SetErrorDelta(fErrorDelta);
	this->SetValue(fValue);
	this->SetID(iID);
}

template <class Type>
AbsNeuron<Type>::~AbsNeuron() {
	EraseAll();
}

template <class Type>
void AbsNeuron<Type>::EraseAll() {
	// here we delete all edges which branch out, starting at this neuron
	for(int i = 0; i < m_lOutgoingConnections.size(); i++) {
		delete m_lOutgoingConnections[i];
	}
	
	m_lIncomingConnections.clear();
	m_lOutgoingConnections.clear();
}

template <class Type>
void AbsNeuron<Type>::AddConO(Edge<Type> *Edge) {
	m_lOutgoingConnections.push_back(Edge);
}

template <class Type>
void AbsNeuron<Type>::AddConI(Edge<Type> *Edge) {
	m_lIncomingConnections.push_back(Edge);
}

template <class Type>
void AbsNeuron<Type>::SetConO(Edge<Type> *Edge, const unsigned int iID) {
	m_lOutgoingConnections[iID] = Edge;
}

template <class Type>
void AbsNeuron<Type>::SetConI(Edge<Type> *Edge, const unsigned int iID) {
	m_lIncomingConnections[iID] = Edge;
}

template <class Type>
unsigned int AbsNeuron<Type>::GetID() const {
	return m_iNeuronID;
}

template <class Type>
void AbsNeuron<Type>::SetID(const int ID) {
	m_iNeuronID = ID;
}

template <class Type>
std::vector<Edge<Type> *> AbsNeuron<Type>::GetConsI() const{
	return m_lIncomingConnections;
}

template <class Type>
std::vector<Edge<Type> *> AbsNeuron<Type>::GetConsO() const{
	return m_lOutgoingConnections;
}

template <class Type>
Edge<Type>* AbsNeuron<Type>::GetConI(const unsigned int &pos) const {
	return m_lIncomingConnections.at(pos);
}

template <class Type>
Edge<Type>* AbsNeuron<Type>::GetConO(const unsigned int &pos) const {
	return m_lOutgoingConnections.at(pos);
}

template <class Type>
void AbsNeuron<Type>::SetValue(const Type &value) {
	m_fValue = value;
}

template <class Type>
void AbsNeuron<Type>::SetErrorDelta(const Type &value)
{
	m_fErrorDelta = value;
}

template <class Type>
const Type &AbsNeuron<Type>::GetValue() const {
	return m_fValue;
}

template <class Type>
const std::vector<Type> AbsNeuron<Type>::GetPosition() const {
	return m_vPosition;
}

template <class Type>
void AbsNeuron<Type>::SetPosition(const std::vector<Type> &vPos) {
	m_vPosition = vPos;
}

template <class Type>
const Type &AbsNeuron<Type>::GetErrorDelta() const {
	return m_fErrorDelta;
}

template <class Type>
AbsLayer<Type> *AbsNeuron<Type>::GetParent() const {
	return m_pParentLayer;
}

template <class Type>
AbsNeuron<Type>::operator Type() const {
	return this->GetValue();
}

template <class Type>
void AbsNeuron<Type>::ExpToFS(BZFILE* bz2out, int iBZ2Error) {
	unsigned int iNmbDims = this->GetPosition().size();
	unsigned int iNmbOfConnects = this->GetConsO().size();
	int iSrcNeurID = this->GetID();

	Type fEdgeValue = 0.f;
	int iDstLayerID = -1;
	int iDstNeurID = -1;

	BZ2_bzWrite( &iBZ2Error, bz2out, &iSrcNeurID, sizeof(int) );
	/*
	 * Save positions of the neurons
	 * important for SOMs
	 */
	BZ2_bzWrite( &iBZ2Error, bz2out, &iNmbDims, sizeof(int) );
	for(unsigned int k = 0; k < iNmbDims; k++) {
		Type fPos = this->GetPosition().at(k);
		BZ2_bzWrite( &iBZ2Error, bz2out, &fPos, sizeof(Type) );
	}
	/*
	 * Save data of connections
	 */
	BZ2_bzWrite( &iBZ2Error, bz2out, &iNmbOfConnects, sizeof(int) );
	for(unsigned int k = 0; k < iNmbOfConnects; k++) {
		Edge<Type> *pCurEdge = this->GetConO(k);
		iDstLayerID = pCurEdge->GetDestination(this)->GetParent()->GetID();
		iDstNeurID = pCurEdge->GetDestinationID(this);
		fEdgeValue = pCurEdge->GetValue();
		BZ2_bzWrite( &iBZ2Error, bz2out, &iDstLayerID, sizeof(int) );
		BZ2_bzWrite( &iBZ2Error, bz2out, &iDstNeurID, sizeof(int) );
		BZ2_bzWrite( &iBZ2Error, bz2out, &fEdgeValue, sizeof(Type) );
	}
}

template <class Type>
void AbsNeuron<Type>::ImpFromFS(BZFILE* bz2in, int iBZ2Error, ConTable<Type> &Table) {
	unsigned int iNmbDims = 0;
	unsigned int iNmbOfConnects = 0;

	std::vector<Type> vNeuronPos;

	Type fEdgeValue = 0.0f;
	int iDstLayerID = -1;
	int iDstNeurID = -1;
	int iSrcNeurID = -1;

	BZ2_bzRead( &iBZ2Error, bz2in, &iSrcNeurID, sizeof(int) );
	/*
	 * Save positions of the neurons
	 * important for SOMs
	 */
	BZ2_bzRead( &iBZ2Error, bz2in, &iNmbDims, sizeof(int) );
	vNeuronPos.resize(iNmbDims);
	for(unsigned int k = 0; k < iNmbDims; k++) {
		BZ2_bzRead( &iBZ2Error, bz2in, &vNeuronPos[k], sizeof(Type) );
	}
	Table.Neurons.back().m_iNeurID 		= iSrcNeurID;
	
	Table.Neurons.back().m_vMisc.push_back(vNeuronPos.size() );
	for(int i = 0; i < vNeuronPos.size(); i++) {
		Table.Neurons.back().m_vMisc.push_back(vNeuronPos.at(i) );
	}

	/*
	 * Save data of connections
	 */
	BZ2_bzRead( &iBZ2Error, bz2in, &iNmbOfConnects, sizeof(int) );
	for(unsigned int k = 0; k < iNmbOfConnects; k++) {
		BZ2_bzRead( &iBZ2Error, bz2in, &iDstLayerID, sizeof(int) );
		BZ2_bzRead( &iBZ2Error, bz2in, &iDstNeurID, sizeof(int) );
		BZ2_bzRead( &iBZ2Error, bz2in, &fEdgeValue, sizeof(Type) );
		ConDescr<Type> cCurCon;
		cCurCon.m_fVal = fEdgeValue;
		cCurCon.m_iSrcNeurID = iSrcNeurID;
		cCurCon.m_iDstNeurID = iDstNeurID;
		cCurCon.m_iSrcLayerID = Table.Neurons.back().m_iLayerID;	// current array always equal to current index, so valid
		cCurCon.m_iDstLayerID = iDstLayerID;						// last chge
		Table.NeurCons.push_back(cCurCon);
	}
}

/*
template class AbsNeuron<float>;
template class AbsNeuron<double>;
template class AbsNeuron<long double>;
template class AbsNeuron<short>;
template class AbsNeuron<int>;
template class AbsNeuron<long>;
template class AbsNeuron<long long>;
*/
