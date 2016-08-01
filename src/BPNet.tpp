/*
 * Net.cpp
 *
 *  Created on: 30.05.2009
 *      Author: Daniel <dgrat> Frenzel
 */


template <class Type, class Functor>
bool smallestFunctor(AbsLayer<Type> *i, AbsLayer<Type> *j) {
	return ( ((ANN::BPLayer<Type, Functor>*)i)->GetZLayer() < ((ANN::BPLayer<Type, Functor>*)j)->GetZLayer() );
}

template <class Type, class Functor>
BPNet<Type, Functor>::BPNet() {
	this->m_fTypeFlag = ANNetBP;
}

template <class Type, class Functor>
BPNet<Type, Functor>::BPNet(BPNet<Type, Functor> *pNet) //: AbsNet(pNet)
{
	assert( pNet != NULL );
	*this = *pNet;
}

template <class Type, class Functor>
BPNet<Type, Functor>::~BPNet() {
}

template <class Type, class Functor>
void BPNet<Type, Functor>::AddLayer(const unsigned int &iSize, const LayerTypeFlag &flType) {
	this->AddLayer( new BPLayer<Type, Functor>(iSize, flType, -1) );
}

template <class Type, class Functor>
void BPNet<Type, Functor>::CreateNet(const ConTable<Type> &Net) {
	std::cout<<"Create BPNet"<<std::endl;

	/*
	 * Init
	 */
	unsigned int iDstNeurID = 0;
	unsigned int iDstLayerID = 0;
	unsigned int iSrcLayerID = 0;

	Type fEdgeValue = 0.f;

	AbsLayer<Type> *pDstLayer = NULL;
	AbsLayer<Type> *pSrcLayer = NULL;
	AbsNeuron<Type> *pDstNeur = NULL;
	AbsNeuron<Type> *pSrcNeur = NULL;

	/*
	 * For all nets necessary: Create Connections (Edges)
	 */
	AbsNet<Type>::CreateNet(Net);

	/*
	 * Support z-layers
	 */
	for(unsigned int i = 0; i < this->m_lLayers.size(); i++) {
		BPLayer<Type, Functor> *curLayer = ((BPLayer<Type, Functor>*)this->GetLayer(i) );
		curLayer->SetZLayer(Net.ZValOfLayer[i]);
	}
}

template <class Type, class Functor>
void BPNet<Type, Functor>::AddLayer(BPLayer<Type, Functor> *pLayer) {
	AbsNet<Type>::AddLayer(pLayer);

	if( ( (BPLayer<Type, Functor>*)pLayer)->GetFlag() & ANLayerInput ) {
		this->m_pIPLayer = pLayer;
	}
	else if( ( (BPLayer<Type, Functor>*)pLayer)->GetFlag() & ANLayerOutput ) {
		this->m_pOPLayer = pLayer;
	}
	else {
	}
}

/*
 * TODO better use of copy constructors
 */
template <class Type, class Functor>
BPNet<Type, Functor> *BPNet<Type, Functor>::GetSubNet(const unsigned int &iStartID, const unsigned int &iStopID) {
	assert( iStopID < this->GetLayers().size() );
	assert( iStartID >= 0 );

	/*
	 * Return value
	 */
	BPNet<Type, Functor> *pNet = new BPNet<Type, Functor>;

	/*
	 * Create layers like in pNet
	 */
	for(unsigned int i = iStartID; i <= iStopID; i++) {
		BPLayer<Type, Functor> *pLayer = new BPLayer<Type, Functor>((BPLayer<Type, Functor>*)this->GetLayer(i), -1);
		
		if( i == iStartID && !(( (BPLayer<Type, Functor>*)this->GetLayer(i) )->GetFlag() & ANLayerInput) )
			pLayer->AddFlag( ANLayerInput );
		if( i == iStopID && !(( (BPLayer<Type, Functor>*)this->GetLayer(i) )->GetFlag() & ANLayerOutput) )
			pLayer->AddFlag( ANLayerOutput );

		pNet->AbsNet<Type>::AddLayer( pLayer );
	}

	BPLayer<Type, Functor> *pCurLayer;
	AbsNeuron<Type> *pCurNeuron;
	Edge<Type> *pCurEdge;
	for(unsigned int i = iStartID; i <= iStopID; i++) { 	// layers ..
		// NORMAL NEURON
		pCurLayer = ( (BPLayer<Type, Functor>*)this->GetLayer(i) );
		for(unsigned int j = 0; j < pCurLayer->GetNeurons().size(); j++) { 		// neurons ..
			pCurNeuron = pCurLayer->GetNeurons().at(j);
			AbsNeuron<Type> *pSrcNeuron = ( (BPLayer<Type, Functor>*)pNet->GetLayer(i) )->GetNeuron(j);
			for(unsigned int k = 0; k < pCurNeuron->GetConsO().size(); k++) { 			// edges ..
				pCurEdge = pCurNeuron->GetConO(k);

				// get iID of the destination neuron of the (next) layer i+1 (j is iID of (the current) layer i)
				int iDestNeurID 	= pCurEdge->GetDestinationID(pCurNeuron);
				int iDestLayerID 	= pCurEdge->GetDestination(pCurNeuron)->GetParent()->GetID();

				// copy edge
				AbsNeuron<Type> *pDstNeuron = pNet->GetLayers().at(iDestLayerID)->GetNeuron(iDestNeurID);

				// create edge
				ANN::Connect( pSrcNeuron, pDstNeuron,
						pCurEdge->GetValue(),
						pCurEdge->GetMomentum(),
						pCurEdge->GetAdaptationState() );
			}
		}
	}

	// Import further properties
	pNet->SetTrainingSet(this->GetTrainingSet() );
	pNet->Setup(this->m_Setup);

	return pNet;
}

template <class Type, class Functor>
void BPNet<Type, Functor>::PropagateFW() {
	for(unsigned int i = 1; i < this->m_lLayers.size(); i++) {
		BPLayer<Type, Functor> *curLayer = ((BPLayer<Type, Functor>*)this->GetLayer(i) );
		//#pragma omp parallel for
		for(int j = 0; j < static_cast<int>(curLayer->GetNeurons().size() ); j++) {
			curLayer->GetNeuron(j)->CalcValue();
		}
	}
}

template <class Type, class Functor>
void BPNet<Type, Functor>::PropagateBW() {
	for(int i = this->m_lLayers.size()-1; i >= 0; i--) {
		BPLayer<Type, Functor> *curLayer = ( (BPLayer<Type, Functor>*)this->GetLayer(i) );
		//#pragma omp parallel for
		for(int j = 0; j < static_cast<int>( curLayer->GetNeurons().size() ); j++) {
			curLayer->GetNeuron(j)->AdaptEdges();
		}
	}
}

template <class Type, class Functor>
std::vector<Type> BPNet<Type, Functor>::TrainFromData(const unsigned int &iCycles, const Type &fTolerance, const bool &bBreak, Type &fProgress) {
	bool bZSort = false;
	for(int i = 0; i < this->m_lLayers.size(); i++) {
		if(((BPLayer<Type, Functor>*)this->m_lLayers[i])->GetZLayer() > -1)
			bZSort = true;
		else {
			bZSort = false;
			break;
		}
	}
	if(bZSort) {
		std::sort(this->m_lLayers.begin(), this->m_lLayers.end(), smallestFunctor<Type, Functor>);
		// The IDs of the layers must be set according to Z-Value
		for(int i = 0; i < this->m_lLayers.size(); i++) {
			this->m_lLayers.at(i)->SetID(i);
		}
	}

	return AbsNet<Type>::TrainFromData(iCycles, fTolerance, bBreak, fProgress);
}

template <class Type, class Functor>
void BPNet<Type, Functor>::Setup(const HebbianConf<Type> &config) {
	m_Setup = config;
	#pragma omp parallel for
	for(int i = 0; i < static_cast<int>(this->m_lLayers.size() ); i++) {
		( (BPLayer<Type, Functor>*)this->GetLayer(i) )->Setup(config);
	}
}
