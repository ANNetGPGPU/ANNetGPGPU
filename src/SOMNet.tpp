template<class F>
SOMNet<F>::SOMNet() {
	m_pIPLayer 		= NULL;
	m_pOPLayer 		= NULL;
	m_pBMNeuron 		= NULL;

	m_iCycle 		= 0;
	

	m_iWidthI 		= 0.f;
	m_iHeightI 		= 0.f;
	m_iWidthO 		= 0.f;
	m_iHeightO 		= 0.f;
	
	// Conscience mechanism
	m_fConscienceRate 	= 0.f;

	m_fTypeFlag 	= ANNetSOM;
}

template<class F>
SOMNet<F>::SOMNet(AbsNet *pNet) {
	if(pNet == NULL)
		return;

	std::vector<unsigned int> vDimI = ((SOMLayer*)(pNet->GetIPLayer() ))->GetDim();
	std::vector<unsigned int> vDimO = ((SOMLayer*)(pNet->GetOPLayer() ))->GetDim();

	// Copy weights between neurons of the input and output layer
	ANN::F2DArray f2dEdges = pNet->GetOPLayer()->ExpEdgesIn();
	// Copy positions of the neurons in the output layer
	ANN::F2DArray f2dPosistions = pNet->GetOPLayer()->ExpPositions();
	// Create the net finally
	CreateSOM(vDimI, vDimO, f2dEdges, f2dPosistions);
	// Copy training set
	SetTrainingSet(pNet->GetTrainingSet() );

	m_fTypeFlag 	= ANNetSOM;
}

template<class F>
SOMNet<F>::SOMNet(const std::vector<unsigned int> &vDimI, const std::vector<unsigned int> &vDimO) {
  	m_pIPLayer 		= NULL;
	m_pOPLayer 		= NULL;
	m_pBMNeuron 		= NULL;

	m_iCycle 		= 0;
	SetLearningRate(0.5f);

	m_iWidthI 		= 0.f;
	m_iHeightI 		= 0.f;
	m_iWidthO 		= 0.f;
	m_iHeightO 		= 0.f;
	
	// Conscience mechanism
	m_fConscienceRate 	= 0.f;

	m_fTypeFlag 	= ANNetSOM;
	
	CreateSOM(vDimI, vDimO);
}

template<class F>
SOMNet<F>::SOMNet(	const unsigned int &iWidthI, const unsigned int &iHeightI,
		const unsigned int &iWidthO, const unsigned int &iHeightO) 
{
	m_pIPLayer 		= NULL;
	m_pOPLayer 		= NULL;
	m_pBMNeuron 		= NULL;

	m_iCycle 		= 0;
	SetLearningRate(0.5f);

	m_iWidthI 		= 0.f;
	m_iHeightI 		= 0.f;
	m_iWidthO 		= 0.f;
	m_iHeightO 		= 0.f;
	
	// Conscience mechanism
	m_fConscienceRate 	= 0.f;

	m_fTypeFlag 	= ANNetSOM;
	
	CreateSOM(iWidthI, iHeightI, iWidthO, iHeightO);
}

template<class F>
void SOMNet<F>::AddLayer(AbsLayer *pLayer) {
	AbsNet::AddLayer(pLayer);
}

template<class F>
void SOMNet<F>::AddLayer(const unsigned int &iSize, const LayerTypeFlag &flType) {
	AbsNet::AddLayer( new SOMLayer(iSize, flType) );
}

template<class F>
void SOMNet<F>::CreateNet(const ConTable &Net) {
	std::cout<<"Create SOMNet"<<std::endl;

	/*
	 * For all nets necessary: Create Connections (Edges)
	 */
	AbsNet::CreateNet(Net);

	/*
	 * Set Positions
	 */
	for(unsigned int i = 0; i < Net.Neurons.size(); i++) {
		int iLayerID 	= Net.Neurons.at(i).m_iLayerID;
		int iNeurID 	= Net.Neurons.at(i).m_iNeurID;
		
		// Get position
		int iPosSize = Net.Neurons.at(i).m_vMisc.at(0);
		std::vector<float> vPos(iPosSize);
		for(int j = 0; j < iPosSize; j++) {
			vPos[j] = Net.Neurons.at(i).m_vMisc[1+j];
		}
		
		// Save other information of the neuron
		ANN::SOMNeuron *pNeuron = (ANN::SOMNeuron *)GetLayer(iLayerID)->GetNeuron(iNeurID);
		pNeuron->SetPosition(vPos);
		pNeuron->SetLearningRate(Net.Neurons.at(i).m_vMisc[iPosSize+1]);
		pNeuron->SetSigma0(Net.Neurons.at(i).m_vMisc[iPosSize+2]);
	}
}

template<class F>
SOMNet<F>::~SOMNet() {
	// TODO Auto-generated destructor stub
}

template<class F>
void SOMNet<F>::FindSigma0() {
	SOMLayer 	*pLayer  = (SOMLayer*)GetOPLayer();
	SOMNeuron 	*pNeuron = (SOMNeuron*)pLayer->GetNeuron(0);
	unsigned int iSize 	 = pLayer->GetNeurons().size();

	unsigned int iDim = pNeuron->GetPosition().size();
	std::vector<float> vDimMax(iDim, 0.f);
	std::vector<float> vDimMin(iDim, std::numeric_limits<float>::max() );

	// look in all the nodes
	for(unsigned int i = 0; i < iSize; i++) {
		pNeuron = (SOMNeuron*)pLayer->GetNeuron(i);
		// find the smallest and greatest positions in the network
		for(unsigned int j = 0; j < iDim; j++) {
			// find greatest coordinate
			vDimMin[j] = std::min(vDimMin[j], pNeuron->GetPosition().at(j) );
			vDimMax[j] = std::max(vDimMax[j], pNeuron->GetPosition().at(j) );
		}
	}
	std::sort(vDimMin.begin(), vDimMin.end() );
	std::sort(vDimMax.begin(), vDimMax.end() );

	// save in fSigma0
	float fSigma0 = *(vDimMax.end()-1)+1 - *(vDimMin.begin()+1);
	fSigma0 /= 2.f;
	
	// Apply Sigma0 to all neurons
	SetSigma0(fSigma0);
}

template<class F>
void SOMNet<F>::SetSigma0(const float &fVal) {
	if(fVal < 0.f) {
		return;
	}

	#pragma omp parallel for
	for(int i = 0; i < static_cast<int>(m_pOPLayer->GetNeurons().size() ); i++) {
		((SOMNeuron*)m_pOPLayer->GetNeuron(i))->SetSigma0(fVal);
	}
}

template<class F>
void SOMNet<F>::CreateSOM(const std::vector<unsigned int> &vDimI, const std::vector<unsigned int> &vDimO) {
	if(m_pIPLayer != NULL || m_pOPLayer != NULL) {
		AbsNet::EraseAll();
	}

	std::cout<< "Create input layer" <<std::endl;
	m_pIPLayer = new SOMLayer(vDimI, ANLayerInput);
	m_pIPLayer->SetID(0);
	AbsNet::AddLayer(m_pIPLayer);

	std::cout<< "Create output layer" <<std::endl;
	m_pOPLayer = new SOMLayer(vDimO, ANLayerOutput);
	m_pOPLayer->SetID(1);
	AbsNet::AddLayer(m_pOPLayer);

	std::cout<< "Connect layer .." <<std::endl;
	((SOMLayer*)m_pIPLayer)->ConnectLayer(m_pOPLayer);

	// find sigma0
	FindSigma0();
}

template<class F>
void SOMNet<F>::CreateSOM(const std::vector<unsigned int> &vDimI, const std::vector<unsigned int> &vDimO,
		const F2DArray &f2dEdgeMat, const F2DArray &f2dNeurPos) {
	if(m_pIPLayer != NULL || m_pOPLayer != NULL) {
		AbsNet::EraseAll();
	}

	std::cout<< "Create input layer" <<std::endl;
	m_pIPLayer = new SOMLayer(vDimI, ANLayerInput);
	m_pIPLayer->SetID(0);
	AbsNet::AddLayer(m_pIPLayer);

	std::cout<< "Create output layer" <<std::endl;
	m_pOPLayer = new SOMLayer(vDimO, ANLayerOutput);
	m_pOPLayer->SetID(1);
	AbsNet::AddLayer(m_pOPLayer);

	std::cout<< "Connect layer .." <<std::endl;
	((SOMLayer*)m_pIPLayer)->ConnectLayer(m_pOPLayer, f2dEdgeMat);

	m_pOPLayer->ImpPositions(f2dNeurPos);

	// find sigma0
	FindSigma0();
}

template<class F>
void SOMNet<F>::CreateSOM(	const unsigned int &iWidthI, const unsigned int &iHeightI,
						const unsigned int &iWidthO, const unsigned int &iHeightO)
{
	if(m_pIPLayer != NULL || m_pOPLayer != NULL) {
		AbsNet::EraseAll();
	}

	m_iWidthI 	= iWidthI;
	m_iHeightI 	= iHeightI;
	m_iWidthO 	= iWidthO;
	m_iHeightO 	= iHeightO;

	std::cout<< "Create input layer" <<std::endl;
	m_pIPLayer = new SOMLayer(iWidthI, iHeightI, ANLayerInput);
	m_pIPLayer->SetID(0);
	AbsNet::AddLayer(m_pIPLayer);

	std::cout<< "Create output layer" <<std::endl;
	m_pOPLayer = new SOMLayer(iWidthO, iHeightO, ANLayerOutput);
	m_pOPLayer->SetID(1);
	AbsNet::AddLayer(m_pOPLayer);

	std::cout<< "Connect layer .." <<std::endl;
	((SOMLayer*)m_pIPLayer)->ConnectLayer(m_pOPLayer);

	// find sigma0
	FindSigma0();
}

template<class F>
void SOMNet<F>::TrainHelper(unsigned int i) {
	assert(i < this->GetTrainingSet()->GetNrElements() );
	
	SetInput(this->GetTrainingSet()->GetInput(i) );

	// Present the input vector to each node and determine the BMU
	this->FindBMNeuron();
	
	// Adjust the weight vector of the BMU and its neighbors
	this->PropagateBW();
}

template<class F>
void SOMNet<F>::Training(const unsigned int &iCycles, const TrainingMode &eMode) {
	assert(iCycles > 0);
	
	if(GetTrainingSet() == NULL) {
		std::cout<<"No training set available!"<<std::endl;
		return;
	}
	
	m_iCycles 	= iCycles;
	int iMin 	= 0;
	int iMax 	= GetTrainingSet()->GetNrElements()-1;
	unsigned int iProgCount = 1;

	std::cout<< "Process the SOM now" <<std::endl;
	for(m_iCycle = 0; m_iCycle < static_cast<unsigned int>(m_iCycles); m_iCycle++) {
		if(m_iCycles >= 10) {
			if(((m_iCycle+1) / (m_iCycles/10)) == iProgCount && (m_iCycle+1) % (m_iCycles/10) == 0) {
				std::cout<<"Current training progress calculated by the CPU is: "<<iProgCount*10.f<<"%/Step="<<m_iCycle+1<<std::endl;
				iProgCount++;
			}
		} else {
			std::cout<<"Current training progress calculated by the CPU is: "<<(float)(m_iCycle+1.f)/(float)m_iCycles*100.f<<"%/Step="<<m_iCycle+1<<std::endl;
		}

		// The input vectors are presented to the network at random
		if(eMode == ANN::ANRandomMode) {
			unsigned int iRandID = RandInt(iMin, iMax);
			TrainHelper(iRandID);
		}
		// The input vectors are presented to the network in serial order
		else if(eMode == ANN::ANSerialMode) {
			for(unsigned int i = 0; i < GetTrainingSet()->GetNrElements(); i++) {
				TrainHelper(i);
			}
		}
	}
}

template<class F>
void SOMNet<F>::PropagateBW() {
	// Run through neurons
	#pragma omp parallel for
	for(int i = 0; i < static_cast<int>(this->m_pOPLayer->GetNeurons().size() ); i++) {		
		// Set some values used below ..
		SOMNeuron *pNeuron 	= (SOMNeuron*)m_pOPLayer->GetNeuron(i);
		
		float fLambda 		= this->m_iCycles / log(pNeuron->GetSigma0() ); // time constant
		float fSigmaT 		= this->m_DistFunction.rad_decay(pNeuron->GetSigma0(), this->m_iCycle, fLambda);		
		float fLearningRateT 	= this->m_DistFunction.lrate_decay(pNeuron->GetLearningRate(), this->m_iCycle, this->m_iCycles);
		float fDist 		= pNeuron->GetDistance2Neur(*m_pBMNeuron);

		if(fDist < fSigmaT) {
			//calculate by how much weights get adjusted ..
			float fInfluence = this->m_DistFunction.distance(fDist, fSigmaT);
			pNeuron->SetInfluence(fInfluence);
			// .. and adjust them
			pNeuron->AdaptEdges();
		}
	}
}

template<class F>
std::vector<Centroid> SOMNet<F>::GetCentrOInpList() {
	std::vector<Centroid> vCentroids;
	for(unsigned int i = 0; i < GetTrainingSet()->GetNrElements(); i++) {
		vCentroids.push_back(Centroid() );
		SetInput(GetTrainingSet()->GetInput(i) );

		// Present the input vector to each node and determine the BMU
		FindBMNeuron();
		
		vCentroids[i].m_iBMUID 	= m_pBMNeuron->GetID();
		vCentroids[i].m_vCentroid.clear();
		for(unsigned int j = 0; j < m_pBMNeuron->GetConsI().size(); j++) {
			vCentroids[i].m_vCentroid.push_back(m_pBMNeuron->GetConI(j)->GetValue());
		}
		vCentroids[i].m_vInput = GetTrainingSet()->GetInput(i);
		vCentroids[i].m_fEucDist = sqrt(m_pBMNeuron->GetValue() );
	}
	return vCentroids;
}

template<class F>
std::vector<Centroid> SOMNet<F>::GetCentroidList() {
	std::vector<Centroid> vCentroids;
	for(unsigned int i = 0; i < GetTrainingSet()->GetNrElements(); i++) {
		vCentroids.push_back(Centroid() );
		SetInput(GetTrainingSet()->GetInput(i) );

		// Present the input vector to each node and determine the BMU
		FindBMNeuron();
		
		vCentroids[i].m_iBMUID 	= m_pBMNeuron->GetID();
		vCentroids[i].m_vCentroid.clear();
		for(unsigned int j = 0; j < m_pBMNeuron->GetConsI().size(); j++) {
			vCentroids[i].m_vCentroid.push_back(m_pBMNeuron->GetConI(j)->GetValue());
		}
		vCentroids[i].m_vInput = std::vector<float>(0);
		vCentroids[i].m_fEucDist = -1.f;
	}
	// Count the number of centroids
	//std::sort(vCentroids.begin(), vCentroids.end() );
	//vCentroids.erase(std::unique(vCentroids.begin(), vCentroids.end()), vCentroids.end() );
	std::cout<<"Number of clusters found: "<<vCentroids.size()<<std::endl;
	return vCentroids;
}

template<class F>
void SOMNet<F>::PropagateFW() {
	// TODO
}

template<class F>
void SOMNet<F>::SetLearningRate(const float &fVal) {
	#pragma omp parallel for
	for(int i = 0; i < static_cast<int>(m_lLayers.size() ); i++) {
		( (SOMLayer*)GetLayer(i) )->SetLearningRate(fVal);
	}
}

template<class F>
void SOMNet<F>::FindBMNeuron() {
	assert(m_pIPLayer != NULL && m_pOPLayer != NULL);

	float fCurVal 	= 0.f;
	float fSmallest = std::numeric_limits<float>::max();
	float fNrOfNeurons 	= (float)(m_pOPLayer->GetNeurons().size() );

	//#pragma omp parallel for
	for(int i = 0; i < static_cast<int>(m_pOPLayer->GetNeurons().size() ); i++) {
		SOMNeuron *pNeuron = (SOMNeuron*)m_pOPLayer->GetNeuron(i);
		pNeuron->CalcDistance2Inp();
		fCurVal = pNeuron->GetValue();

		// with implementation of conscience mechanism (2nd term)
		float fConscienceBias = 1.f/fNrOfNeurons - pNeuron->GetConscience();
		if(m_fConscienceRate > 0.f) {
			fCurVal -= fConscienceBias;
		}
		// end of implementation of conscience mechanism

		if(fSmallest > fCurVal) {
			fSmallest = fCurVal;
			m_pBMNeuron = pNeuron;
		}
	}

	// implementation of conscience mechanism
	//float fConscience = m_fConscienceRate * (m_pBMNeuron->GetValue() - m_pBMNeuron->GetConscience() ); 	// standard implementation seems to have some problems
	//m_pBMNeuron->AddConscience(fConscience); 																// standard implementation seems to have some problems
	if(m_fConscienceRate > 0.f) {
		#pragma omp parallel for
		for(int i = 0; i < static_cast<int>(m_pOPLayer->GetNeurons().size() ); i++) {
			SOMNeuron *pNeuron = (SOMNeuron*)m_pOPLayer->GetNeuron(i);
			float fConscience = m_fConscienceRate * (pNeuron->GetValue() - pNeuron->GetConscience() );
			pNeuron->SetConscience(fConscience);
		}
	}
	// end of implementation of conscience mechanism
	assert(m_pBMNeuron != NULL);
}

template<class F>
void SOMNet<F>::SetConscienceRate(const float &fVal) {
	m_fConscienceRate = fVal;
}

template<class F>
float SOMNet<F>::GetConscienceRate() {
	return m_fConscienceRate;
}

template<class F>
std::vector<float> SOMNet<F>::GetPosition(const unsigned int iNeuronID) {
	std::vector<float> vPos = ((ANN::SOMLayer*)m_pOPLayer)->GetPosition(iNeuronID);
	return vPos;
}

