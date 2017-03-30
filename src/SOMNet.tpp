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

template<class Type, class Functor>
SOMNet<Type, Functor>::SOMNet() {
	this->m_pIPLayer = NULL;
	this->m_pOPLayer = NULL;
	
	m_pBMNeuron = NULL;
	m_iCycle = 0;
	m_iWidthI = 0.f;
	m_iHeightI = 0.f;
	m_iWidthO = 0.f;
	m_iHeightO = 0.f;
	
	// Conscience mechanism
	m_fConscienceRate 	= 0.f;
	this->m_fTypeFlag 	= ANNetSOM;
}

template<class Type, class Functor>
SOMNet<Type, Functor>::SOMNet(AbsNet<Type> *pNet) {
	if(pNet == NULL)
		return;

	std::vector<unsigned int> vDimI = ((SOMLayer<Type>*)(pNet->GetIPLayer() ))->GetDim();
	std::vector<unsigned int> vDimO = ((SOMLayer<Type>*)(pNet->GetOPLayer() ))->GetDim();

	// Copy weights between neurons of the input and output layer
	ANN::F2DArray<Type> f2dEdges = pNet->GetOPLayer()->ExpEdgesIn();
	// Copy positions of the neurons in the output layer
	ANN::F2DArray<Type> f2dPosistions = pNet->GetOPLayer()->ExpPositions();
	// Create the net finally
	CreateSOM(vDimI, vDimO, f2dEdges, f2dPosistions);
	// Copy training set
	ANN::AbsNet<Type>::SetTrainingSet(pNet->GetTrainingSet() );

	this->m_fTypeFlag 	= ANNetSOM;
}

template<class Type, class Functor>
SOMNet<Type, Functor>::SOMNet(const std::vector<unsigned int> &vDimI, const std::vector<unsigned int> &vDimO) {
	this->m_pIPLayer = NULL;
	this->m_pOPLayer = NULL;
	m_pBMNeuron = NULL;

	m_iCycle = 0;
	SetLearningRate(0.5f);

	m_iWidthI = 0.f;
	m_iHeightI = 0.f;
	m_iWidthO = 0.f;
	m_iHeightO = 0.f;
	
	// Conscience mechanism
	m_fConscienceRate = 0.f;

	this->m_fTypeFlag = ANNetSOM;
	
	CreateSOM(vDimI, vDimO);
}

template<class Type, class Functor>
SOMNet<Type, Functor>::SOMNet(	const unsigned int &iWidthI, const unsigned int &iHeightI,
		const unsigned int &iWidthO, const unsigned int &iHeightO) 
{
	this->m_pIPLayer = NULL;
	this->m_pOPLayer = NULL;
	m_pBMNeuron = NULL;

	m_iCycle = 0;
	SetLearningRate(0.5f);

	m_iWidthI = 0.f;
	m_iHeightI = 0.f;
	m_iWidthO = 0.f;
	m_iHeightO = 0.f;
	
	// Conscience mechanism
	m_fConscienceRate = 0.f;

	this->m_fTypeFlag = ANNetSOM;
	
	CreateSOM(iWidthI, iHeightI, iWidthO, iHeightO);
}

template<class Type, class Functor>
void SOMNet<Type, Functor>::AddLayer(AbsLayer<Type> *pLayer) {
	AbsNet<Type>::AddLayer(pLayer);
}

template<class Type, class Functor>
AbsLayer<Type> *SOMNet<Type, Functor>::AddLayer(const unsigned int &iSize, const LayerTypeFlag &flType) {
	AbsLayer<Type> *pRet = new SOMLayer<Type>(iSize, flType);
	AbsNet<Type>::AddLayer(pRet);
	return pRet;
}

template<class Type, class Functor>
void SOMNet<Type, Functor>::CreateNet(const ConTable<Type> &Net) {
	ANN::printf("Create SOMNet\n");
	
	/*
	 * For all nets necessary: Create Connections (Edges)
	 */
	AbsNet<Type>::CreateNet(Net);

	/*
	 * Set Positions
	 */
	for(unsigned int i = 0; i < Net.Neurons.size(); i++) {
		int iLayerID 	= Net.Neurons.at(i).m_iLayerID;
		int iNeurID 	= Net.Neurons.at(i).m_iNeurID;
		
		// Get position
		int iPosSize = Net.Neurons.at(i).m_vMisc.at(0);
		std::vector<Type> vPos(iPosSize);
		for(int j = 0; j < iPosSize; j++) {
			vPos[j] = Net.Neurons.at(i).m_vMisc[1+j];
		}
		
		// Save other information of the neuron
		ANN::SOMNeuron<Type> *pNeuron = (ANN::SOMNeuron<Type> *)this->GetLayer(iLayerID)->GetNeuron(iNeurID);
		pNeuron->SetPosition(vPos);
		pNeuron->SetLearningRate(Net.Neurons.at(i).m_vMisc[iPosSize+1]);
		pNeuron->SetSigma0(Net.Neurons.at(i).m_vMisc[iPosSize+2]);
	}
}

template<class Type, class Functor>
void SOMNet<Type, Functor>::FindSigma0() {
	SOMLayer<Type> *pLayer  = (SOMLayer<Type>*)this->GetOPLayer();
	SOMNeuron<Type> *pNeuron = (SOMNeuron<Type>*)pLayer->GetNeuron(0);
	unsigned int iSize = pLayer->GetNeurons().size();

	unsigned int iDim = pNeuron->GetPosition().size();
	std::vector<Type> vDimMax(iDim, 0.f);
	std::vector<Type> vDimMin(iDim, std::numeric_limits<Type>::max() );

	// look in all the nodes
	for(unsigned int i = 0; i < iSize; i++) {
		pNeuron = (SOMNeuron<Type>*)pLayer->GetNeuron(i);
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
	Type fSigma0 = *(vDimMax.end()-1)+1 - *(vDimMin.begin()+1);
	fSigma0 /= 2.f;
	
	// Apply Sigma0 to all neurons
	SetSigma0(fSigma0);
}

template<class Type, class Functor>
void SOMNet<Type, Functor>::SetSigma0(const Type &fVal) {
	if(fVal < 0.f) {
		return;
	}

	#pragma omp parallel for
	for(int i = 0; i < static_cast<int>(this->m_pOPLayer->GetNeurons().size() ); i++) {
		((SOMNeuron<Type>*)this->m_pOPLayer->GetNeuron(i))->SetSigma0(fVal);
	}
}

template<class Type, class Functor>
void SOMNet<Type, Functor>::CreateSOM(const std::vector<unsigned int> &vDimI, const std::vector<unsigned int> &vDimO) {
	if(this->m_pIPLayer != NULL || this->m_pOPLayer != NULL) {
		AbsNet<Type>::EraseAll();
	}

	ANN::printf("Create input layer\n");
	this->m_pIPLayer = new SOMLayer<Type>(vDimI, ANLayerInput);
	this->m_pIPLayer->SetID(0);
	AbsNet<Type>::AddLayer(this->m_pIPLayer);

	ANN::printf("Create output layer\n");
	this->m_pOPLayer = new SOMLayer<Type>(vDimO, ANLayerOutput);
	this->m_pOPLayer->SetID(1);
	AbsNet<Type>::AddLayer(this->m_pOPLayer);

	ANN::printf("Connect layers ..\n");
	((SOMLayer<Type>*)this->m_pIPLayer)->ConnectLayer(this->m_pOPLayer);

	// find sigma0
	FindSigma0();
}

template<class Type, class Functor>
void SOMNet<Type, Functor>::CreateSOM(const std::vector<unsigned int> &vDimI, const std::vector<unsigned int> &vDimO,
		const F2DArray<Type> &f2dEdgeMat, const F2DArray<Type> &f2dNeurPos) {
	if(this->m_pIPLayer != NULL || this->m_pOPLayer != NULL) {
		AbsNet<Type>::EraseAll();
	}

	ANN::printf("Create input layer\n");
	this->m_pIPLayer = new SOMLayer<Type>(vDimI, ANLayerInput);
	this->m_pIPLayer->SetID(0);
	AbsNet<Type>::AddLayer(this->m_pIPLayer);

	ANN::printf("Create output layer\n");
	this->m_pOPLayer = new SOMLayer<Type>(vDimO, ANLayerOutput);
	this->m_pOPLayer->SetID(1);
	AbsNet<Type>::AddLayer(this->m_pOPLayer);

	ANN::printf("Connect layers ..\n");
	((SOMLayer<Type>*)this->m_pIPLayer)->ConnectLayer(this->m_pOPLayer, f2dEdgeMat);

	this->m_pOPLayer->ImpPositions(f2dNeurPos);

	// find sigma0
	FindSigma0();
}

template<class Type, class Functor>
void SOMNet<Type, Functor>::CreateSOM(	const unsigned int &iWidthI, const unsigned int &iHeightI,
						const unsigned int &iWidthO, const unsigned int &iHeightO)
{
	if(this->m_pIPLayer != NULL || this->m_pOPLayer != NULL) {
		AbsNet<Type>::EraseAll();
	}

	m_iWidthI = iWidthI;
	m_iHeightI = iHeightI;
	m_iWidthO = iWidthO;
	m_iHeightO = iHeightO;

	ANN::printf("Create input layer\n");
	this->m_pIPLayer = new SOMLayer<Type>(iWidthI, iHeightI, ANLayerInput);
	this->m_pIPLayer->SetID(0);
	AbsNet<Type>::AddLayer(this->m_pIPLayer);

	ANN::printf("Create output layer\n");
	this->m_pOPLayer = new SOMLayer<Type>(iWidthO, iHeightO, ANLayerOutput);
	this->m_pOPLayer->SetID(1);
	AbsNet<Type>::AddLayer(this->m_pOPLayer);

	ANN::printf("Connect layers ..\n");
	((SOMLayer<Type>*)this->m_pIPLayer)->ConnectLayer(this->m_pOPLayer);

	// find sigma0
	FindSigma0();
}

template<class Type, class Functor>
void SOMNet<Type, Functor>::TrainHelper(unsigned int i) {
	assert(i < this->GetTrainingSet()->GetNrElements() );
	
	this->SetInput(this->GetTrainingSet()->GetInput(i) );

	// Present the input vector to each node and determine the BMU
	this->FindBMNeuron();
	
	// Adjust the weight vector of the BMU and its neighbors
	this->PropagateBW();
}

template<class Type, class Functor>
void SOMNet<Type, Functor>::Training(const unsigned int &iCycles, const TrainingMode &eMode) {
	assert(iCycles > 0);
	
	if(this->GetTrainingSet() == NULL) {
		ANN::printf("No training set available\n");
		return;
	}
	
	m_iCycles = iCycles;
	int iMin = 0;
	int iMax = this->GetTrainingSet()->GetNrElements()-1;
	unsigned int iProgCount = 1;

	ANN::printf("Start calculation now\n");
	for(m_iCycle = 0; m_iCycle < static_cast<unsigned int>(m_iCycles); m_iCycle++) {
		if(m_iCycles >= 10) {
			if(((m_iCycle+1) / (m_iCycles/10)) == iProgCount && (m_iCycle+1) % (m_iCycles/10) == 0) {
				ANN::printf("Current training progress calculated by the CPU is %f%%/Step: %d/%d\n", iProgCount*10.f, m_iCycle+1, m_iCycles);
				iProgCount++;
			}
		} else {
			ANN::printf("Current training progress calculated by the CPU is %f%%/Step: %d/%d\n", (Type)(m_iCycle+1.f)/(Type)m_iCycles*100.f, m_iCycle+1, m_iCycles);
		}

		// The input vectors are presented to the network at random
		if(eMode == ANN::ANRandomMode) {
			unsigned int iRandID = GetRandInt(iMin, iMax);
			TrainHelper(iRandID);
		}
		// The input vectors are presented to the network in serial order
		else if(eMode == ANN::ANSerialMode) {
			for(unsigned int i = 0; i < this->GetTrainingSet()->GetNrElements(); i++) {
				TrainHelper(i);
			}
		}
	}
}

template<class Type, class Functor>
void SOMNet<Type, Functor>::PropagateBW() {
	// Run through neurons
	#pragma omp parallel for
	for(int i = 0; i < static_cast<int>(this->m_pOPLayer->GetNeurons().size() ); i++) {		
		// Set some values used below ..
		SOMNeuron<Type> *pNeuron 	= (SOMNeuron<Type>*)this->m_pOPLayer->GetNeuron(i);
		
		Type fLambda = this->m_iCycles / log(pNeuron->GetSigma0() ); // time constant
		Type fSigmaT = this->m_DistFunction.rad_decay(pNeuron->GetSigma0(), this->m_iCycle, fLambda);		
		Type fLearningRateT = this->m_DistFunction.lrate_decay(pNeuron->GetLearningRate(), this->m_iCycle, this->m_iCycles);
		Type fDist = pNeuron->GetDistance2Neur(*m_pBMNeuron);

		if(fDist < fSigmaT) {
			//calculate by how much weights get adjusted ..
			Type fInfluence = this->m_DistFunction.distance(fDist, fSigmaT);
			pNeuron->SetInfluence(fInfluence);
			// .. and adjust them
			pNeuron->AdaptEdges();
		}
	}
}

template<class Type, class Functor>
std::vector<Centroid<Type>> SOMNet<Type, Functor>::GetCentroidList() {
	std::vector<Centroid<Type>> vCentroids;
	for(unsigned int i = 0; i < this->GetTrainingSet()->GetNrElements(); i++) {
		vCentroids.push_back(Centroid<Type>() );
		this->SetInput(this->GetTrainingSet()->GetInput(i) );

		// Present the input vector to each node and determine the BMU
		FindBMNeuron();
		
		vCentroids[i].m_iBMUID 	= m_pBMNeuron->GetID();
		vCentroids[i].m_vCentroid.clear();
		for(unsigned int j = 0; j < m_pBMNeuron->GetConsI().size(); j++) {
			vCentroids[i].m_vCentroid.push_back(m_pBMNeuron->GetConI(j)->GetValue());
		}
		vCentroids[i].m_vInput = this->GetTrainingSet()->GetInput(i);
		vCentroids[i].m_fEucDist = m_pBMNeuron->GetValue();
	}
	return vCentroids;
}

template<class Type, class Functor>
void SOMNet<Type, Functor>::PropagateFW() {
	// TODO
}

template<class Type, class Functor>
void SOMNet<Type, Functor>::SetLearningRate(const Type &fVal) {
	#pragma omp parallel for
	for(int i = 0; i < static_cast<int>(this->m_lLayers.size() ); i++) {
		( (SOMLayer<Type>*)this->GetLayer(i) )->SetLearningRate(fVal);
	}
}

template<class Type, class Functor>
void SOMNet<Type, Functor>::FindBMNeuron() {
	assert(this->m_pIPLayer != NULL && this->m_pOPLayer != NULL);

	Type fCurVal 	= 0.f;
	Type fSmallest = std::numeric_limits<Type>::max();
	Type fNrOfNeurons 	= (Type)(this->m_pOPLayer->GetNeurons().size() );

	//#pragma omp parallel for
	for(int i = 0; i < static_cast<int>(this->m_pOPLayer->GetNeurons().size() ); i++) {
		SOMNeuron<Type> *pNeuron = (SOMNeuron<Type>*)this->m_pOPLayer->GetNeuron(i);
		pNeuron->CalcDistance2Inp();
		fCurVal = pNeuron->GetValue();

		// with implementation of conscience mechanism (2nd term)
		Type fConscienceBias = 1.f/fNrOfNeurons - pNeuron->GetConscience();
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
	//Type fConscience = m_fConscienceRate * (m_pBMNeuron->GetValue() - m_pBMNeuron->GetConscience() ); // standard implementation seems to have some problems
	//m_pBMNeuron->AddConscience(fConscience); // standard implementation seems to have some problems
	if(m_fConscienceRate > 0.f) {
		#pragma omp parallel for
		for(int i = 0; i < static_cast<int>(this->m_pOPLayer->GetNeurons().size() ); i++) {
			SOMNeuron<Type> *pNeuron = (SOMNeuron<Type>*)this->m_pOPLayer->GetNeuron(i);
			Type fConscience = m_fConscienceRate * (pNeuron->GetValue() - pNeuron->GetConscience() );
			pNeuron->SetConscience(fConscience);
		}
	}
	// end of implementation of conscience mechanism
	assert(m_pBMNeuron != NULL);
}

template<class Type, class Functor>
void SOMNet<Type, Functor>::SetConscienceRate(const Type &fVal) {
	m_fConscienceRate = fVal;
}

template<class Type, class Functor>
Type SOMNet<Type, Functor>::GetConscienceRate() {
	return m_fConscienceRate;
}

template<class Type, class Functor>
std::vector<Type> SOMNet<Type, Functor>::GetPosition(const unsigned int iNeuronID) {
	std::vector<Type> vPos = ((ANN::SOMLayer<Type>*)this->m_pOPLayer)->GetPosition(iNeuronID);
	return vPos;
}

