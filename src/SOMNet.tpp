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
SOMNet<Type, Functor>::SOMNet(AbsNet<Type> *pNet) {
	this->m_fTypeFlag = ANNetSOM;
	if(pNet == nullptr) return;

	std::vector<uint32_t> vDimI = ((SOMLayer<Type>*)(pNet->GetIPLayer() ))->GetDim();
	std::vector<uint32_t> vDimO = ((SOMLayer<Type>*)(pNet->GetOPLayer() ))->GetDim();

	// Copy weights between neurons of the input and output layer
	ANN::F2DArray<Type> f2dEdges = pNet->GetOPLayer()->ExpEdgesIn();
	// Copy positions of the neurons in the output layer
	ANN::F2DArray<Type> f2dPosistions = pNet->GetOPLayer()->ExpPositions();

	// Create the net finally
	CreateSOM(vDimI, vDimO, f2dEdges, f2dPosistions);
	// Copy training set
	ANN::AbsNet<Type>::SetTrainingSet(pNet->GetTrainingSet() );
}

template<class Type, class Functor>
SOMNet<Type, Functor>::SOMNet( const std::vector<uint32_t> &vDimI, 
			       const std::vector<uint32_t> &vDimO) {
	SetLearningRate(0.5f);
	this->m_fTypeFlag = ANNetSOM;
	CreateSOM(vDimI, vDimO);
}

template<class Type, class Functor>
SOMNet<Type, Functor>::SOMNet( const uint32_t &iWidthI, const uint32_t &iHeightI,
			       const uint32_t &iWidthO, const uint32_t &iHeightO) 
{
	SetLearningRate(0.5f);
	this->m_fTypeFlag = ANNetSOM;
	CreateSOM(iWidthI, iHeightI, iWidthO, iHeightO);
}

template<class Type, class Functor>
void SOMNet<Type, Functor>::AddLayer(AbsLayer<Type> *pLayer) {
	AbsNet<Type>::AddLayer(pLayer);
}

template<class Type, class Functor>
AbsLayer<Type> *SOMNet<Type, Functor>::AddLayer(const uint32_t &iSize, const LayerTypeFlag &flType) {
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
	for(uint32_t i = 0; i < Net.Neurons.size(); i++) {
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
	uint32_t iSize = pLayer->GetNeurons().size();

	uint32_t iDim = pNeuron->GetPosition().size();
	std::vector<Type> vDimMax(iDim, 0.f);
	std::vector<Type> vDimMin(iDim, std::numeric_limits<Type>::max() );

	// look in all the nodes
	for(uint32_t i = 0; i < iSize; i++) {
		pNeuron = (SOMNeuron<Type>*)pLayer->GetNeuron(i);
		// find the smallest and greatest positions in the network
		for(uint32_t j = 0; j < iDim; j++) {
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
void SOMNet<Type, Functor>::CreateSOM(const std::vector<uint32_t> &vDimI, const std::vector<uint32_t> &vDimO) {
	if(this->m_pIPLayer != nullptr || this->m_pOPLayer != nullptr) {
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
void SOMNet<Type, Functor>::CreateSOM(const std::vector<uint32_t> &vDimI, const std::vector<uint32_t> &vDimO,
		const F2DArray<Type> &f2dEdgeMat, const F2DArray<Type> &f2dNeurPos) {
	if(this->m_pIPLayer != nullptr || this->m_pOPLayer != nullptr) {
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
	static_cast<SOMLayer<Type>*>(this->m_pIPLayer)->ConnectLayer(this->m_pOPLayer, f2dEdgeMat);
	this->m_pOPLayer->ImpPositions(f2dNeurPos);

	// find sigma0
	FindSigma0();
}

template<class Type, class Functor>
void SOMNet<Type, Functor>::CreateSOM(	const uint32_t &iWidthI, const uint32_t &iHeightI,
					const uint32_t &iWidthO, const uint32_t &iHeightO)
{
	if(this->m_pIPLayer != nullptr || this->m_pOPLayer != nullptr) {
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
void SOMNet<Type, Functor>::TrainHelper(uint32_t i) {
	assert(i < this->GetTrainingSet()->GetNrElements() );
	
	this->SetInput(this->GetTrainingSet()->GetInput(i) );

	// Present the input vector to each node and determine the BMU
	this->FindBMNeuron();
	
	// Adjust the weight vector of the BMU and its neighbors
	this->PropagateBW();
}

template<class Type, class Functor>
void SOMNet<Type, Functor>::Training(const uint32_t &iCycles, const TrainingMode &eMode) {
	assert(iCycles > 0);
	
	if(this->GetTrainingSet() == nullptr) {
		ANN::printf("No training set available\n");
		return;
	}
	
	m_iCycles = iCycles;
	int iMin = 0;
	int iMax = this->GetTrainingSet()->GetNrElements()-1;
	uint32_t iProgCount = 1;

	ANN::printf("Start calculation now\n");
	for(m_iCycle = 0; m_iCycle < static_cast<uint32_t>(m_iCycles); m_iCycle++) {
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
			uint32_t iRandID = GetRandInt(iMin, iMax);
			TrainHelper(iRandID);
		}
		// The input vectors are presented to the network in serial order
		else if(eMode == ANN::ANSerialMode) {
			for(uint32_t i = 0; i < this->GetTrainingSet()->GetNrElements(); i++) {
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
std::vector<ANN::Centroid<Type>> SOMNet<Type, Functor>::FindCentroids() {
	std::vector<Centroid<Type>> vCentroids;
	for(uint32_t i = 0; i < this->GetTrainingSet()->GetNrElements(); i++) {
		// Present the input vector to each node and determine the BMU
		this->SetInput(this->GetTrainingSet()->GetInput(i) );
		FindBMNeuron();

		ANN::Centroid<Type> centr;
		centr._unitID = m_pBMNeuron->GetID();
		for(uint32_t j = 0; j < m_pBMNeuron->GetConsI().size(); j++) {
			centr._edges.push_back(m_pBMNeuron->GetConI(j)->GetValue());
		}
		centr._input = this->GetTrainingSet()->GetInput(i);
		centr._distance = m_pBMNeuron->GetValue();
		
		vCentroids.push_back(centr);
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
	assert(this->m_pIPLayer != nullptr && this->m_pOPLayer != nullptr);

	Type fCurVal = 0.f;
	Type fSmallest = std::numeric_limits<Type>::max();

	//#pragma omp parallel for
	for(int i = 0; i < static_cast<int>(this->m_pOPLayer->GetNeurons().size() ); i++) {
		SOMNeuron<Type> *pNeuron = (SOMNeuron<Type>*)this->m_pOPLayer->GetNeuron(i);
		pNeuron->CalcDistance2Inp();
		fCurVal = pNeuron->GetValue();

		if(fSmallest > fCurVal) {
			fSmallest = fCurVal;
			m_pBMNeuron = pNeuron;
		}
	}
	
	// end of implementation of conscience mechanism
	assert(m_pBMNeuron != nullptr);
}

template<class Type, class Functor>
std::vector<Type> SOMNet<Type, Functor>::GetPosition(const uint32_t iNeuronID) {
	std::vector<Type> vPos = ((ANN::SOMLayer<Type>*)this->m_pOPLayer)->GetPosition(iNeuronID);
	return vPos;
}

