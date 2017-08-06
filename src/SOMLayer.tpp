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
SOMLayer<Type>::SOMLayer() {

}

template <class Type>
SOMLayer<Type>::SOMLayer(const SOMLayer<Type> *pLayer) {
	int iNumber = pLayer->GetNeurons().size();
	LayerTypeFlag fType = pLayer->GetFlag();

	Resize(iNumber);
	this->SetFlag(fType);
}

template <class Type>
SOMLayer<Type>::SOMLayer(const uint32_t &iSize, LayerTypeFlag fType) {
	Resize(iSize);
	this->SetFlag(fType);
}

template <class Type>
SOMLayer<Type>::SOMLayer(const uint32_t &iWidth, const uint32_t &iHeight, LayerTypeFlag fType) {
	Resize(iWidth, iHeight);
	this->SetFlag(fType);
}

template <class Type>
SOMLayer<Type>::SOMLayer(const std::vector<uint32_t> &vDim, LayerTypeFlag fType) {
	Resize(vDim);
	this->SetFlag(fType);
}

template <class Type>
void SOMLayer<Type>::AddNeurons(const uint32_t &iSize) {
	std::vector<Type> vPos(1);
	for(uint32_t x = 0; x < iSize; x++) {
		SOMNeuron<Type> *pNeuron = new SOMNeuron<Type>(this);
		this->m_lNeurons.push_back(pNeuron);
		pNeuron->SetID(this->m_lNeurons.size()-1);

		vPos[0] = this->m_lNeurons.size()-1;
		pNeuron->SetPosition(vPos);
	}
}

template <class Type>
void SOMLayer<Type>::Resize(const uint32_t &iSize) {
	this->EraseAll();

	std::vector<Type> vPos(1);
	for(uint32_t x = 0; x < iSize; x++) {
		SOMNeuron<Type> *pNeuron = new SOMNeuron<Type>(this);
		this->m_lNeurons.push_back(pNeuron);
		pNeuron->SetID(this->m_lNeurons.size()-1);

		vPos[0] = this->m_lNeurons.size()-1;
		pNeuron->SetPosition(vPos);
	}
}

template <class Type>
void SOMLayer<Type>::Resize(const uint32_t &iWidth, const uint32_t &iHeight) {
	this->EraseAll();

	// Set m_vDim properly
	m_vDim.clear();
	m_vDim.push_back(iWidth);
	m_vDim.push_back(iHeight);

	std::vector<Type> vPos(2);
	for(uint32_t y = 0; y < iHeight; y++) {
		for(uint32_t x = 0; x < iWidth; x++) {
			SOMNeuron<Type> *pNeuron = new SOMNeuron<Type>(this);
			pNeuron->SetID(y*iWidth + x);
			this->m_lNeurons.push_back(pNeuron);

			vPos[0] = x; vPos[1] = y;
			pNeuron->SetPosition(vPos);
		}
	}
}

template <class Type>
void SOMLayer<Type>::Resize(const std::vector<uint32_t> &vDim) {
	this->EraseAll();

	assert(vDim.size() > 0);

	m_vDim = vDim;

	uint32_t iSize = 1;
	for(uint32_t i = 0; i < vDim.size(); i++) {
		iSize *= vDim[i];
	}
	Resize(iSize);
}

template <class Type>
void SOMLayer<Type>::ConnectLayer(AbsLayer<Type> *pDestLayer, const bool &bAllowAdapt) {
	AbsNeuron<Type> *pSrcNeuron = nullptr;

	/*
	 * Vernetze jedes Neuron dieser Schicht mit jedem Neuron in "pDestLayer"
	 */
	for(int i = 0; i < static_cast<int>(this->m_lNeurons.size() ); i++) {
		ANN::printf("Connect input neuron %d to output layer.. %d/%d\n", i, i+1, this->m_lNeurons.size());
		pSrcNeuron = this->m_lNeurons[i];
		
		if(pSrcNeuron != nullptr) {
			Connect(pSrcNeuron, pDestLayer, bAllowAdapt);
		}
	}
}

template <class Type>
void SOMLayer<Type>::ConnectLayer(AbsLayer<Type> *pDestLayer, const F2DArray<Type> &f2dEdgeMat, const bool &bAllowAdapt) {
	AbsNeuron<Type> *pSrcNeuron = nullptr;

	// Connect each neuron in this laer with eery neuron in "pDestLayer"
	std::vector<Type> fMoms(f2dEdgeMat.GetH(), 0);	// TODO not used by SOMs
	std::vector<Type> fVals(f2dEdgeMat.GetH(), 0);

	for(int i = 0; i < static_cast<int>(this->m_lNeurons.size() ); i++) {
		ANN::printf("Connect input neuron %d to output layer.. %d/%d\n", i, i+1, this->m_lNeurons.size());
		pSrcNeuron = this->m_lNeurons[i];

		fVals = f2dEdgeMat.GetSubArrayX(i);
		if(pSrcNeuron != nullptr) {
			Connect(pSrcNeuron, pDestLayer, fVals, fMoms, bAllowAdapt);
		}
	}
}

template <class Type>
std::vector<float> SOMLayer<Type>::GetPosition(const uint32_t iNeuronID) {
	AbsNeuron<Type> *pSrcNeuron = this->m_lNeurons[iNeuronID];
	if(pSrcNeuron == nullptr) {
		return std::vector<float>();
	}
	return pSrcNeuron->GetPosition();
}

template <class Type>
void SOMLayer<Type>::SetLearningRate(const float &fVal) {
	#pragma omp parallel for
	for(int j = 0; j < static_cast<int>( this->m_lNeurons.size() ); j++) {
		static_cast<SOMNeuron<Type>*>(this->m_lNeurons[j])->SetLearningRate(fVal);
	}
}

template <class Type>
std::vector<uint32_t> SOMLayer<Type>::GetDim() const {
	return m_vDim;
}

template <class Type>
uint32_t SOMLayer<Type>::GetDim(const uint32_t &iInd) const {
	return m_vDim.at(iInd);
}
