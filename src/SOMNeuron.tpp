template <class Type>
SOMNeuron<Type>::SOMNeuron(SOMLayer<Type> *parent) : AbsNeuron<Type>(parent) {
	m_fLearningRate = 0.5f;
	m_fConscience = 0.f;

	// gives neuron random coordinates
	for(unsigned int i = 0; i < this->m_vPosition.size(); i++) {
		int iMax = parent->GetDim(i) * 10;
		this->m_vPosition[i] = GetRandReal<Type>(0, iMax);
	}
}

template <class Type>
void SOMNeuron<Type>::ExpToFS(BZFILE* bz2out, int iBZ2Error) {
	std::cout<<"Save SOMNeuron to FS()"<<std::endl;
	AbsNeuron<Type>::ExpToFS(bz2out, iBZ2Error);

	BZ2_bzWrite( &iBZ2Error, bz2out, &m_fLearningRate, sizeof(Type) );
	BZ2_bzWrite( &iBZ2Error, bz2out, &m_fSigma0, sizeof(Type) );
}

template <class Type>
void SOMNeuron<Type>::ImpFromFS(BZFILE* bz2in, int iBZ2Error, ConTable<Type> &Table) {
	std::cout<<"Load SOMNeuron to FS()"<<std::endl;
	AbsNeuron<Type>::ImpFromFS(bz2in, iBZ2Error, Table);

	Type 	fLearningRate; // learning rate
	Type	fSigma0; // inital distance bias to get activated
	
	BZ2_bzRead( &iBZ2Error, bz2in, &fLearningRate, sizeof(Type) );
	BZ2_bzRead( &iBZ2Error, bz2in, &fSigma0, sizeof(Type) );
	
	Table.Neurons.back().m_vMisc.push_back(fLearningRate);
	Table.Neurons.back().m_vMisc.push_back(fSigma0);
}

template <class Type>
void SOMNeuron<Type>::AdaptEdges() {
	Edge<Type> *pEdge = NULL;
	Type fInput = 0.f;
	Type fWeight = 0.f;

	for(unsigned int i = 0; i < this->GetConsI().size(); i++) {
		pEdge 	= this->GetConI(i);
		fWeight = *pEdge;
		fInput 	= *pEdge->GetDestination(this);
		pEdge->SetValue(fWeight + (m_fInfluence*m_fLearningRate*(fInput-fWeight) ) );
	}
}

template <class Type>
Type SOMNeuron<Type>::GetSigma0() {
	return m_fSigma0;
}

template <class Type>
void SOMNeuron<Type>::SetSigma0(const Type &fVal) {
	if(fVal < 0.f) {
		return;
	}
	m_fSigma0 = fVal;
}

template <class Type>
void SOMNeuron<Type>::CalcValue() {
	// TODO
}

template <class Type>
void SOMNeuron<Type>::CalcDistance2Inp() {
	this->m_fValue = 0.f;
	for (unsigned int i = 0; i < this->GetConsI().size(); ++i) {
		this->m_fValue += std::pow(*this->GetConI(i)->GetDestination(this) - *this->GetConI(i), 2);	// both have a Type() operator!
	}
	this->m_fValue = std::sqrt(this->m_fValue);
}

template <class Type>
Type SOMNeuron<Type>::GetLearningRate() const {
	return m_fLearningRate;
}

template <class Type>
void SOMNeuron<Type>::SetLearningRate(const Type &fVal) {
	m_fLearningRate = fVal;
}

template <class Type>
Type SOMNeuron<Type>::GetInfluence() const {
	return m_fInfluence;
}

template <class Type>
void SOMNeuron<Type>::SetInfluence(const Type &fVal) {
	m_fInfluence = fVal;
}

template <class Type>
Type SOMNeuron<Type>::GetDistance2Neur(const SOMNeuron &pNeurDst) {
	assert(this->GetPosition().size() == pNeurDst.GetPosition().size() );

	Type fDist = 0.f;
	for(unsigned int i = 0; i < this->GetPosition().size(); i++) {
		fDist += pow(pNeurDst.GetPosition().at(i) - this->GetPosition().at(i), 2);
	}
	return sqrt(fDist);
}

template <class Type>
void SOMNeuron<Type>::SetConscience(const Type &fVal) {
	m_fConscience = fVal;
}

template <class Type>
void SOMNeuron<Type>::AddConscience(const Type &fVal) {
	m_fConscience += fVal;
}

template <class Type>
Type SOMNeuron<Type>::GetConscience() {
	return m_fConscience;
}

