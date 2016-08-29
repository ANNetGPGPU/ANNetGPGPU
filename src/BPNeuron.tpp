/*
 * BPNeuron.tpp
 */


template <class Type, class Functor>
BPNeuron<Type, Functor>::BPNeuron() {
	m_Setup = {0.1f, 0, 0};
}

template <class Type, class Functor>
BPNeuron<Type, Functor>::BPNeuron(AbsLayer<Type> *parentLayer) : AbsNeuron<Type>(parentLayer) {
	m_Setup = {0.1f, 0, 0};
}

template <class Type, class Functor>
BPNeuron<Type, Functor>::BPNeuron(BPNeuron *pNeuron) : AbsNeuron<Type>(pNeuron) {
}

template <class Type, class Functor>
void BPNeuron<Type, Functor>::Setup(const HebbianConf<Type> &config) {
	m_Setup = config;
}

template <class Type, class Functor>
void BPNeuron<Type, Functor>::CalcValue() {
	if(this->GetConsI().size() == 0) {
		return;
	}

	Type val = 0;
	for(unsigned int i = 0; i < this->GetConsI().size(); i++) {
		AbsNeuron<Type> *from = this->GetConI(i)->GetDestination(this);
		val += from->GetValue() * this->GetConI(i)->GetValue();
	}
	this->SetValue(val);

	val = Functor::transfer( this->GetValue(), 0.f );
	this->SetValue(val);
}

template <class Type, class Functor>
void BPNeuron<Type, Functor>::AdaptEdges() {
	if(this->GetConsO().size() == 0)
		return;

	AbsNeuron<Type> *pCurNeuron;
	Edge<Type> 	*pCurEdge;
	Type 		val;

	// calc error deltas
	val = this->GetErrorDelta();
	for(unsigned int i = 0; i < this->GetConsO().size(); i++) {
		pCurEdge 	= this->GetConO(i);
		pCurNeuron 	= pCurEdge->GetDestination(this);
		val += pCurNeuron->GetErrorDelta() * pCurEdge->GetValue();
	}
	
	val *= Functor::derivate( this->GetValue(), 0.f );
	this->SetErrorDelta(val);

	// adapt weights
	for(unsigned int i = 0; i < this->GetConsO().size(); i++) {
		pCurEdge = this->GetConO(i);
		if(pCurEdge->GetAdaptationState() == true) {
			val = Functor::learn( 	this->GetValue(), 
						pCurEdge->GetValue(), 
						pCurEdge->GetMomentum(),
						pCurEdge->GetDestination(this)->GetErrorDelta(),
						m_Setup );
			
			pCurEdge->SetMomentum( val );
			pCurEdge->SetValue( val+pCurEdge->GetValue() );
		}
	}
}

