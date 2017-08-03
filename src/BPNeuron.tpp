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

template <class Type, class Functor>
BPNeuron<Type, Functor>::BPNeuron(AbsLayer<Type> *parentLayer) : AbsNeuron<Type>(parentLayer) {
	m_Setup = {0.1f, 0, 0};
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

