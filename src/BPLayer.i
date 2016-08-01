%{
#include "BPLayer.h"
%}

%include "BPLayer.h"
%include "std_sstream.i"

namespace ANN {
	%extend BPLayer<float, TransfFunction<float, hebbian_learn<float>, fcn_log_normal<float>, fcn_log_derivate<float> > > {
		char *__str__() {
			std::ostringstream ostrs;
			char *c_str;
			
			for(unsigned int i = 0; i < $self->GetNeurons().size(); i++) {
				float fVal = $self->GetNeuron(i)->GetValue();
				ostrs << fVal << std::endl;
			}

			c_str = new char[ostrs.str().length()+1];
			strcpy(c_str, ostrs.str().c_str());
			return c_str;
		}
	}
	
	%extend BPLayer<float, TransfFunction<float, hebbian_learn<float>, fcn_tanh_normal<float>, fcn_tanh_derivate<float> > > {
		char *__str__() {
			std::ostringstream ostrs;
			char *c_str;
			
			for(unsigned int i = 0; i < $self->GetNeurons().size(); i++) {
				float fVal = $self->GetNeuron(i)->GetValue();
				ostrs << fVal << std::endl;
			}

			c_str = new char[ostrs.str().length()+1];
			strcpy(c_str, ostrs.str().c_str());
			return c_str;
		}
	}
	
	%extend BPLayer<float, TransfFunction<float, hebbian_learn<float>, fcn_linear_normal<float>, fcn_linear_derivate<float> > > {
		char *__str__() {
			std::ostringstream ostrs;
			char *c_str;
			
			for(unsigned int i = 0; i < $self->GetNeurons().size(); i++) {
				float fVal = $self->GetNeuron(i)->GetValue();
				ostrs << fVal << std::endl;
			}

			c_str = new char[ostrs.str().length()+1];
			strcpy(c_str, ostrs.str().c_str());
			return c_str;
		}
	}
	
	%extend BPLayer<float, TransfFunction<float, hebbian_learn<float>, fcn_binary_normal<float>, fcn_binary_derivate<float> > > {
		char *__str__() {
			std::ostringstream ostrs;
			char *c_str;
			
			for(unsigned int i = 0; i < $self->GetNeurons().size(); i++) {
				float fVal = $self->GetNeuron(i)->GetValue();
				ostrs << fVal << std::endl;
			}

			c_str = new char[ostrs.str().length()+1];
			strcpy(c_str, ostrs.str().c_str());
			return c_str;
		}
	}
}

namespace ANN {
	%template(BPLayerLogF) BPLayer<float, TransfFunction<float, hebbian_learn<float>, fcn_log_normal<float>, fcn_log_derivate<float> > >;
	%template(BPLayerTanF) BPLayer<float, TransfFunction<float, hebbian_learn<float>, fcn_tanh_normal<float>, fcn_tanh_derivate<float> > >;
	%template(BPLayerLinF) BPLayer<float, TransfFunction<float, hebbian_learn<float>, fcn_linear_normal<float>, fcn_linear_derivate<float> > >;
	%template(BPLayerBinF) BPLayer<float, TransfFunction<float, hebbian_learn<float>, fcn_binary_normal<float>, fcn_binary_derivate<float> > >;
}
