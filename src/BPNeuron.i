%{
#include "BPNeuron.h"
%}

%include "BPNeuron.h" 
%include "std_sstream.i"

namespace ANN {
	%extend BPNeuron<float, TransfFunction<float, hebbian_learn<float>, fcn_log_normal<float>, fcn_log_derivate<float> > > {
		char *__str__() {
			std::ostringstream ostrs;
			char *c_str;
			
			float fVal = $self->GetValue();
			ostrs << fVal << std::endl;

			c_str = new char[ostrs.str().length()+1];
			strcpy(c_str, ostrs.str().c_str());
			return c_str;
		}
	}
	
	%extend BPNeuron<float, TransfFunction<float, hebbian_learn<float>, fcn_tanh_normal<float>, fcn_tanh_derivate<float> > > {
		char *__str__() {
			std::ostringstream ostrs;
			char *c_str;
			
			float fVal = $self->GetValue();
			ostrs << fVal << std::endl;

			c_str = new char[ostrs.str().length()+1];
			strcpy(c_str, ostrs.str().c_str());
			return c_str;
		}
	}
	
	%extend BPNeuron<float, TransfFunction<float, hebbian_learn<float>, fcn_linear_normal<float>, fcn_linear_derivate<float> > > {
		char *__str__() {
			std::ostringstream ostrs;
			char *c_str;
			
			float fVal = $self->GetValue();
			ostrs << fVal << std::endl;

			c_str = new char[ostrs.str().length()+1];
			strcpy(c_str, ostrs.str().c_str());
			return c_str;
		}
	}
	
	%extend BPNeuron<float, TransfFunction<float, hebbian_learn<float>, fcn_binary_normal<float>, fcn_binary_derivate<float> > > {
		char *__str__() {
			std::ostringstream ostrs;
			char *c_str;
			
			float fVal = $self->GetValue();
			ostrs << fVal << std::endl;

			c_str = new char[ostrs.str().length()+1];
			strcpy(c_str, ostrs.str().c_str());
			return c_str;
		}
	}
}

namespace ANN {	
	%template(BPNeuronLogF) BPNeuron<float, TransfFunction<float, hebbian_learn<float>, fcn_log_normal<float>, fcn_log_derivate<float> > >;
	%template(BPNeuronTanF) BPNeuron<float, TransfFunction<float, hebbian_learn<float>, fcn_tanh_normal<float>, fcn_tanh_derivate<float> > >;
	%template(BPNeuronLinF) BPNeuron<float, TransfFunction<float, hebbian_learn<float>, fcn_linear_normal<float>, fcn_linear_derivate<float> > >;
	%template(BPNeuronBinF) BPNeuron<float, TransfFunction<float, hebbian_learn<float>, fcn_binary_normal<float>, fcn_binary_derivate<float> > >;
}
