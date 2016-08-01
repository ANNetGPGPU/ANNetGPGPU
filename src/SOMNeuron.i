%{
#include "SOMNeuron.h"
%}

%ignore ANN::GetDistance2Neur;

%include "SOMNeuron.h"  
%include "std_sstream.i"

namespace ANN {
	%extend SOMNeuron<float> {
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
	%template(SOMNeuronF) SOMNeuron<float>;
}
