%{
#include "HFNeuron.h"
%}

%include "HFNeuron.h"
%include "std_sstream.i"

namespace ANN {
	%extend HFNeuron {
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