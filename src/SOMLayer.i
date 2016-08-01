%{
#include "SOMLayer.h"
%}

%include "SOMLayer.h"  
%include "std_sstream.i"

namespace ANN {
	%extend SOMLayer<T> {
		char *__str__() {
			std::ostringstream ostrs;
			char *c_str;
			
			for(unsigned int i = 0; i < $self->GetNeurons().size(); i++) {
				float fVal = static_cast<float>($self->GetNeuron(i)->GetValue());
				ostrs << fVal << std::endl;
			}

			c_str = new char[ostrs.str().length()+1];
			strcpy(c_str, ostrs.str().c_str());
			return c_str;
		}
	}
}

namespace ANN {
	%template(SOMLayerF) SOMLayer<float>;
}