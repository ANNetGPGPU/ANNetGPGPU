%{
#include "Edge.h"
%}

%include "Edge.h"
%include "std_sstream.i"

namespace ANN {
	%extend Edge<float> {
		char *__str__() {
			std::ostringstream ostrs;
			char *c_str;
			
			float fVal = static_cast<float>($self->GetValue());
			ostrs << fVal << std::endl;

			c_str = new char[ostrs.str().length()+1];
			strcpy(c_str, ostrs.str().c_str());
			return c_str;
		}
	}
}

namespace ANN {
	%template(EdgeF) Edge<float>;
}
