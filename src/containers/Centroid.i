%{
#include "containers/Centroid.h"
%}

%include "containers/Centroid.h"

%include <std_sstream.i>

namespace ANN {
	%template(CentroidF) Centroid<float>;
}

%extend ANN::Centroid<float> {
	char *__str__() {
		std::ostringstream ostrs;
		char *c_str;

		ostrs << "Centroid" << std::endl;
		ostrs << "Distance: " << $self->_distance << std::endl;
		ostrs << "VCen.: ";
		ostrs << "[";
		for(unsigned int i = 0; i < $self->_edges.size(); i++) {
			float fVal = static_cast<float>($self->_edges[i]);
			ostrs << fVal;
			if(i < $self->_edges.size()-1) {
				ostrs << ", ";
			}
		}
		ostrs << "]\n";
		ostrs << "VInp.: ";
		ostrs << "[";
		for(unsigned int i = 0; i < $self->_input.size(); i++) {
			float fVal = $self->_input[i];
			ostrs << fVal;
			if(i < $self->_input.size()-1) {
				ostrs << ", ";
			}
		}
		ostrs << "]";

		c_str = new char[ostrs.str().length()+1];
		strcpy(c_str, ostrs.str().c_str());
		return c_str;
	}

	float __getitem__(int i) {
		return self->_edges[i];
	}
}
