%{
#include "containers/Centroid.h"
%}

%include "containers/Centroid.h"


%include <std_sstream.i>
%extend ANN::Centroid {
	char *__str__() {
		std::ostringstream ostrs;
		char *c_str;

		ostrs << "Centroid: " << $self->m_iBMUID << std::endl;
		ostrs << "Distance: " << $self->m_fEucDist << std::endl;
		ostrs << "VCen.: ";
		ostrs << "[";
		for(unsigned int i = 0; i < $self->m_vCentroid.size(); i++) {
			float fVal = $self->m_vCentroid[i];
			ostrs << fVal;
			if(i < $self->m_vCentroid.size()-1) {
				ostrs << ", ";
			}
		}
		ostrs << "]\n";
		ostrs << "VInp.: ";
		ostrs << "[";
		for(unsigned int i = 0; i < $self->m_vInput.size(); i++) {
			float fVal = $self->m_vInput[i];
			ostrs << fVal;
			if(i < $self->m_vInput.size()-1) {
				ostrs << ", ";
			}
		}
		ostrs << "]";

		c_str = new char[ostrs.str().length()+1];
		strcpy(c_str, ostrs.str().c_str());
		return c_str;
	}
}