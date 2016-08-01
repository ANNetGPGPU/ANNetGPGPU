%{
#include "AbsLayer.h"
%}

%ignore ANN::SetEdgesToValue;

%include "AbsLayer.h" 

namespace ANN {
	%template(AbsLayerF) AbsLayer<float>;
}