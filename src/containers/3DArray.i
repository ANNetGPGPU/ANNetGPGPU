%{
#include "containers/3DArray.h"
%}

%ignore ANN::F3DArray::operator [];

%include "containers/3DArray.h"  

%extend ANN::F3DArray {
	ANN::F2DArray __getitem__(int z) {
		return self->GetSubArrayXY(z);
	}
};