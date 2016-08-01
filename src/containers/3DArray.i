%{
#include "containers/3DArray.h"
%}

%ignore ANN::F3DArray::operator [];

%include "containers/3DArray.h"  

%extend ANN::F3DArray<float> {
	ANN::F2DArray<float> __getitem__(int z) {
		return self->GetSubArrayXY(z);
	}
};
