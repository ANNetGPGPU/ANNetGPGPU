%{
#include "containers/TrainingSet.h"
%}

%include "containers/TrainingSet.h"  

namespace ANN {
	%template(TrainingSetF) TrainingSet<float>;
}