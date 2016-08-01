%{
#include "AbsNet.h"
%}

%ignore ANN::AbsNet::SetTrainingSet(const TrainingSet *);
%ignore ANN::AbsNet::SetTransfFunction(const TransfFunction *);

%include "AbsNet.h"  

namespace ANN {
	%template(AbsNetF) AbsNet<float>;
}