%{
#include "base/AbsNet.h"
%}

%ignore ANN::AbsNet::SetTrainingSet(const TrainingSet *);
%ignore ANN::AbsNet::SetTransfFunction(const TransfFunction *);

%include "base/AbsNet.h"  
