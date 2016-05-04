%{
#include "SOMNet.h"
%}

%ignore ANN::SOMNet::SetDistFunction(const DistFunction *);

%include "SOMNet.h" 
