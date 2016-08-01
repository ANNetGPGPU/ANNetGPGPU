%{
#include "AbsNeuron.h"
%}

%ignore ANN::Connect(AbsNeuron *, AbsNeuron *, const bool &);
%ignore ANN::Connect(AbsNeuron *, AbsLayer  *, const bool &);
%ignore ANN::Connect(AbsNeuron *, AbsNeuron *, const float &, const float &, const bool &);
%ignore ANN::Connect(AbsNeuron *, AbsLayer *, const std::vector<float> &, const std::vector<float> &, const bool &);

%include "AbsNeuron.h"

namespace ANN {
	%template(AbsNeuronF) AbsNeuron<float>;
}