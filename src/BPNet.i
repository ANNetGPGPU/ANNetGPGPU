%{
#include "BPNet.h"
%}

%include "BPNet.h" 

namespace ANN {
	%template(BPNetLogF) BPNet<float, TransfFunction<float, hebbian_learn<float>, fcn_log_normal<float>, fcn_log_derivate<float> > >;
	%template(BPNetTanF) BPNet<float, TransfFunction<float, hebbian_learn<float>, fcn_tanh_normal<float>, fcn_tanh_derivate<float> > >;
	%template(BPNetLinF) BPNet<float, TransfFunction<float, hebbian_learn<float>, fcn_linear_normal<float>, fcn_linear_derivate<float> > >;
	%template(BPNetBinF) BPNet<float, TransfFunction<float, hebbian_learn<float>, fcn_binary_normal<float>, fcn_binary_derivate<float> > >;
}