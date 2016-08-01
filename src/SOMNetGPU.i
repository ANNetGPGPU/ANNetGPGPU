%{
#include "SOMNetGPU.h"
%}

%include "SOMNetGPU.h" 

namespace ANN {
	%template(SOMNetGPUGaussF) SOMNet<float, DistFunction<float, som_learn<float>, fcn_gaussian_nhood<float>, fcn_rad_decay<float>, fcn_lrate_decay<float> > >;
	%template(SOMNetGPUBubbleF) SOMNet<float, DistFunction<float, som_learn<float>, fcn_bubble_nhood<float>, fcn_rad_decay<float>, fcn_lrate_decay<float> > >;
	%template(SOMNetGPUCGaussF) SOMNet<float, DistFunction<float, som_learn<float>, fcn_cutgaussian_nhood<float>, fcn_rad_decay<float>, fcn_lrate_decay<float> > >;
	%template(SOMNetGPUEpanF) SOMNet<float, DistFunction<float, som_learn<float>, fcn_epanechicov_nhood<float>, fcn_rad_decay<float>, fcn_lrate_decay<float> > >;
	%template(SOMNetGPUMexicanF) SOMNet<float, DistFunction<float, som_learn<float>, fcn_mexican_nhood<float>, fcn_rad_decay<float>, fcn_lrate_decay<float> > >;
}
