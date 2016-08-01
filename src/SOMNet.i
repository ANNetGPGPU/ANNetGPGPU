%{
#include "SOMNet.h"
%}

%include "SOMNet.h" 

namespace ANN {
	%template(SOMNetGaussF) SOMNet<float, DistFunction<float, som_learn<float>, fcn_gaussian_nhood<float>, fcn_rad_decay<float>, fcn_lrate_decay<float> > >;
	%template(SOMNetBubbleF) SOMNet<float, DistFunction<float, som_learn<float>, fcn_bubble_nhood<float>, fcn_rad_decay<float>, fcn_lrate_decay<float> > >;
	%template(SOMNetCGaussF) SOMNet<float, DistFunction<float, som_learn<float>, fcn_cutgaussian_nhood<float>, fcn_rad_decay<float>, fcn_lrate_decay<float> > >;
	%template(SOMNetEpanF) SOMNet<float, DistFunction<float, som_learn<float>, fcn_epanechicov_nhood<float>, fcn_rad_decay<float>, fcn_lrate_decay<float> > >;
	%template(SOMNetMexicanF) SOMNet<float, DistFunction<float, som_learn<float>, fcn_mexican_nhood<float>, fcn_rad_decay<float>, fcn_lrate_decay<float> > >;
}
