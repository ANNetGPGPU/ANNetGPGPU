%{
#include "SOMNetGPU.h"
%}

%ignore ANNGPGPU::SOMNetGPU::SetDistFunction(const ANN::DistFunction *);

%include "SOMNetGPU.h" 
