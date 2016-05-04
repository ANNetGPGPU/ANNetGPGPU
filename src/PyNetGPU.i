%define DOCSTRING
"ANNet is a small library to create neural networks. 
At the moment there are implementations of several neural network models with usage of OpenMP and/or Thrust.
Author: Daniel Frenzel"
%enddef
%module(docstring=DOCSTRING) ANPyNetGPU

%include Ignore.i

%include base/Edge.i
%include base/AbsNeuron.i
%include base/AbsLayer.i
%include base/AbsNet.i

%include containers/Centroid.i
%include containers/2DArray.i
%include containers/3DArray.i
%include containers/TrainingSet.i
%include containers/ConTable.i

%include HFNeuron.i
%include HFLayer.i
%include HFNet.i

%include SOMNeuron.i
%include SOMLayer.i
%include SOMNet.i

%include BPNeuron.i
%include BPLayer.i
%include BPNet.i

%include math/Functions.i
//%include math/Random.i

//%include gpgpu/Kernels.i
//%include gpgpu/Matrix.i
%include gpgpu/BPNetGPU.i
%include gpgpu/SOMNetGPU.i

%include StdOut.i