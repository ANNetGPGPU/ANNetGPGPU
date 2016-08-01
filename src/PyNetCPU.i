%define DOCSTRING
"ANNet is a small library to create neural networks. 
At the moment there are implementations of several neural network models with usage of OpenMP and/or Thrust.
Author: Daniel Frenzel"
%enddef
%module(docstring=DOCSTRING) ANPyNetCPU

%include Ignore.i

%include math/Functions.i

%include containers/Centroid.i
%include containers/2DArray.i
%include containers/3DArray.i
%include containers/TrainingSet.i
%include containers/ConTable.i

%include Edge.i
%include AbsNeuron.i
%include AbsLayer.i
%include AbsNet.i

%include SOMNeuron.i
%include SOMLayer.i
%include SOMNet.i

%include BPNeuron.i
%include BPLayer.i
%include BPNet.i

%include StdOut.i
