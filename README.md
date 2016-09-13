# Updates - 09/03/2016

- Improvements of the implementation of the base classes
- Added a Qt4 demo, illustrating how to implement a GUI on the example of a back-propagation network 
  - Note that the networks can be highly asymmetrical

![artw_i](https://cloud.githubusercontent.com/assets/4668178/18225607/b8ddff64-71f6-11e6-872f-78c3b0450717.png)
__GUI-example: Designer for back propagation networks.__ The layout of the underlying library is 1:1 represented as a QSceneGraph. After definition of the network topology, the in- and output can be defined by the user and the network trained accordingly. At the end, the error of each test-training cycle is plotted, which gives a handy representation of the network performance.


# Updates - 07/31/2016

- The project was rewritten into a template library
- The CUDA device function pointers are poorly implemented, thus the function extensions work with template parameters now
- As most of the code gets created just at compile time, the compile time increased in comparison to the old version
- All classes but SOMNetGPU are now header only
- A bug with SOMs on GPU was fixed, which led to wrong results of the radius decay function

Here is an example of the new device pointer replacement. The distance function of the network is now a template argument, which will simplify library functionality extensions.

```
int main(int argc, char *argv[]) {
	// ..
	ANNGPGPU::SOMNetGPU<float, ANN::functor_gaussian<float>> gpu;
```

# Updates - 04/12/2016

- The support for CUDA > version 4 was broken. Now it works again (tested with CUDA 7.5)

# Projects
## Artwork from Ben Bogart

[![Image decomposition](https://cloud.githubusercontent.com/assets/4668178/18225709/9fe765e2-71f9-11e6-8846-1c47de4ad4f2.png)](https://player.vimeo.com/video/181111922?autoplay=1&loop=1&title=0&byline=0&portrait=0)
__This video is from [ekran.org](http://www.ekran.org/ben/wp/2016/09/) and shows the decomposition of a high resolution panorama by a SOM on the GPU__

![artw_i](http://www.ekran.org/ben/wp/wp-content/uploads/2016/07/good_result_24mm-final-detail-11.jpg)
__Linewise growing neighborhood from [ekran.org](http://www.ekran.org/ben/wp/2016/03/)__

# Introduction

ANNet is a small library to create neural nets. A hallmark of the project are implementations of several neural network models with usage of OpenMP and/or Thrust. 
See quickstart guide to learn how to use them. 
Especially self organizing maps (SOMs) benefit strongly from calculations on GPUs and speed-ups by a factor of 100 can easily be achieved for bigger networks (>256x256 nodes). 
The GPU implementation of SOMs is also supporting asynchronous calculation on more than one device. 

# Features
- Implementation:
  - Self organizing maps using CUDA
  - Back propagation networks
- Python interface for all classes
- Multi core support using OpenMP
- Plugin system based on template parameters
- With the exception of the CUDA implementation, this project is a header only library

# Build

To build the library with all features you need:

- Qt5 (for some examples required)
- SWIG for python bindings (just required for python bindings)
- CUDA/Thrust (shipped with CUDA; just required for GPGPU implementation)
- Doxygen (required for documentation generation)
- OpenMP (required if multi CPU support is wished)
- Lib bzip2 (required)
- CMake (required if you want to use the CMake scripts)
- A C++ compiler (GCC or MinGW; required)
- How you build the library:

Clone the repository with git:

```
git clone https://github.com/ANNetGPGPU/ANNetGPGPU.git
```

Create a build directory, where your compiler stores the objects:

```
cd ANNetGPGPU
mkdir build
cd build
```

Run CMake and make to build. Dependent on the installed libraries, either all or just some example programs will be built:

```
cmake .. && make
```

# Usage (Python interface)

There is a python interface for the library, which may have access to the GPU too. 
The python demo here shows a k-means like clustering approach with the SOM implementation. 
If the number of nodes is reduced you can calculate the "centroids" of given input vectors.
In this example a three-dimensional input was chosen for simplicity.
Note: Currently, the template library could work with double precision. 
However, the python interface is currently implemented for float.

```
from ANPyNetCPU import *
black 	= vectorf([0,0,0])
white 	= vectorf([1,1,1])
red 	= vectorf([1,0,0])
green 	= vectorf([0,1,0])
blue 	= vectorf([0,0,1])

trainSet = TrainingSetF()
trainSet.AddInput(black)
trainSet.AddInput(white)
trainSet.AddInput(red)
trainSet.AddInput(green)
trainSet.AddInput(blue)

widthMap = 5
heightMap = 1

inpWidth = 3
inpHeight = 1

SOM = SOMNetGaussF(inpWidth,inpHeight,widthMap,heightMap)
SOM.SetTrainingSet(trainSet)
SOM.SetLearningRate(0.75)
SOM.Training(100)

# gets an ordered list of different centroids with the ID of the corresponding BMU
centroids = SOM.GetCentroidList()

# output for fun
for i in centroids:
	print (i)
```
  
The output of the demo would be:

```
White
> 1
> 1
> 1

Red
> 1
> 7.2e-23
> 7.2e-23

Black
> 9.6e-18
> 1.4e-45
> 3.0e-18

Blue
> 2.9e-22
> 1.4e-45
> 1
```

# Usage (C++)

I prepared working examples for many typical use cases. These examples can be found in the "/src/examples" folder.
In the following I write a bit about these examples, to help you understand the layout of the library.


## 1. Back propagation networks

Define the input, e.g. numbers from 0 to 9 could be declared.

```
float fInp1[56] = { 0,0,0,0,1,0,0,
		    0,0,0,1,1,0,0,
		    0,0,1,0,1,0,0,
		    0,1,0,0,1,0,0,
		    0,0,0,0,1,0,0,
		    0,0,0,0,1,0,0,
		    0,0,0,0,1,0,0,
		    0,0,0,0,1,0,0 };
					
float fInp2[56] = { 0,0,0,1,1,0,0,
		    0,0,1,0,0,1,0,
		    0,1,0,0,0,1,0,
		    0,1,0,0,1,0,0,
		    0,0,0,1,0,0,0,
		    0,0,0,1,0,0,0,
		    0,0,1,0,0,0,0,
		    0,1,1,1,1,1,0 };
					
// ..
```

AddLayer connects all neurons of both layers with each other.
It is also possible to create networks by defining each connection. 
This can be done by a vector describing the graph.
Internally the networks works as a linked list (cpu). 
For the gpu implementation, a vector is created based on the connection graph.

```
#include <ANNet>
#include <ANContainers>
#include <ANMath>

#include <Samples.h>

#include <ctime>
#include <iostream>


int main(int argc, char *argv[]) {
	ANN::BPNet<float, ANN::fcn_log<float>> cpu_one;
        
	ANN::BPLayer<float, ANN::fcn_log<float>> *layer1 = cpu_one.AddLayer(56, ANN::ANLayerInput);
	ANN::BPLayer<float, ANN::fcn_log<float>> *layer2 = cpu_one.AddLayer(64, ANN::ANLayerHidden);
	ANN::BPLayer<float, ANN::fcn_log<float>> *layer3 = cpu_one.AddLayer(9, ANN::ANLayerOutput);

	layer1->ConnectLayer(layer2);
	layer2->ConnectLayer(layer3);
	
	ANN::TrainingSet<float> input;
	input.AddInput(fInp1, 56);
	input.AddOutput(fOut1, 9);
	// .. more input/output samples
	
	std::vector<float> errors;
	
	ANN::HebbianConf<float> conf = {0.5, 0, 0};
	cpu_one.Setup(conf);
	cpu_one.SetTrainingSet(input);

	bool b = false;
	float f;
	errors = cpu_one.TrainFromData(50, 0, b, f);
	std::cout<< &cpu_one <<std::endl;

	cpu_one.ExpToFS("foo.bar");
	ANN::BPNet<float, ANN::fcn_log<float>> cpu_two;
	cpu_two.ImpFromFS("foo.bar");
	cpu_two.SetTrainingSet(input);
	
	std::cout<< &cpu_two <<std::endl;
	return 0;
}

```

HebbianConf is a small struct storing the learning rates and related constants of the network.
A zero will automatically disable the related procedure like the momentum term, or weight decay during training.

```
template <class T>
struct HebbianConf {
	T learning_rate;
	T momentum_rate;
	T weight_decay;
};
```

## 2. Self organizing maps (SOMs)

### 1. CPU implementation

Self-organizing maps (SOM) are a type of network that is trained using unsupervised learning to produce a low-dimensional (typically two-dimensional), 
representation of training samples. 
SOMs are different from other networks in the sense that they use a neighborhood function for learning. 
The example shows a SOM 128x128 node network, each node can process a 3x1 input vector.

```
ANN::SOMNet<float, ANN::functor_gaussian<float>> cpu;
cpu.CreateSOM(3, 1, 128,128);
cpu.SetTrainingSet(input);
cpu.Training(100, ANN::ANSerialMode);
```

### 2. GPGPU implementation

It is possible to speed up the calculation of SOMs by processing them on GPU. 
One can train, save and load the net with CPU as well and later continue with GPU and the other way round, 
simply by calling the CTOR:

```
ANN::SOMNet<float, ANN::functor_gaussian<float>> cpu;
cpu.CreateSOM(3, 1, 128, 128); // input w, h, net w, h
ANN::SOMNetGPU<float, ANN::functor_gaussian<float>> gpu(&cpu);      // use copy CTOR or create it like above

// do stuff
// ..

cpu.ExpToFS(foo.bar);
gpu.ImpFromFS(foo.bar);
```

### 3. GPGPU <-> CPU comparison 
![Figure 1](https://cloud.githubusercontent.com/assets/4668178/8379809/abfa53c2-1c22-11e5-8574-ebe873044d6d.png)

__Figure 1: Training progress on quad core CPU (A) and GTX 570 (B) with three dimensional vectors as input.__
Both maps have the same size and the computation time was fixed. 
In the same time the GPU implementation was able to process more iterations, 
thus the classification of the input vectors resulted in a lower error.

# General class extensions

You certainly know that in C++ it is not possible to add methods to an existing class. So how that's possible? 
The trick is to include in the declaration of a class a file defined by the preprocessor token e.g. "\_\_ConTable_ADDON".
You can extent any class and most files like this. The token always have a "\_\_" prefix followed by the class name and the "\_ADDON" suffix.
Classes/Structs have such injection points always in the "public" space. 

```
template <class T>
struct ConTable {
	// ..
	std::vector<T> m_vMisc;
	
#ifdef __ConTable_ADDON
	#include __ConTable_ADDON
#endif
};
```

For this we just need to create a new header file, e.g. "foo.h" with the example content:

```
std::vector<T> m_vYourVector; // vector for your addon
void your_new_function() { }  // your custom function ..
```

Then we just need to pass the path of this file to the build system.
In CMake, this would look like this.

```
add_definitions(-D__ConTable_ADDON="${SOM_GPU_ADDON_SOURCE_DIR}/foo.h")
```

Here, "SOM_GPU_ADDON_SOURCE_DIR" is the path of the file "foo.h"

# Adding custom learning/distance functions

## Adding new functions for the CPU based library

As long as you work with the CPU implementation it is super simple to define your own functions and pass them to the related functor. 


```
/*
 * Here we declare just the basic learning functions 
 */
template <class T>
inline T custom_learn(T fWeight, T fInfluence, T fInput) {
	return fWeight + (fInfluence*(fInput-fWeight) );
}

template <class T>
inline T custom_gaussian_nhood (T dist, T sigmaT) {
	return exp(-pow(dist, 2.f)/(2.f*pow(sigmaT, 2.f)));
}

template <class T>
inline T custom_rad_decay (T sigma0, T t, T lambda) {
	return std::floor(sigma0*exp(-t/lambda) + 0.5f);
}

template <class T>
inline T custom_lrate_decay (T sigma0, T t, T lambda) {
	return sigma0*exp(-t/lambda);
}

/*
 * Here we define the functor for the network 
 */
template<class T> using custom_functor = ANN::DistFunction<T, custom_learn<T>, custom_gaussian_nhood<T>, custom_rad_decay<T>, custom_lrate_decay<T> >;

int main(int argc, char *argv[]) {
	// ..
	ANN::SOMNetCPU<float, custom_functor<float> > cpu;
	// ..
```

## Adding new functions for the GPU based library

Adding new functions/functors for the GPU implementation is not as simple. The NVCC requires instantiation of the class "SOMNetGPU", 
because not all of the class implementation can be shifted into a regular header file (device code). 
This means we need to create an instance of our "extended" class right when the library is build.

For this we can make use of the __general class extensions__. To achieve the same as illustrated in the previous example. 
We define a file "NewFunctions.h":

```
template <class T>
inline T __host__ __device__ custom_learn(T fWeight, T fInfluence, T fInput) {
	return fWeight + (fInfluence*(fInput-fWeight) );
}

template <class T>
inline T __host__ __device__ custom_gaussian_nhood (T dist, T sigmaT) {
	return exp(-pow(dist, 2.f)/(2.f*pow(sigmaT, 2.f)));
}

template <class T>
inline T __host__ __device__ custom_rad_decay (T sigma0, T t, T lambda) {
	return std::floor(sigma0*exp(-t/lambda) + 0.5f);
}

template <class T>
inline T __host__ __device__ custom_lrate_decay (T sigma0, T t, T lambda) {
	return sigma0*exp(-t/lambda);
}

template<class T> using custom_functor = ANN::DistFunction<T, custom_learn<T>, custom_gaussian_nhood<T>, custom_rad_decay<T>, custom_lrate_decay<T> >;
```
.. and a file "NewInstances.h":

```
template ANNGPGPU::SOMNetGPU<float, custom_functor<float> >::SOMNetGPU();
template ANNGPGPU::SOMNetGPU<double, custom_functor<double> >::SOMNetGPU();
// ..
```
then we adapt the CMake build file and add the following lines:

```
# here we pass the extension headers to the build system
add_definitions(-D__Functions_ADDONS="${SOM_GPU_ADDON_SOURCE_DIR}/NewFunctions.h")
add_definitions(-D__SOMNetGPU_INSTANCES="${SOM_GPU_ADDON_SOURCE_DIR}/NewInstances.h")
```

# Advanced implementation guide

Usually a network consists of nodes and edges. Most implementations of neural networks store their data in a array. 
Calculation is simple with this approach, but asymmetric networks get hard to implement and new functionality is hard to add.
To create more complex variants of networks it makes sense to put the information of the network in the edges and the functionality to calculate parts of the network into the nodes. 
This makes it easier to include new functionality and to re-use old code.
Because information flow in neuronal networks is often directed, container classes help to keep an order. 
The network class on the other hand calls learning or training functions and implements the principal learning procedure, 
e.g. switching training patterns or break the learning procedure if a certain error limit was hit.

To make the long story short, the three important classes to derive from are:

* AbsNeuron
* AbsLayer
* AbsNet

## AbsNeuron
In the library some functions of the abstract base classes are meant to get implemented in the derived class. 
Here I give a small overview about them.

```
virtual void AdaptEdges()       = 0;
virtual void CalcValue()        = 0;
```

In CalcValue() you calculate the data you want to store in the neuron. 
Every neuron (or node) in the network has a list of edges which direct to neurons of another (or the same) layer. 
This example shows you how to run through this list to implement a neuron in a back propagation network.

```
template <class Type, class Functor>
void BPNeuron<Type, Functor>::CalcValue() {
	if(this->GetConsI().size() == 0) {
		return;
	}

	Type val = 0;
	for(unsigned int i = 0; i < this->GetConsI().size(); i++) {
		AbsNeuron<Type> *from = this->GetConI(i)->GetDestination(this);
		val += from->GetValue() * this->GetConI(i)->GetValue();
	}
	this->SetValue(val);

	val = Functor::transfer( this->GetValue(), 0.f );
	this->SetValue(val);
}
```
 
The algorithm to adapt the edges is implemented in AdaptEdges(). 
Again we use the internal list to run through all edges (outgoing ones) the neuron is connected with.

```
template <class Type, class Functor>
void BPNeuron<Type, Functor>::AdaptEdges() {
	if(this->GetConsO().size() == 0)
		return;

	AbsNeuron<Type> *pCurNeuron;
	Edge<Type> 	*pCurEdge;
	Type 		val;

	// calc error deltas
	val = this->GetErrorDelta();
	for(unsigned int i = 0; i < this->GetConsO().size(); i++) {
		pCurEdge 	= this->GetConO(i);
		pCurNeuron 	= pCurEdge->GetDestination(this);
		val += pCurNeuron->GetErrorDelta() * pCurEdge->GetValue();
	}
	
	val *= Functor::derivate( this->GetValue(), 0.f );
	this->SetErrorDelta(val);

	// adapt weights
	for(unsigned int i = 0; i < this->GetConsO().size(); i++) {
		pCurEdge = this->GetConO(i);
		if(pCurEdge->GetAdaptationState() == true) {
			val = Functor::learn( 	this->GetValue(), 
						pCurEdge->GetValue(), 
						pCurEdge->GetMomentum(),
						pCurEdge->GetDestination(this)->GetErrorDelta(),
						m_Setup );
			
			pCurEdge->SetMomentum( val );
			pCurEdge->SetValue( val+pCurEdge->GetValue() );
		}
	}
}
```

## AbsLayer

Neurons are stored in layers. If you decide to write your own layer class, then you have to implement the Resize() function. 
This could be useful especially if you have strange layer topologies (e.g. 2-dimensional or 3-dimensional).

```
template <class Type, class Functor>
void BPLayer<Type, Functor>::Resize(const unsigned int &iSize) {
	this->EraseAll();
	this->AddNeurons(iSize);
}
```

## AbsNet

The last class you may want to derive from, is AbsNet. 
Here are three functions you have to overload: PropagateFW() and PropagateBW(). 

```
virtual void PropagateFW() = 0;
virtual void PropagateBW() = 0;
```

Here I show, how to implement these functions in a back propagation network.

```
template <class Type, class Functor>
void BPNet<Type, Functor>::PropagateFW() {
	for(unsigned int i = 1; i < this->m_lLayers.size(); i++) {
		BPLayer<Type, Functor> *curLayer = ((BPLayer<Type, Functor>*)this->GetLayer(i) );
		//#pragma omp parallel for
		for(int j = 0; j < static_cast<int>(curLayer->GetNeurons().size() ); j++) {
			curLayer->GetNeuron(j)->CalcValue();
		}
	}
}

template <class Type, class Functor>
void BPNet<Type, Functor>::PropagateBW() {
	for(int i = this->m_lLayers.size()-1; i >= 0; i--) {
		BPLayer<Type, Functor> *curLayer = ( (BPLayer<Type, Functor>*)this->GetLayer(i) );
		//#pragma omp parallel for
		for(int j = 0; j < static_cast<int>( curLayer->GetNeurons().size() ); j++) {
			curLayer->GetNeuron(j)->AdaptEdges();
		}
	}
}
```

Different Implementations use different types of layers.
This is why you may want to overload AddLayer().

```
template <class Type, class Functor>
void BPNet<Type, Functor>::AddLayer(const unsigned int &iSize, const LayerTypeFlag &flType) {
	this->AddLayer( new BPLayer<Type, Functor>(iSize, flType, -1) );
}
```
# Save and load your custom class data to file-system

## Example: Back propagation networks

If you decide to create your own e.g. Layer class, then you probably want to add features which require to be stored on the hdd. 
For this the ExpToFS() and ImpFromFS() functions are required to get modified. This works more or less like reimplementing virtual functions in Qt.
Calling the virtual base class ensures to save the base class contents. 
The following example shows how the freshly inserted integer storing the layer depth in a back propagation network can be saved and recovered from the hdd. 

```
template <class Type, class Functor>
void BPLayer<Type, Functor>::ExpToFS(BZFILE* bz2out, int iBZ2Error) {
	std::cout<<"Save BPLayer to FS()"<<std::endl;
	AbsLayer<Type>::ExpToFS(bz2out, iBZ2Error);

	int iZLayer = m_iZLayer;
	BZ2_bzWrite( &iBZ2Error, bz2out, &iZLayer, sizeof(int) );
}
```

Now the other way round, we load the content from the file-system.

```
template <class Type, class Functor>
int BPLayer<Type, Functor>::ImpFromFS(BZFILE* bz2in, int iBZ2Error, ConTable<Type> &Table) {
	std::cout<<"Load BPLayer from FS()"<<std::endl;
	int iLayerID = AbsLayer<Type>::ImpFromFS(bz2in, iBZ2Error, Table);

	int iZLayer = -1;
	BZ2_bzRead( &iBZ2Error, bz2in, &iZLayer, sizeof(int) );
	Table.ZValOfLayer.push_back(iZLayer);

	return iLayerID;
}
```

The last function which has to be implemented is CreateNet(). 
Here the content loaded from the file system is used to create a copy of the net in the memory. 
The base implementation creates the layers and the connections of the network, so we just have to implement the bias neuron.

```
template <class Type, class Functor>
void BPNet<Type, Functor>::CreateNet(const ConTable<Type> &Net) {
	std::cout<<"Create BPNet"<<std::endl;

	/*
	* Init
	*/
	unsigned int iDstNeurID = 0;
	unsigned int iDstLayerID = 0;
	unsigned int iSrcLayerID = 0;

	Type fEdgeValue = 0.f;

	AbsLayer<Type> *pDstLayer = NULL;
	AbsLayer<Type> *pSrcLayer = NULL;
	AbsNeuron<Type> *pDstNeur = NULL;
	AbsNeuron<Type> *pSrcNeur = NULL;

	/*
	* For all nets necessary: Create Connections (Edges)
	*/
	AbsNet<Type>::CreateNet(Net);

	/*
	* Support z-layers
	*/
	for(unsigned int i = 0; i < this->m_lLayers.size(); i++) {
		BPLayer<Type, Functor> *curLayer = ((BPLayer<Type, Functor>*)this->GetLayer(i) );
		curLayer->SetZLayer(Net.ZValOfLayer[i]);
	}
}
```

## Example: Self organizing maps

Here is another example for SOMs. Only the import of the positions has to be added.

```
template<class Type, class Functor>
void SOMNet<Type, Functor>::CreateNet(const ConTable<Type> &Net) {
	std::cout<<"Create SOMNet"<<std::endl;

	/*
	* For all nets necessary: Create Connections (Edges)
	*/
	AbsNet<Type>::CreateNet(Net);

	/*
	* Set Positions
	*/
	for(unsigned int i = 0; i < Net.Neurons.size(); i++) {
		int iLayerID 	= Net.Neurons.at(i).m_iLayerID;
		int iNeurID 	= Net.Neurons.at(i).m_iNeurID;
		
		// Get position
		int iPosSize = Net.Neurons.at(i).m_vMisc.at(0);
		std::vector<Type> vPos(iPosSize);
		for(int j = 0; j < iPosSize; j++) {
			vPos[j] = Net.Neurons.at(i).m_vMisc[1+j];
		}
		
		// Save other information of the neuron
		ANN::SOMNeuron<Type> *pNeuron = (ANN::SOMNeuron<Type> *)this->GetLayer(iLayerID)->GetNeuron(iNeurID);
		pNeuron->SetPosition(vPos);
		pNeuron->SetLearningRate(Net.Neurons.at(i).m_vMisc[iPosSize+1]);
		pNeuron->SetSigma0(Net.Neurons.at(i).m_vMisc[iPosSize+2]);
	}
}
```
