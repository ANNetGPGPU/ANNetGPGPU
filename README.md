# Updates - 07/31/2016

- The project was more or less rewritten into a template library
- The CUDA device function pointers are poorly implemented by CUDA, thus the function extensions work with template parameters now
- As most of the code gets created just at compile time, the compile time increased in comparison to the old version
- Most classes are now header only
- A bug with SOMs on GPU was fixed, which led to wrong results of the radius decay function

Here is the new device pointer replacement. The distance function is now a template argument. Guess, this will make things easier in future.

```
int main(int argc, char *argv[]) {
	// ..
	ANNGPGPU::SOMNetGPU<float, ANN::functor_gaussian<float>> gpu;
```

# Updates - 04/12/2016

- The support for CUDA > version 4 was broken. Now it works again (tested with CUDA 7.5)

# Projects
## Artwork from Ben Bogart

![artw_i](http://www.ekran.org/ben/wp/wp-content/uploads/2016/05/SOMResults_noSegmentation_SOMScale5_h1352_ns200_i1000000-scaler8.jpg)
__This image is from [ekran.org](http://www.ekran.org/ben/wp/2016/results-without-segmentation/#more-3490) and shows a panorama after several training cycles of a SOM__

![artw_i](http://www.ekran.org/ben/wp/wp-content/uploads/2016/03/still-proxy-pano-edit-montage-5_500-1_0.5-SOM-100000.jpg)
__Arranged composition of SOMs from [ekran.org](http://www.ekran.org/ben/wp/2016/03/)__

# Introduction

ANNet is a small library to create neural nets. A hallmark of the project are implementations of several neural network models with usage of OpenMP and/or Thrust. 
See quickstart guide to learn how to use them. 
Especially self organizing maps (SOMs) benefit strongly from calculations on GPUs and speed-ups by a factor of 100 can easily be achieved for bigger networks (>256x256 nodes). 
The GPU implementation of SOMs is also supporting asynchronous calculation on more than one device. So computation on clusters is possible.

# Features
- Implementation:
  - Self organizing maps
    * Very fast SOM implementation with CUDA with nearly 100 % scalability
    * Suitable for cluster analysis
  - Back propagation networks
    * Additional, initial implementation using CUDA
  - Hopfield networks
- Full Python interface for all classes
- Complete multi core support
- Plugin system: 
  - Transfer-, distance- and learning-functions can be replaced, extended, ..
  - Plugin system works also for the CUDA implementation for hardware supporting device function pointers (see examples in the repository)
- Minimalistic source code with focus on readability

# Build

To build the library with all features you need:

- Qt4 (just for some examples and designer required)
- SWIG for python bindings (just required for python bindings)
- CUDA/Thrust (shipped with CUDA; just required for GPGPU implementation)
- Doxygen (required for documentation generation)
- OpenMP (required if multi CPU support is wished)
- Lib bzip2 (required)
- CMake (required if you want to use the cmake scripts)
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

Run cmake and make to build. Dependent on the installed libraries, either all or just some example programs will be built:

```
cmake .. && make
```

# Usage (Python interface)

There is a python interface for the library, which even has access to the GPU. 
The python demo here shows a k-means like clustering approach with the SOM implementation. 
If the number of nodes is reduced you can calculate the "centroids" of given input vectors.
In this example a three-dimensional input was chosen for simplicity.

```
# .. input definition ..
..
# .. end input definition ..
trainSet = TrainingSet()
trainSet.AddInput(red)
trainSet.AddInput(black)
trainSet.AddInput(white)
trainSet.AddInput(blue)

widthMap = 4
heightMap = 1
inpWidth = 3
inpHeight = 1

SOM = SOMNet(inpWidth,inpHeight,widthMap,heightMap)
SOM.SetTrainingSet(trainSet)
SOM.Training(1000)

centroids = SOM.GetCentroidList()

for i in centroids:
  print i
```
  
The output of the demo would be:

```
White
> 1
> 1
> 1

Red
> 1
> 7.24969e-023
> 7.24969e-023

Black
> 9.65357e-018
> 1.4013e-045
> 3.06738e-018

Blue
> 2.91553e-022
> 1.4013e-045
> 1
```

# Usage (C++)

## 1. Classical back propagation networks

It is a type of classifier that performs its predictions based on a linear or non-linear transfer function, 
The example below shows how the creation of a simple networks can be done with a few lines.

```
ANN::BPLayer layer1(64*64, ANN::ANLayerInput); // input layer of size: 64X64
ANN::BPLayer layer2(16, ANN::ANLayerOutput);   // output layer of size: 16

layer1.ConnectLayer(&layer2);

ANN::BPNet cpu;
cpu.AddLayer(&layer1);
cpu.AddLayer(&layer2);
cpu.SetNetFunction(&Functions::fcn_linear);
```

AddLayer connects all neurons of both layers with each other.  
It is also possible to create asymmetric networks by defining the connections.
This can be done by a vector describing the graph or by hand, as related functions exist in the library. 
Member functions for this exist already.
Internally the networks works as a linked list (cpu). For the gpu a vector is created based on the connection graph. 

## 2. Multilayer (back propagation) networks

Adding additional layers can be achieved like this.
A sigmoid transfer function is used as standard if SetNetFunction() is not called.

```
ANN::BPLayer layer1(64*64, ANN::ANLayerInput); // input layer of size: 64X64
ANN::BPLayer layer2(64, ANN::ANLayerHidden);   // hidden layer of size: 64
ANN::BPLayer layer3(16, ANN::ANLayerOutput);   // output layer of size: 16

layer1.ConnectLayer(&layer2);
layer2.ConnectLayer(&layer3);

ANN::BPNet cpu;
cpu.AddLayer(&layer1);
cpu.AddLayer(&layer2);
cpu.AddLayer(&layer3);
```

## 3. Hopfield networks

Hopfield networks have been described by John Hopfield. 
They serve as content-addressable memory system with binary threshold units. 
They are guaranteed to converge to a local minimum, but convergence to one of the stored patterns is not guaranteed. 

```
ANN::HFNet cpu;
cpu.Resize(16,16); // create a hopfield net of size: 16X16
```

## 4. Self organizing maps (SOMs)

### 1. CPU implementation

Self-organizing maps (SOM) are a type of artificial neural network that is trained using unsupervised learning to produce a low-dimensional (typically two-dimensional), 
representation of training samples. 
SOMs are different from other artificial neural networks in the sense that they use a neighbourhood function for learning. 
The example shows a SOM 128x128 node network, each node can process a 3x1 input vector.

```
ANN::SOMNet cpu;
cpu.SetTrainingSet(input);
cpu.CreateSOM(3, 1, 128, 128); // create a SOM of size: 128X128
```

### 2. GPGPU implementation

At the moment it is possible to speed up the calculation of SOMs with processing on the side of the GPU. 
One can train, save and load the net with CPU as well and later continue with GPU and the other way round, 
simply by calling the CTOR:

```
ANN::SOMNet cpu;
cpu.CreateSOM(3, 1, 128, 128); // input w, h, net w, h

ANN::SOMNetGPU gpu(&cpu);      // use copy CTOR or create it like above
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

# Implementation guide

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
To decrease the probability that someone forgets to overload some important functions are declared as pure abstract. 
This approach makes it possible the implement a new neuron class and use it with already implemented layer or network classes.

```
virtual void AdaptEdges()       = 0;
virtual void CalcValue()        = 0;
```

It doesn't make sense to implement them in every case (e. g. in Hopfield networks).
if not needed just implement them empty.

```
void HFNeuron::AdaptEdges() {
}
```

In other cases (e.g. back propagation networks) things are a little bit different. 
In CalcValue() you have to implement the algorithm to calculate the data you want to store in a certain neuron. 
Every neuron (or node) in the network has a list of edges which direct to neurons of another (or the same) layer. 
This example shows you how to run through this list to implement a neuron in a back propagation network.

```
void BPNeuron::CalcValue() {
        if(GetConsI().size() == 0)
                return;

        // bias neuron
        float fBias = 0.f;
        SetValue( 0.f );
        if(GetBiasEdge() ) {
                fBias = GetBiasEdge()->GetValue();
                SetValue(fBias);
        }

        // sum from product of all incoming neurons with their weights
        AbsNeuron *from;
        for(unsigned int i = 0; i < GetConsI().size(); i++) {
                from = GetConI(i)->GetDestination(this);
                SetValue(GetValue() + (from->GetValue() * GetConI(i)->GetValue()));
        }

        float fVal = GetNetFunction()->normal( GetValue(), fBias );
        SetValue(fVal);
}
```
 
The algorithm to adapt the edges is implemented in AdaptEdges(). 
Again we use the internal list to run through all edges (outgoing ones) the neuron is connected with.

```
void BPNeuron::AdaptEdges() {
        if(GetConsO().size() == 0)
                return;

        AbsNeuron *pCurNeuron;
        Edge            *pCurEdge;
        float           fVal;

        // calc error deltas
        fVal = GetErrorDelta();
        for(unsigned int i = 0; i < GetConsO().size(); i++) {
                pCurNeuron = GetConO(i)->GetDestination(this);
                fVal += pCurNeuron->GetErrorDelta() * GetConO(i)->GetValue();
        }
        fVal *= GetNetFunction()->derivate( GetValue(), 0.f );
        SetErrorDelta(fVal);

        // adapt weights
        for(unsigned int i = 0; i < GetConsO().size(); i++) {
                pCurEdge = GetConO(i);
                if(pCurEdge->GetAdaptationState() == true) {
                        fVal = 0.f;     // delta for momentum
                        // stdard backpropagation
                        fVal += pCurEdge->GetDestination(this)->GetErrorDelta() * m_fLearningRate * GetValue()
                        // weight decay term
                        - m_fWeightDecay * pCurEdge->GetValue()
                        // momentum term
                        + m_fMomentum * pCurEdge->GetMomentum();

                        pCurEdge->SetMomentum( fVal );
                        pCurEdge->SetValue( fVal+pCurEdge->GetValue() );
                }
        }
}
```

## AbsLayer

Neurons are stored in layers. If you decide to write your own layer class, then you have to implement the Resize() function. 
This could be useful especially if you have strange layer topologies (e.g. 2-dimensional or 3-dimensional).

```
virtual void Resize(const unsigned int &iSize) = 0;
This is the proper implementation example.

void BPLayer::Resize(const unsigned int &iSize) {
        EraseAll();
        for(unsigned int i = 0; i < iSize; i++) {
                AbsNeuron *pNeuron = new BPNeuron(this);
                pNeuron->SetID(i);
                m_lNeurons.push_back(pNeuron);
        }
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
void BPNet::PropagateFW() {
        for(unsigned int i = 1; i < m_lLayers.size(); i++) {
                BPLayer *curLayer = ( (BPLayer*)GetLayer(i) );
                #pragma omp parallel for
                for(int j = 0; j < static_cast<int>( curLayer->GetNeurons().size() ); j++) {
                        curLayer->GetNeuron(j)->CalcValue();
                }
        }
}

void BPNet::PropagateBW() {
        for(int i = m_lLayers.size()-1; i >= 0; i--) {
                BPLayer *curLayer = ( (BPLayer*)GetLayer(i) );
                #pragma omp parallel for
                for(int j = 0; j < static_cast<int>( curLayer->GetNeurons().size() ); j++) {
                        curLayer->GetNeuron(j)->AdaptEdges();
                }

                #pragma omp parallel
                if(curLayer->GetBiasNeuron() != NULL) {
                        curLayer->GetBiasNeuron()->AdaptEdges();
                }
        }
}
```

Different Implementations use different types of layers.
This is why you may want to overload AddLayer().

```
void BPNet::AddLayer(const unsigned int &iSize, const LayerTypeFlag &flType) {
        AbsNet::AddLayer( new BPLayer(iSize, flType) );
}
```

# Demos

## Front-end with Qt

![Figure 2](https://cloud.githubusercontent.com/assets/4668178/8386410/59a60548-1c50-11e5-988b-4a89f0b69587.png)
__Figure 2: This shows how a basic Front end implementation with Qt may look like.__
In the example just back propagation networks are used. 
Basically, every class of the network is again represented as a Qt graphics item in the canvas.

## Learning letter/numbers with back propagation

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
                    
float fInp3[56] = { 0,0,0,0,0,0,0,
                    0,1,1,1,1,0,0,
                    0,0,0,0,0,1,0,
                    0,0,0,0,0,1,0,
                    0,1,1,1,1,0,0,
                    0,0,0,0,0,1,0,
                    0,0,0,0,0,1,0,
                    0,1,1,1,1,0,0 };
```

Create a network: 

```
int main(int argc, char *argv[]) {
  ANN::BPNet cpu_one;

  ANN::BPLayer layer1(56, ANN::ANLayerInput);
  ANN::BPLayer layer2(64, ANN::ANLayerHidden);
  ANN::BPLayer layer3(64, ANN::ANLayerHidden);
  ANN::BPLayer layer4(64, ANN::ANLayerHidden);
  ANN::BPLayer layer5(9, ANN::ANLayerOutput);

  layer1.ConnectLayer(&layer2);
  layer2.ConnectLayer(&layer3);
  layer3.ConnectLayer(&layer4);
  layer4.ConnectLayer(&layer5);

  cpu_one.AddLayer(&layer1);
  cpu_one.AddLayer(&layer2);
  cpu_one.AddLayer(&layer3);
  cpu_one.AddLayer(&layer4);
  cpu_one.AddLayer(&layer5);
```
Feed it with the input: 

```
	ANN::TrainingSet input;
	input.AddInput(fInp1, 56);
	input.AddOutput(fOut1, 9);
	input.AddInput(fInp2, 56);
	input.AddOutput(fOut2, 9);
	input.AddInput(fInp3, 56);
	input.AddOutput(fOut3, 9);
	input.AddInput(fInp4, 56);
	input.AddOutput(fOut4, 9);
	input.AddInput(fInp5, 56);
	input.AddOutput(fOut5, 9);
	input.AddInput(fInp6, 56);
	input.AddOutput(fOut6, 9);
	input.AddInput(fInp7, 56);
	input.AddOutput(fOut7, 9);
	input.AddInput(fInp8, 56);
	input.AddOutput(fOut8, 9);
	input.AddInput(fInp9, 56);
	input.AddOutput(fOut9, 9);
	input.AddInput(fInp10, 56);
	input.AddOutput(fOut10, 9);
	
	std::vector<float> errors;
	cpu_one.SetLearningRate(0.5);
	cpu_one.SetMomentum(0);
	cpu_one.SetWeightDecay(0);
	cpu_one.SetTrainingSet(input);
```

Finally, return an output:

```
	bool b = false;
	float f;
	errors = cpu_one.TrainFromData(300, 0, b, f);
	std::cout<< cpu_one <<std::endl;

	return 0;
}
```

## Learning functions: Define your own

The library supports a plug-in system for your own learning functions.
Here is demonstrated how to use it.

```
inline static float
fcn_nearest_nhood (float sigma0, float T, float lambda) {
	return sqrt(2.f);
}

int main(int argc, char *argv[]) {
	QApplication a(argc, argv);

	ANN::TrainingSet input;
	input.AddInput(red);
	input.AddInput(green);
	input.AddInput(dk_green);
	input.AddInput(blue);
	input.AddInput(dk_blue);
	input.AddInput(yellow);
	input.AddInput(orange);
	input.AddInput(purple);
	input.AddInput(black);
	input.AddInput(white);

	std::vector<float> vCol(3);
	int w1 = 40;
	int w2 = 4;

	ANN::SOMNet cpu;
	cpu.CreateSOM(3, 1, w1,w1);
	cpu.SetTrainingSet(input);
	cpu.SetConscienceRate(0.1);
	
	ANN::DistFunction distFn = ANN::Functions::fcn_gaussian;
	distFn.rad_decay = fcn_nearest_nhood;
	cpu.SetDistFunction(distFn);

	cpu.Training(1000);

	SOMReader w(w1, w1, w2);
	for(int x = 0; x < w1*w1; x++) {
		ANN::SOMNeuron *pNeur = (ANN::SOMNeuron*)((ANN::SOMLayer*)cpu.GetOPLayer())->GetNeuron(x);
		vCol[0] = pNeur->GetConI(0)->GetValue();
		vCol[1] = pNeur->GetConI(1)->GetValue();
		vCol[2] = pNeur->GetConI(2)->GetValue();

		w.SetField(QPoint(pNeur->GetPosition()[0], pNeur->GetPosition()[1]), vCol );
	}
	w.Save("SimpFnExtByCPU.png");
	return 0;
}
```

## GPGPU: Device function pointers:

Unfortunately with CUDA device function pointers are inconvenient to use. 
Since, linking of the library must be static and the function are hard coded, 
I am not even sure, whether device function pointers are really function pointers..
Nevertheless, this example illustrates how to define own CUDA-style learning or whatever functions. 

Make a prototype (*.h file):

```
#include "include/math/Functions.h"

void SetFcn(ANN::DistFunction *fcn);
```

.. and implement it (*.cu file): 

```
#include "SetFcn.h"


typedef float (*pDistanceFu) (float, float);

// nonsense function to show it
__device__ static float foo(float r, float t) {
	return t*pow(r, 2);
}

__device__ pDistanceFu pOwn = foo; 

void SetFcn(ANN::DistFunction *fcn) {
	pDistanceFu hOwn;
	cudaMemcpyFromSymbol(&hOwn, pOwn, sizeof(pDistanceFu) );
	fcn->distance = hOwn;
}
```

Finally, use it (*.cpp file):

```
ANN::DistFunction ownFn = {
	(char*)"own",
	NULL,
	fcn_rad_decay,
	fcn_lrate_decay
};

int main(int argc, char *argv[]) {
	QApplication a(argc, argv);

	TrainingSet input;
	input.AddInput(red);
	input.AddInput(green);
	input.AddInput(dk_green);
	input.AddInput(blue);
	input.AddInput(dk_blue);
	input.AddInput(yellow);
	input.AddInput(orange);
	input.AddInput(purple);
	input.AddInput(black);
	input.AddInput(white);

	std::vector<float> vCol(3);
	int w1 = 40;
	int w2 = 4;

	SOMNetGPU gpu;
	gpu.CreateSOM(3, 1, w1,w1);
	gpu.SetTrainingSet(input);
	
	SetFcn(&ownFn);
	gpu.SetDistFunction(ownFn);

	gpu.Training(1000);

	SOMReader w(w1, w1, w2);
	for(int x = 0; x < w1*w1; x++) {
		SOMNeuron *pNeur = (SOMNeuron*)((SOMLayer*)gpu.GetOPLayer())->GetNeuron(x);
		vCol[0] = pNeur->GetConI(0)->GetValue();
		vCol[1] = pNeur->GetConI(1)->GetValue();
		vCol[2] = pNeur->GetConI(2)->GetValue();

		w.SetField(QPoint(pNeur->GetPosition()[0], pNeur->GetPosition()[1]), vCol );
	}
	w.Save("SimpFnExtByGPU.png");
	return 0;
}
```

# Save custom models to file-system

## Back propagation networks


The standard implementation saves all the content of the classes you derive from. 
Nevertheless if you decide to add features you may want to overload the ExpToFS() and ImpFromFS() functions as well. 
Calling the virtual base class ensures to save the base content of the class you derived from. 
The following example shows how to add support for a bias neuron, which has to get handled different by the network. 
Currently, the export reference implementation uses bzip2 compression, because bigger nets may allocate much space.

```
void BPLayer::ExpToFS(BZFILE* bz2out, int iBZ2Error) {
        std::cout<<"Save BPLayer to FS()"<<std::endl;
        AbsLayer::ExpToFS(bz2out, iBZ2Error);

        unsigned int iNmbOfConnects     = 0;
        float fEdgeValue        = 0.0f;
        int iDstLayerID         = -1;
        int iDstNeurID          = -1;

        bool bHasBias = false;
        (GetBiasNeuron() == NULL) ? bHasBias = false : bHasBias = true;
        BZ2_bzWrite( &iBZ2Error, bz2out, &bHasBias, sizeof(bool) );

        if(bHasBias) {
                AbsNeuron *pCurNeur = GetBiasNeuron();
                iNmbOfConnects = pCurNeur->GetConsO().size();
                BZ2_bzWrite( &iBZ2Error, bz2out, &iNmbOfConnects, sizeof(int) );
                for(unsigned int k = 0; k < iNmbOfConnects; k++) {
                        Edge *pCurEdge = pCurNeur->GetConO(k);
                        iDstLayerID = pCurEdge->GetDestination(pCurNeur)->GetParent()->GetID();
                        iDstNeurID = pCurEdge->GetDestinationID(pCurNeur);
                        fEdgeValue = pCurEdge->GetValue();
                        BZ2_bzWrite( &iBZ2Error, bz2out, &iDstLayerID, sizeof(int) );
                        BZ2_bzWrite( &iBZ2Error, bz2out, &iDstNeurID, sizeof(int) );
                        BZ2_bzWrite( &iBZ2Error, bz2out, &fEdgeValue, sizeof(float) );
                }
        }
}
```

Now the other way round, we load the content from the file-system.

```
int BPLayer::ImpFromFS(BZFILE* bz2in, int iBZ2Error, ConTable &Table) {
        std::cout<<"Load BPLayer from FS()"<<std::endl;
        int iLayerID = AbsLayer::ImpFromFS(bz2in, iBZ2Error, Table);

        unsigned int iNmbOfConnects     = 0;
        float fEdgeValue        = 0.0f;
        int iDstLayerID         = -1;
        int iDstNeurID          = -1;

        bool bHasBias = false;

        BZ2_bzRead( &iBZ2Error, bz2in, &bHasBias, sizeof(bool) );

        if(bHasBias) {
                BZ2_bzRead( &iBZ2Error, bz2in, &iNmbOfConnects, sizeof(int) );
                for(unsigned int j = 0; j < iNmbOfConnects; j++) {
                        BZ2_bzRead( &iBZ2Error, bz2in, &iDstLayerID, sizeof(int) );
                        BZ2_bzRead( &iBZ2Error, bz2in, &iDstNeurID, sizeof(int) );
                        BZ2_bzRead( &iBZ2Error, bz2in, &fEdgeValue, sizeof(float) );
                        ConDescr        cCurCon;
                        cCurCon.m_fVal                  = fEdgeValue;
                        cCurCon.m_iDstNeurID    = iDstNeurID;
                        cCurCon.m_iSrcLayerID   = iLayerID;
                        cCurCon.m_iDstLayerID   = iDstLayerID;
                        Table.BiasCons.push_back(cCurCon);
                }
        }

        return iLayerID;
}
```

The last thing: overloading CreateNet(). 
Here the content loaded from the filesystem is used to create a copy of the net in the memory. 
The base implementation creates the layers and the connections of the network, so we just have to implement the bias neuron.

```
/*
 * Init
 */
unsigned int iDstNeurID         = 0;
unsigned int iDstLayerID        = 0;
unsigned int iSrcLayerID        = 0;

float fEdgeValue                = 0.f;

AbsLayer *pDstLayer             = NULL;
AbsLayer *pSrcLayer             = NULL;
AbsNeuron *pDstNeur             = NULL;
AbsNeuron *pSrcNeur             = NULL;
Then we call the base implementation to connect the layers:

/*
 * For all nets necessary: Create Connections (Edges)
 */
AbsNet::CreateNet(Net);
Then we handle our special bias neuron which we previously added to our back propagation network:

/*
 * Only for back propagation networks
 */
if(Net.NetType == ANNetBP) {
        for(unsigned int i = 0; i < Net.BiasCons.size(); i++) {
                iDstNeurID = Net.BiasCons.at(i).m_iDstNeurID;
                iDstLayerID = Net.BiasCons.at(i).m_iDstLayerID;
                iSrcLayerID = Net.BiasCons.at(i).m_iSrcLayerID;
                if(iDstNeurID < 0 || iDstLayerID < 0 || GetLayers().size() < iDstLayerID || GetLayers().size() < iSrcLayerID) {
                        return;
                }
                        else {
                        fEdgeValue      = Net.BiasCons.at(i).m_fVal;

                        pDstLayer       = ( (BPLayer*)GetLayer(iDstLayerID) );
                        pSrcLayer       = ( (BPLayer*)GetLayer(iSrcLayerID) );
                        pSrcNeur        = ( (BPLayer*)pSrcLayer)->GetBiasNeuron();

                        pDstNeur        = pDstLayer->GetNeuron(iDstNeurID);
                        Connect(pSrcNeur, pDstNeur, fEdgeValue, 0.f, true);
                }
        }
}
```

## Self organizing maps

Here is another example for SOMs. Only the import of the positions has to be added.

```
/*
 * For all nets necessary: Create Connections (Edges)
 */
AbsNet::CreateNet(Net);

/*
 * Set Positions
 */
for(unsigned int i = 0; i < Net.Neurons.size(); i++) {
        int iLayerID    = Net.Neurons.at(i).m_iLayerID;
        int iNeurID     = Net.Neurons.at(i).m_iNeurID;
        std::vector<float> vPos = Net.Neurons.at(i).m_vPos;

        GetLayer(iLayerID)->GetNeuron(iNeurID)->SetPosition(vPos);
}
```

## Overloading the net description struct

The transient struct which stores the network could get extended by overloading. 
You are free to extend it for your needs and use it with the functions shown in this guide.

```
struct ConTable {
        NetTypeFlag                     NetType;
        unsigned int                    NrOfLayers;

        std::vector<unsigned int>       SizeOfLayer;
        std::vector<LayerTypeFlag>      TypeOfLayer;

        std::vector<NeurDescr>          Neurons;

        std::vector<ConDescr>           BiasCons;
        std::vector<ConDescr>           NeurCons;
};
```
