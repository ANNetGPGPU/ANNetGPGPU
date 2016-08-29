/*
 * BPLayer.cpp
 */


template <class Type, class Functor>
BPLayer<Type, Functor>::BPLayer() {
	m_iZLayer = -1;
}

template <class Type, class Functor>
BPLayer<Type, Functor>::BPLayer(int iZLayer) {
	m_iZLayer = iZLayer;
}

template <class Type, class Functor>
BPLayer<Type, Functor>::BPLayer(const BPLayer *pLayer, int iZLayer) {
	int iNumber 		= pLayer->GetNeurons().size();
	LayerTypeFlag fType 	= pLayer->GetFlag();

	m_iZLayer = iZLayer;

	this->Resize(iNumber);
	this->SetFlag(fType);
}

template <class Type, class Functor>
BPLayer<Type, Functor>::BPLayer(const unsigned int &iNumber, LayerTypeFlag fType, int iZLayer) {
	this->Resize(iNumber);
	this->SetFlag(fType);

	m_iZLayer = iZLayer;
}

template <class Type, class Functor>
void BPLayer<Type, Functor>::SetZLayer(int iZLayer) {
	m_iZLayer = iZLayer;
}

template <class Type, class Functor>
int BPLayer<Type, Functor>::GetZLayer() {
	return m_iZLayer;
}

template <class Type, class Functor>
void BPLayer<Type, Functor>::Resize(const unsigned int &iSize) {
	this->EraseAll();
	this->AddNeurons(iSize);
}

template <class Type, class Functor>
void BPLayer<Type, Functor>::AddNeurons(const unsigned int &iSize) {
	for(unsigned int i = 0; i < iSize; i++) {
		AbsNeuron<Type> *pNeuron = new BPNeuron<Type, Functor>(this);
		this->m_lNeurons.push_back(pNeuron);
		pNeuron->SetID(this->m_lNeurons.size()-1);
	}
}

template <class Type, class Functor>
void BPLayer<Type, Functor>::ConnectLayer(AbsLayer<Type> *pDestLayer, const bool &bAllowAdapt) {
	AbsNeuron<Type> *pSrcNeuron;

	for(unsigned int i = 0; i < this->m_lNeurons.size(); i++) {
		pSrcNeuron = this->m_lNeurons[i];
		ANN::Connect(pSrcNeuron, pDestLayer, bAllowAdapt);
	}
}

template <class Type, class Functor>
void BPLayer<Type, Functor>::ConnectLayer(
		AbsLayer<Type> *pDestLayer,
		std::vector<std::vector<int> > Connections,
		const bool bAllowAdapt)
{
	AbsNeuron<Type> *pSrcNeuron;

	assert( Connections.size() != this->m_lNeurons.size() );
	for(unsigned int i = 0; i < Connections.size(); i++) {
		std::vector<int> subArray = Connections.at(i);
		pSrcNeuron = this->GetNeuron(i);
		assert(i != pSrcNeuron->GetID() );

		for(unsigned int j = 0; j < subArray.size(); j++) {
			assert( j < pDestLayer->GetNeurons().size() );
			AbsNeuron<Type> *pDestNeuron = pDestLayer->GetNeuron(j);
			assert( j < pDestNeuron->GetID() );
			ANN::Connect(pSrcNeuron, pDestNeuron, bAllowAdapt);
		}
	}
}

template <class Type, class Functor>
void BPLayer<Type, Functor>::SetFlag(const LayerTypeFlag &fType) {
	this->m_fTypeFlag = fType;
}

template <class Type, class Functor>
void BPLayer<Type, Functor>::AddFlag(const LayerTypeFlag &fType) {
	if(!(this->m_fTypeFlag & fType) )
	this->m_fTypeFlag |= fType;
}

template <class Type, class Functor>
void BPLayer<Type, Functor>::Setup(const HebbianConf<Type> &config) {
	#pragma omp parallel for
	for(int j = 0; j < static_cast<int>(this->m_lNeurons.size() ); j++) {
		((BPNeuron<Type, Functor>*)this->m_lNeurons[j])->Setup(config);
	}
}

template <class Type, class Functor>
void BPLayer<Type, Functor>::ImpMomentumsEdgesIn(const F2DArray<Type> &mat) {
	unsigned int iHeight 	= this->m_lNeurons.at(0)->GetConsI().size();
	unsigned int iWidth 	= this->m_lNeurons.size();

	assert(iHeight == mat.GetH() );
	assert(iWidth == mat.GetW() );

	#pragma omp parallel for
	for(int y = 0; y < static_cast<int>(iHeight); y++) {
		for(unsigned int x = 0; x < iWidth; x++) {
			this->m_lNeurons.at(x)->GetConI(y)->SetMomentum(mat[y][x]);
		}
	}
}

template <class Type, class Functor>
void BPLayer<Type, Functor>::ImpMomentumsEdgesOut(const F2DArray<Type> &mat) {
	unsigned int iHeight 	= this->m_lNeurons.at(0)->GetConsO().size();
	unsigned int iWidth 	= this->m_lNeurons.size();

	assert(iHeight == mat.GetH() );
	assert(iWidth == mat.GetW() );

	#pragma omp parallel for
	for(int y = 0; y < static_cast<int>(iHeight); y++) {
		for(unsigned int x = 0; x < iWidth; x++) {
			this->m_lNeurons.at(x)->GetConO(y)->SetMomentum(mat[y][x]);
		}
	}
}

template <class Type, class Functor>
void BPLayer<Type, Functor>::ExpToFS(BZFILE* bz2out, int iBZ2Error) {
	std::cout<<"Save BPLayer to FS()"<<std::endl;
	AbsLayer<Type>::ExpToFS(bz2out, iBZ2Error);

	int iZLayer = m_iZLayer;
	BZ2_bzWrite( &iBZ2Error, bz2out, &iZLayer, sizeof(int) );
}

template <class Type, class Functor>
int BPLayer<Type, Functor>::ImpFromFS(BZFILE* bz2in, int iBZ2Error, ConTable<Type> &Table) {
	std::cout<<"Load BPLayer from FS()"<<std::endl;
	int iLayerID = AbsLayer<Type>::ImpFromFS(bz2in, iBZ2Error, Table);

	int iZLayer = -1;
	BZ2_bzRead( &iBZ2Error, bz2in, &iZLayer, sizeof(int) );
	Table.ZValOfLayer.push_back(iZLayer);

	return iLayerID;
}
