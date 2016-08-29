template <class Type>
AbsLayer<Type>::AbsLayer() {

}

template <class Type>
AbsLayer<Type>::~AbsLayer() {
	EraseAll();
}

template <class Type>
void AbsLayer<Type>::EraseAll() {
	for(unsigned int i = 0; i < m_lNeurons.size(); i++) {
		delete m_lNeurons[i];
	}
	m_lNeurons.clear();
}

template <class Type>
void AbsLayer<Type>::SetID(const int &iID) {
	m_iID = iID;
}

template <class Type>
int AbsLayer<Type>::GetID() const {
	return m_iID;
}

template <class Type>
const std::vector<AbsNeuron<Type> *> &AbsLayer<Type>::GetNeurons() const {
	return m_lNeurons;
}

template <class Type>
AbsNeuron<Type> *AbsLayer<Type>::GetNeuron(const unsigned int &iID) const {
	// quick try
	if(m_lNeurons.at(iID)->GetID() == iID) {
		return m_lNeurons.at(iID);
	}
	// fall back scenario
	else {
		for(unsigned int i = 0; i < m_lNeurons.size(); i++) {
			if(m_lNeurons.at(i)->GetID() == iID)
				return m_lNeurons.at(i);
		}
	}

	return NULL;
}

template <class Type>
void AbsLayer<Type>::SetFlag(const LayerTypeFlag &fType) {
	m_fTypeFlag = fType;
}

template <class Type>
void AbsLayer<Type>::AddFlag(const LayerTypeFlag &fType) {
	if( !(m_fTypeFlag & fType) )
	m_fTypeFlag |= fType;
}

template <class Type>
LayerTypeFlag AbsLayer<Type>::GetFlag() const {
	return m_fTypeFlag;
}

template <class Type>
void AbsLayer<Type>::ExpToFS(BZFILE* bz2out, int iBZ2Error) {
	int iLayerID = GetID();
	BZ2_bzWrite( &iBZ2Error, bz2out, &iLayerID, sizeof(int) );

	std::cout<<"Save AbsLayer to FS()"<<std::endl;

	LayerTypeFlag fLayerType = GetFlag();
	unsigned int iNmbOfNeurons = this->GetNeurons().size();

	BZ2_bzWrite( &iBZ2Error, bz2out, &fLayerType, sizeof(LayerTypeFlag) );	// Type of layer
	BZ2_bzWrite( &iBZ2Error, bz2out, &iNmbOfNeurons, sizeof(int) );		// Number of neuron in this layer (except bias)
	for(unsigned int j = 0; j < iNmbOfNeurons; j++) {
		AbsNeuron<Type> *pCurNeur = this->GetNeuron(j);
		pCurNeur->ExpToFS(bz2out, iBZ2Error);
	}
}

template <class Type>
int AbsLayer<Type>::ImpFromFS(BZFILE* bz2in, int iBZ2Error, ConTable<Type> &Table) {
	int iLayerID = -1;
	BZ2_bzRead( &iBZ2Error, bz2in, &iLayerID, sizeof(int) );

	std::cout<<"Load AbsLayer from FS()"<<std::endl;

	LayerTypeFlag fLayerType = 0;
	unsigned int iNmbOfNeurons = 0;

	BZ2_bzRead( &iBZ2Error, bz2in, &fLayerType, sizeof(LayerTypeFlag) );
	BZ2_bzRead( &iBZ2Error, bz2in, &iNmbOfNeurons, sizeof(int) );
	Table.TypeOfLayer.push_back(fLayerType);
	Table.SizeOfLayer.push_back(iNmbOfNeurons);

	for(unsigned int j = 0; j < iNmbOfNeurons; j++) {
		AddNeurons(1); // Create dummy neuron; more neurons than needed don't disturb, but are necessary if using empty nets

		NeurDescr<Type> cCurNeur;
		cCurNeur.m_iLayerID = iLayerID;
		Table.Neurons.push_back(cCurNeur);
		AbsNeuron<Type> *pCurNeur = this->GetNeuron(j);
		pCurNeur->ImpFromFS(bz2in, iBZ2Error, Table);
	}

	return iLayerID;
}

template <class Type>
F2DArray<Type> AbsLayer<Type>::ExpEdgesIn() const {
	unsigned int iHeight 	= m_lNeurons.front()->GetConsI().size();
	unsigned int iWidth 	= m_lNeurons.size();

	assert(iWidth > 0 && iHeight > 0);

	F2DArray<Type> vRes;
	vRes.Alloc(iWidth, iHeight);

	#pragma omp parallel for
	for(int y = 0; y < static_cast<int>(iHeight); y++) {
		for(unsigned int x = 0; x < iWidth; x++) {
			vRes[y][x] = m_lNeurons.at(x)->GetConI(y)->GetValue();
		}
	}
	return vRes;
}

template <class Type>
F2DArray<Type> AbsLayer<Type>::ExpEdgesIn(int iStart, int iStop) const {
	unsigned int iWidth 	= iStop-iStart+1;
	unsigned int iHeight 	= m_lNeurons.front()->GetConsI().size();

	assert(iWidth > 0);
	assert(iStart >= 0);
	assert(iStop < m_lNeurons.size() );

	F2DArray<Type> vRes;
	vRes.Alloc(iWidth, iHeight);
	
	#pragma omp parallel for
	for(int y = 0; y < static_cast<int>(iHeight); y++) {
		int iC = 0;
		for(unsigned int x = iStart; x <= iStop; x++) {
			vRes[y][iC] = m_lNeurons.at(x)->GetConI(y)->GetValue();
			iC++;
		}
	}
	return vRes;
}

template <class Type>
F2DArray<Type> AbsLayer<Type>::ExpEdgesOut() const {
	unsigned int iHeight 	= m_lNeurons.front()->GetConsO().size();
	unsigned int iWidth 	= m_lNeurons.size();

	assert(iWidth > 0 && iHeight > 0);

	F2DArray<Type> vRes;
	vRes.Alloc(iWidth, iHeight);

	#pragma omp parallel for
	for(int y = 0; y < static_cast<int>(iHeight); y++) {
		for(unsigned int x = 0; x < iWidth; x++) {
			vRes[y][x] = m_lNeurons.at(x)->GetConO(y)->GetValue();
		}
	}
	return vRes;
}

template <class Type>
F2DArray<Type> AbsLayer<Type>::ExpEdgesOut(int iStart, int iStop) const {
	unsigned int iWidth 	= iStop-iStart+1;
	unsigned int iHeight 	= m_lNeurons.front()->GetConsO().size();

	assert(iWidth > 0);
	assert(iStart >= 0);
	assert(iStop < m_lNeurons.size() );

	F2DArray<Type> vRes;
	vRes.Alloc(iWidth, iHeight);
	
	#pragma omp parallel for
	for(int y = 0; y < static_cast<int>(iHeight); y++) {
		int iC = 0;
		for(unsigned int x = iStart; x <= iStop; x++) {
			vRes[y][iC] = m_lNeurons.at(x)->GetConI(y)->GetValue();
			iC++;
		}
	}
	return vRes;
}

template <class Type>
void AbsLayer<Type>::ImpEdgesIn(const F2DArray<Type> &mat) {
	unsigned int iHeight 	= m_lNeurons.front()->GetConsI().size();
	unsigned int iWidth 	= m_lNeurons.size();

	assert(iHeight == mat.GetH() );
	assert(iWidth == mat.GetW() );

	#pragma omp parallel for
	for(int y = 0; y < static_cast<int>(iHeight); y++) {
		for(unsigned int x = 0; x < iWidth; x++) {
			m_lNeurons.at(x)->GetConI(y)->SetValue(mat[y][x]);
		}
	}
}

template <class Type>
void AbsLayer<Type>::ImpEdgesIn(const F2DArray<Type> &mat, int iStart, int iStop) {
	unsigned int iHeight 	= m_lNeurons.front()->GetConsI().size();
	
	assert(iHeight == mat.GetH() );
	assert(iStop-iStart <= mat.GetW() );

	#pragma omp parallel for
	for(int y = 0; y < static_cast<int>(iHeight); y++) {
		int iC = 0;
		for(unsigned int x = iStart; x <= iStop; x++) {
			m_lNeurons.at(x)->GetConI(y)->SetValue(mat[y][iC]);
			iC++;
		}
	}
}

template <class Type>
void AbsLayer<Type>::ImpEdgesOut(const F2DArray<Type> &mat) {
	unsigned int iHeight 	= m_lNeurons.front()->GetConsO().size();
	unsigned int iWidth 	= m_lNeurons.size();

	assert(iHeight == mat.GetH() );
	assert(iWidth == mat.GetW() );

	#pragma omp parallel for
	for(int y = 0; y < static_cast<int>(iHeight); y++) {
		for(unsigned int x = 0; x < iWidth; x++) {
			m_lNeurons.at(x)->GetConO(y)->SetValue(mat[y][x]);
		}
	}
}

template <class Type>
void AbsLayer<Type>::ImpEdgesOut(const F2DArray<Type> &mat, int iStart, int iStop) {
	unsigned int iHeight 	= m_lNeurons.front()->GetConsO().size();
	
	assert(iHeight == mat.GetH() );
	assert(iStop-iStart <= mat.GetW() );

	#pragma omp parallel for
	for(int y = 0; y < static_cast<int>(iHeight); y++) {
		int iC = 0;
		for(unsigned int x = iStart; x <= iStop; x++) {
			m_lNeurons.at(x)->GetConO(y)->SetValue(mat[y][iC]);
			iC++;
		}
	}
}

template <class Type>
F2DArray<Type> AbsLayer<Type>::ExpPositions() const {
	unsigned int iHeight 	= m_lNeurons.at(0)->GetPosition().size();
	unsigned int iWidth 	= m_lNeurons.size();
	
	assert(iWidth > 0 && iHeight > 0);

	F2DArray<Type> vRes;
	vRes.Alloc(iWidth, iHeight);

	#pragma omp parallel for
	for(int y = 0; y < static_cast<int>(iHeight); y++) {
		for(unsigned int x = 0; x < iWidth; x++) {
			vRes[y][x] = m_lNeurons.at(x)->GetPosition().at(y);
		}
	}
	return vRes;
}

template <class Type>
F2DArray<Type> AbsLayer<Type>::ExpPositions(int iStart, int iStop) const {
	unsigned int iHeight 	= m_lNeurons.at(0)->GetPosition().size();
	unsigned int iWidth 	= iStop-iStart+1;

	assert(iStop-iStart <= m_lNeurons.size() );
	assert(iStart >= 0);
	assert(iStop < m_lNeurons.size() );

	F2DArray<Type> vRes;
	vRes.Alloc(iWidth, iHeight);

	#pragma omp parallel for
	for(int y = 0; y < static_cast<int>(iHeight); y++) {
		int iC = 0;
		for(int x = iStart; x <= iStop; x++) {
			vRes[y][iC] = m_lNeurons.at(x)->GetPosition().at(y);
			iC++;
		}
	}
	return vRes;
}

template <class Type>
void AbsLayer<Type>::ImpPositions(const F2DArray<Type> &f2dPos) {
	unsigned int iHeight = f2dPos.GetH();
	unsigned int iWidth = f2dPos.GetW();

	assert(iWidth == m_lNeurons.size() );

	#pragma omp parallel for
	for(int x = 0; x < static_cast<int>(iWidth); x++) {
		std::vector<Type> vPos(iHeight);
		for(unsigned int y = 0; y < iHeight; y++) {
			vPos[y] = f2dPos.GetValue(x, y);
		}
		m_lNeurons.at(x)->SetPosition(vPos);
	}
}

template <class Type>
void AbsLayer<Type>::ImpPositions(const F2DArray<Type> &f2dPos, int iStart, int iStop) {
	unsigned int iHeight = f2dPos.GetH();
	unsigned int iWidth = f2dPos.GetW();

	assert(iStop-iStart <= m_lNeurons.size() );
	
	int iC = 0;
	#pragma omp parallel for
	for(int x = iStart; x <= static_cast<int>(iStop); x++) {
		std::vector<Type> vPos(iHeight);
		for(unsigned int y = 0; y < iHeight; y++) {
			vPos[y] = f2dPos.GetValue(iC, y);
		}
		iC++;
		m_lNeurons.at(x)->SetPosition(vPos);
	}
}
