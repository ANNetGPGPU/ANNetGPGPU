#include <gui/QIOForm.h>
#include <vector>
#include <gui/3rdparty/delegate/spinboxdelegate.h>


IOForm::IOForm(QWidget *parent) : QWidget(parent) {
    QVBoxLayout *pVBLayout 	= new QVBoxLayout(this);

    SpinBoxDelegate *pDelegate= new SpinBoxDelegate;

    m_pITable 				= new TableWidget;
    m_pITable->setItemDelegate(pDelegate);
    //m_pITable->setDragDropMode(QAbstractItemView::DragDrop);
    m_pOTable 				= new TableWidget;
    m_pOTable->setItemDelegate(pDelegate);
    //m_pOTable->setDragDropMode(QAbstractItemView::DragDrop);

	QFont font;
	font.setPointSize(16);
	font.setBold(true);

    QLabel *pILabel 		= new QLabel(QObject::tr("Input data:"));
    QLabel *pOLabel 		= new QLabel(QObject::tr("Output data:"));
    pILabel->setFont(font);
    pOLabel->setFont(font);

    m_iDataSets 			= 1;
	m_pNet 					= NULL;
	m_pTrainingSet 			= NULL;

    pVBLayout->addWidget(pILabel);
    pVBLayout->addWidget(m_pITable);
    pVBLayout->addWidget(pOLabel);
    pVBLayout->addWidget(m_pOTable);

    QObject::connect(m_pITable, SIGNAL(itemChanged(QTableWidgetItem *)), this, SLOT(sl_send()));
    QObject::connect(m_pOTable, SIGNAL(itemChanged(QTableWidgetItem *)), this, SLOT(sl_send()));
}

IOForm::~IOForm() {

}

void IOForm::sl_send() {
	if(!m_pNet || !m_pNet->GetIPLayer() || !m_pNet->GetOPLayer())
		return;

	emit si_contentChanged();
}

ANN::TrainingSet *IOForm::getTrainingSet() {
	if(!m_pNet || !m_pNet->GetIPLayer() || !m_pNet->GetOPLayer())
		return NULL;

	m_pTrainingSet = new ANN::TrainingSet;

	unsigned int iInpColSize = m_pNet->GetIPLayer()->GetNeurons().size();
	unsigned int iOutColSize = m_pNet->GetOPLayer()->GetNeurons().size();

	bool bOK;

	for(unsigned int x = 0; x < m_iDataSets; x++) {
		std::vector<float> vInp;
		for(unsigned int y = 0; y < iInpColSize; y++) {
			float fVal = m_pITable->item(y, x)->text().toFloat(&bOK);
			if(bOK)
				vInp.push_back(fVal);
			else return NULL;
		}
		m_pTrainingSet->AddInput(vInp);
	}
	for(unsigned int x = 0; x < m_iDataSets; x++) {
		std::vector<float> vOut;
		for(unsigned int y = 0; y < iOutColSize; y++) {
			float fVal = m_pOTable->item(y, x)->text().toFloat(&bOK);
			if(bOK)
				vOut.push_back(fVal);
			else return NULL;
		}
		m_pTrainingSet->AddOutput(vOut);
	}

	return m_pTrainingSet;
}

void IOForm::setTrainingSet(ANN::TrainingSet *pTSet) {
	if(pTSet == NULL) {
		return;
	}

	m_pTrainingSet = pTSet;

	int iHeight;
/*
	if(pTSet->GetNrElements() > m_iDataSets) {
		setNmbrOfSets(pTSet->GetNrElements());
		sl_createTables(m_pNet);
	}
*/
	// Set the size of the input table
	for(unsigned int x = 0; x < m_iDataSets; x++) {

		if(x < pTSet->GetNrElements()) {
			if(pTSet->GetInput(x).size() > m_pITable->rowCount() )
				iHeight = m_pITable->rowCount();
			else iHeight = pTSet->GetInput(x).size();

			for(unsigned int y = 0; y < iHeight; y++) {
				if(y < pTSet->GetInput(x).size() ) {
					float fVal = pTSet->GetInput(x)[y];
					QModelIndex index = m_pITable->model()->index(y, x, QModelIndex());
	        		m_pITable->model()->setData(index, QVariant(fVal));
				}
			}
		}
	}

	// Set the size of the input table
	for(unsigned int x = 0; x < m_iDataSets; x++) {

		if(x < pTSet->GetNrElements()) {
			if(pTSet->GetInput(x).size() > m_pOTable->rowCount() )
				iHeight = m_pOTable->rowCount();
			else iHeight = pTSet->GetOutput(x).size();

			for(unsigned int y = 0; y < iHeight; y++) {
				if(y < pTSet->GetOutput(x).size() ) {
					float fVal = pTSet->GetOutput(x)[y];
					QModelIndex index = m_pOTable->model()->index(y, x, QModelIndex());
					m_pOTable->model()->setData(index, QVariant(fVal));
				}
			}
		}
	}
}

int IOForm::getNumberOfSets() const {
	return m_iDataSets;
}

void IOForm::sl_setNmbrOfSets() {
    bool ok;
    int iNumber = QInputDialog::getInt(0, QObject::tr("Add number of training pairs"),
                                 QObject::tr("Number of pairs:"), 1, 1, 128*128, 1, &ok);

    if(ok) {
    	setNmbrOfSets(iNumber);
    	if(m_pNet)
    		sl_createTables(m_pNet);
    	if(m_pTrainingSet)
    		setTrainingSet(m_pTrainingSet);
    }
}

void IOForm::setNmbrOfSets(const int iNmbr) {
	m_iDataSets = iNmbr;
}

void IOForm::reset() {
	setNmbrOfSets(1);

//	m_pITable->clear();
	m_pITable->setRowCount(0);
	m_pITable->setColumnCount(0);
//	m_pOTable->clear();
	m_pOTable->setRowCount(0);
	m_pOTable->setColumnCount(0);
}

void IOForm::sl_createTables(ANN::BPNet *pNet) {
	if(!pNet || !pNet->GetIPLayer() || !pNet->GetOPLayer())
		return;

    disconnect(m_pITable, SIGNAL(itemChanged(QTableWidgetItem *)), this, SLOT(sl_send()));
    disconnect(m_pOTable, SIGNAL(itemChanged(QTableWidgetItem *)), this, SLOT(sl_send()));

	m_pNet 					= pNet;
	QTableWidgetItem *pItem = NULL;
	unsigned int iColSize 	= 0;

	QFont font;
	font.setBold(true);

	// Set the size of the input table
	iColSize = pNet->GetIPLayer()->GetNeurons().size();
	m_pITable->setColumnCount(m_iDataSets);
	m_pITable->setRowCount(iColSize);
	for(unsigned int x = 0; x < m_iDataSets; x++) {
		pItem = new QTableWidgetItem("Input\npair "+QString::number(x+1));
		pItem->setFont(font);
		m_pITable->setHorizontalHeaderItem(x, pItem);

		for(unsigned int y = 0; y < iColSize; y++) {
			pItem = new QTableWidgetItem("Neuron "+QString::number(y+1));
			pItem->setFont(font);
			m_pITable->setVerticalHeaderItem(y, pItem);

			pItem = new QTableWidgetItem;
			m_pITable->setItem(y, x, pItem);

            QModelIndex index = m_pITable->model()->index(y, x, QModelIndex());
            float fVal = 0.f;
            /*if(pNet->GetTrainingSet()) {
            	if(pNet->GetTrainingSet()->GetIArraySize() > y)
            		fVal = pNet->GetTrainingSet()->GetIArray()[y];
            }*/
            m_pITable->model()->setData(index, QVariant(fVal));
		}
	}
	// Set the size of the input table
	iColSize = pNet->GetOPLayer()->GetNeurons().size();
	m_pOTable->setColumnCount(m_iDataSets);
	m_pOTable->setRowCount(iColSize);
	for(unsigned int x = 0; x < m_iDataSets; x++) {
		pItem = new QTableWidgetItem("Output\npair "+QString::number(x+1));
		pItem->setFont(font);
		m_pOTable->setHorizontalHeaderItem(x, pItem);
		for(unsigned int y = 0; y < iColSize; y++) {
			pItem = new QTableWidgetItem("Neuron "+QString::number(y+1));
			pItem->setFont(font);
			m_pOTable->setVerticalHeaderItem(y, pItem);

			pItem = new QTableWidgetItem;
			m_pOTable->setItem(y, x, pItem);

            QModelIndex index = m_pOTable->model()->index(y, x, QModelIndex());
            float fVal = 0.f;
            /*if(pNet->GetTrainingSet()) {
            	if(pNet->GetTrainingSet()->GetOArraySize() > y)
            		fVal = pNet->GetTrainingSet()->GetOArray()[y];
            }*/
            m_pOTable->model()->setData(index, QVariant(fVal));
		}
	}

    connect(m_pITable, SIGNAL(itemChanged(QTableWidgetItem *)), this, SLOT(sl_send()));
    connect(m_pOTable, SIGNAL(itemChanged(QTableWidgetItem *)), this, SLOT(sl_send()));
}
