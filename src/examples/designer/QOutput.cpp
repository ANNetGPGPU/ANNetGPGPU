/*
 * QOutput.cpp
 *
 *  Created on: 01.07.2012
 *      Author: Daniel <dgrat> Frenzel
 */

#include <QOutput.h>
#include <ANContainers>


Output::Output(QWidget *parent) : QWidget(parent) {
	QVBoxLayout *pLayout = new QVBoxLayout(this);

	m_pTableWidget = new QTableWidget;
	m_pTableWidget->setAlternatingRowColors(true);

	QFont font;
	font.setPointSize(16);
	font.setBold(true);

    QLabel *pLabel = new QLabel(QObject::tr("Network output:"));
    pLabel->setFont(font);

    pLayout->addWidget(pLabel);
	pLayout->addWidget(m_pTableWidget);
}

Output::~Output() {
	// TODO Auto-generated destructor stub
}

void Output::reset() {
	m_pTableWidget->setRowCount(0);
	m_pTableWidget->setColumnCount(0);
}

void Output::display(ANN::BPNet<float, ANN::fcn_log<float>> *pNet) {
	if(!pNet)
		return;

	QFont font;
	font.setBold(true);
	QTableWidgetItem *pItem;

	m_pTableWidget->setColumnCount(pNet->GetTrainingSet()->GetNrElements()*2);
	m_pTableWidget->setRowCount(pNet->GetOPLayer()->GetNeurons().size() );
	for(unsigned int i = 0; i < pNet->GetOPLayer()->GetNeurons().size(); i++) {
		pItem = new QTableWidgetItem("Neuron "+QString::number(i+1));
		pItem->setFont(font);
		m_pTableWidget->setVerticalHeaderItem(i, pItem);
	}

	for(unsigned int i = 0; i < pNet->GetTrainingSet()->GetNrElements(); i++) {
		pItem = new QTableWidgetItem("Wished\nfrom set: "+QString::number(i+1));
		pItem->setFont(font);
		m_pTableWidget->setHorizontalHeaderItem(2*i, pItem);

		pItem = new QTableWidgetItem("Achieved\nfrom set: "+QString::number(i+1));
		pItem->setFont(font);
		m_pTableWidget->setHorizontalHeaderItem(2*i+1, pItem);

		pNet->SetInput(pNet->GetTrainingSet()->GetInput(i) );
		pNet->PropagateFW();

		std::vector<float> vOut = pNet->GetTrainingSet()->GetOutput(i);
		for(unsigned int j = 0; j < vOut.size(); j++) {
			m_pTableWidget->setItem(j, 2*i, new QTableWidgetItem(QString::number(vOut.at(j))) );
		}
		vOut = pNet->GetOutput();
		for(unsigned int j = 0; j < vOut.size(); j++) {
			m_pTableWidget->setItem(j, 2*i+1, new QTableWidgetItem(QString::number(vOut.at(j))) );
		}
	}

}
