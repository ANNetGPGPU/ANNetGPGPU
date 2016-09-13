/*
 * QGraphTab.cpp
 *
 *  Created on: 06.07.2012
 *      Author: Daniel <dgrat> Frenzel
 */

#include <QGraphTab.h>


GraphTab::GraphTab() {
	QVBoxLayout *pLayout = new QVBoxLayout(this);

	QFont font;
	font.setBold(true);
	font.setPointSize(16);

	QLabel *pLabel 	= new QLabel("Training progress");
	pLabel->setFont(font);
	m_pTabWidget 	= new QTabWidget;
	m_pTabWidget->setTabsClosable(true);
	//m_pTabWidget->setTabShape(QTabWidget::Triangular);

	pLayout->addWidget(pLabel);
	pLayout->addWidget(m_pTabWidget);

	QObject::connect(m_pTabWidget, SIGNAL(tabCloseRequested(int)), this, SLOT(sl_closeTab(int)));
}

GraphTab::~GraphTab() {
	// TODO Auto-generated destructor stub
}

QTabWidget *GraphTab::getTabWidget() const {
	return m_pTabWidget;
}

void GraphTab::sl_closeTab(int iID) {
	QCustomPlot *pTab = (QCustomPlot *)m_pTabWidget->widget(iID);
	m_pTabWidget->removeTab(iID);
	delete pTab;
}
