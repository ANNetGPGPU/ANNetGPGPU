/*
 * QGraphTab.h
 *
 *  Created on: 06.07.2012
 *      Author: Daniel <dgrat> Frenzel
 */

#ifndef QGRAPHTAB_H_
#define QGRAPHTAB_H_

#include <QtGui>
#include <3rdparty/qcustomplot.h>


class GraphTab: public QWidget {
	Q_OBJECT

private:
	QTabWidget *m_pTabWidget;

public:
	GraphTab();
	virtual ~GraphTab();

	QTabWidget *getTabWidget() const;

public slots:
	void sl_closeTab(int iID);
};

#endif /* QGRAPHTAB_H_ */
