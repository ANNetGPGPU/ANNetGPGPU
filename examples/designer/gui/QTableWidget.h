/*
 * QTableWidget.h
 *
 *  Created on: 06.07.2012
 *      Author: Daniel <dgrat> Frenzel
 */

#ifndef QTABLEWIDGET_H_
#define QTABLEWIDGET_H_

#include <QtGui>


class TableWidget: public QTableWidget {
public:
	TableWidget();
	virtual ~TableWidget();

private:
	void copy();
	void paste();

protected:
	virtual void keyPressEvent(QKeyEvent * event);
};

#endif /* QTABLEWIDGET_H_ */
