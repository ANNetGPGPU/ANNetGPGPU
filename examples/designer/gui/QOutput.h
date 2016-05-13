/*
 * QOutput.h
 *
 *  Created on: 01.07.2012
 *      Author: Daniel <dgrat> Frenzel
 */

#ifndef QOUTPUT_H_
#define QOUTPUT_H_

#include <QtGui>
#include <ANNet>


class Output: public QWidget {
private:
	QTableWidget *m_pTableWidget;

public:
	Output(QWidget *parent = NULL);
	virtual ~Output();

	void display(ANN::BPNet *pNet);
	void reset();
};

#endif /* QOUTPUT_H_ */
