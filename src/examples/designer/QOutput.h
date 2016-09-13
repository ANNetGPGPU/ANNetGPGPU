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

	void display(ANN::BPNet<float, ANN::fcn_log<float>> *pNet);
	void reset();
};

#endif /* QOUTPUT_H_ */
