/*
 * QTrainingThread.h
 *
 *  Created on: 08.07.2012
 *      Author: Daniel <dgrat> Frenzel
 */

#ifndef QTRAININGTHREAD_H_
#define QTRAININGTHREAD_H_

#include <QtGui>
#include <ANNet>


class TrainingThread: public QThread {
private:
	ANN::BPNet<float, ANN::fcn_log<float>> *m_pNet;
	int m_iCycles;
	float m_fError;
	float m_fProgress;

	bool *m_pBreak;

	std::vector<float> m_fErrors;

public:
	TrainingThread(QObject *parent = NULL);
	virtual ~TrainingThread();

	void setNet(ANN::BPNet<float, ANN::fcn_log<float>> *pNet, int iCycles, float fError, bool &bBreak);
	std::vector<float> getErrors() const;
	float getProgress() const;

	void run();
};

#endif /* QTRAININGTHREAD_H_ */
