/*
 * QTrainingThread.cpp
 *
 *  Created on: 08.07.2012
 *      Author: Daniel <dgrat> Frenzel
 */

#include <gui/QTrainingThread.h>


TrainingThread::TrainingThread(QObject *parent) : QThread(parent) {
	m_pNet 		= NULL;
	m_pBreak 	= NULL;
	m_iCycles 	= 0;
	m_fError 	= 0.f;
	m_fProgress = 0.f;
}

TrainingThread::~TrainingThread() {
	// TODO Auto-generated destructor stub
}

void TrainingThread::setNet(ANN::BPNet *pNet, int iCycles, float fError, bool &bBreak) {
	m_pNet 		= pNet;
	m_iCycles 	= iCycles;
	m_fError 	= fError;
	m_pBreak 	= &bBreak;
}

void TrainingThread::run() {
	if(m_pNet != NULL) {
		m_fErrors = m_pNet->TrainFromData(m_iCycles, m_fError, *m_pBreak, m_fProgress);
	}
	else {
		qDebug() << "Training failed";
	}
}

std::vector<float> TrainingThread::getErrors() const {
	return m_fErrors;
}

float TrainingThread::getProgress() const {
	return m_fProgress;
}
