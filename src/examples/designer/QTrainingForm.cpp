#include <QTrainingForm.h>
#include <ui_QTrainingForm.h>


TrainingForm::TrainingForm(QWidget *parent) :
    QWidget(parent),
    ui(new Ui::TrainingForm)
{
    ui->setupUi(this);
}

TrainingForm::~TrainingForm()
{
    delete ui;
}

int TrainingForm::getMaxCycles() const {
	return ui->m_SBMax->value();
}

float TrainingForm::getMaxError() const {
	return (float)ui->m_SBError->value();
}

std::string TrainingForm::getTransfFunct() const {
	return ui->m_CBTransferFunction->currentText().toStdString();
}

float TrainingForm::getLearningRate() const {
	return (float)ui->m_SBLearningRate->value();
}

float TrainingForm::getMomentum() const {
	return (float)ui->m_SBMomentum->value();
}

float TrainingForm::getWeightDecay() const {
	return (float)ui->m_SBWeightDecay->value();
}
