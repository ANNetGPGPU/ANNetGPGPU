#ifndef QIOFORM_H
#define QIOFORM_H

#include <ANNet>
#include <ANContainers>
#include <QTableWidget.h>
#include <QtGui>


class IOForm : public QWidget
{
    Q_OBJECT
    
signals:
    void si_contentChanged();

public:
    explicit IOForm(QWidget *parent = 0);
    virtual ~IOForm();

    void setNmbrOfSets(const int iNmbr);
    int getNumberOfSets() const;

    ANN::TrainingSet<float> *getTrainingSet();
    void setTrainingSet(ANN::TrainingSet<float> *);

    void reset();

public slots:
    void sl_createTables(ANN::BPNet<float, ANN::fcn_log<float>> *pNet);
    void sl_setNmbrOfSets();

private:
    TableWidget *m_pITable;
    TableWidget *m_pOTable;

    unsigned int m_iDataSets;

    ANN::BPNet<float, ANN::fcn_log<float>> *m_pNet;
    ANN::TrainingSet<float> *m_pTrainingSet;

private slots:
    void sl_send();
};

#endif // QIOFORM_H
