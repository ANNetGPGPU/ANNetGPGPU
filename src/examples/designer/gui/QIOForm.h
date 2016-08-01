#ifndef QIOFORM_H
#define QIOFORM_H

#include <ANNet>
#include <ANContainers>
#include <gui/QTableWidget.h>
#include <QtGui>


class IOForm : public QWidget
{
    Q_OBJECT
    
signals:
    void si_contentChanged();

public:
    explicit IOForm(QWidget *parent = 0);
    ~IOForm();

    void setNmbrOfSets(const int iNmbr);
    int getNumberOfSets() const;

    ANN::TrainingSet *getTrainingSet();
    void setTrainingSet(ANN::TrainingSet *);

    void reset();

public slots:
    void sl_createTables(ANN::BPNet *pNet);
    void sl_setNmbrOfSets();

private:
    TableWidget *m_pITable;
    TableWidget *m_pOTable;

    unsigned int m_iDataSets;

    ANN::BPNet *m_pNet;
    ANN::TrainingSet *m_pTrainingSet;

private slots:
    void sl_send();
};

#endif // QIOFORM_H
