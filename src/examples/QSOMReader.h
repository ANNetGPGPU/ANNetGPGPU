/*
#-------------------------------------------------------------------------------
# Copyright (c) 2012 Daniel <dgrat> Frenzel.
# All rights reserved. This program and the accompanying materials
# are made available under the terms of the GNU Lesser Public License v2.1
# which accompanies this distribution, and is available at
# http://www.gnu.org/licenses/old-licenses/gpl-2.0.html
# 
# Contributors:
#     Daniel <dgrat> Frenzel - initial API and implementation
#-------------------------------------------------------------------------------
*/
#pragma once

#include <QtWidgets>
#include <QApplication>


class SOMReader : public QLabel {
Q_OBJECT
private:
	QImage *m_pImage;

	unsigned int m_iFieldSize;
	unsigned int m_iWidth;
	unsigned int m_iHeight;

public:
	explicit SOMReader(const unsigned int &iWidth, const unsigned int iHeight, // Height and Width in fields
			  const unsigned int &iFieldSize = 10,                    // Size of a field in pixels
			  QWidget *parent = 0);
	virtual ~SOMReader();

	void Resize(const unsigned int &iWidth, const unsigned int iHeight,
		    const unsigned int &iFieldSize = 10);

	void Save(const QString &sFileName);
    
public slots:
	void Fill(const QColor &color = Qt::white);
	void SetField(const QPoint &pField, const QColor &color);
	void SetField(const QPoint &pField, const std::vector<float> &vColor);
};

