/**************************************************************************
**
** This file is part of Qt Creator
**
** Copyright (c) 2011 Nokia Corporation and/or its subsidiary(-ies).
**
** Contact: Nokia Corporation (info@qt.nokia.com)
**
**
** GNU Lesser General Public License Usage
**
** This file may be used under the terms of the GNU Lesser General Public
** License version 2.1 as published by the Free Software Foundation and
** appearing in the file LICENSE.LGPL included in the packaging of this file.
** Please review the following information to ensure the GNU Lesser General
** Public License version 2.1 requirements will be met:
** http://www.gnu.org/licenses/old-licenses/lgpl-2.1.html.
**
** In addition, as a special exception, Nokia gives you certain additional
** rights. These rights are described in the Nokia Qt LGPL Exception
** version 1.1, included in the file LGPL_EXCEPTION.txt in this package.
**
** Other Usage
**
** Alternatively, this file may be used in accordance with the terms and
** conditions contained in a signed written agreement between you and Nokia.
**
** If you have questions regarding the use of this file, please contact
** Nokia at qt-info@nokia.com.
**
**************************************************************************/

#include <gui/3rdparty/utils/styledbar.h>
#include <gui/3rdparty/utils/stylehelper.h>

#include <QtCore/QVariant>
#include <QtGui/QPainter>
#include <QtGui/QPixmapCache>
#include <QtGui/QStyle>
#include <QtGui/QStyleOption>

using namespace Utils;

StyledBar::StyledBar(QWidget *parent)
    : QWidget(parent)
{
    setProperty("panelwidget", true);
    setProperty("panelwidget_singlerow", true);
    setProperty("lightColored", false);
}

void StyledBar::setSingleRow(bool singleRow)
{
    setProperty("panelwidget_singlerow", singleRow);
}

bool StyledBar::isSingleRow() const
{
    return property("panelwidget_singlerow").toBool();
}

void StyledBar::setLightColored(bool lightColored)
{
    setProperty("lightColored", lightColored);
}

bool StyledBar::isLightColored() const
{
    return property("lightColored").toBool();
}

void StyledBar::paintEvent(QPaintEvent *event)
{
    Q_UNUSED(event)
    QPainter painter(this);
    QStyleOption option;
    option.rect = rect();
    option.state = QStyle::State_Horizontal;
    style()->drawControl(QStyle::CE_ToolBar, &option, &painter, this);
}

StyledSeparator::StyledSeparator(QWidget *parent)
    : QWidget(parent)
{
    setFixedWidth(10);
}

void StyledSeparator::paintEvent(QPaintEvent *event)
{
    Q_UNUSED(event)
    QPainter painter(this);
    QStyleOption option;
    option.rect = rect();
    option.state = QStyle::State_Horizontal;
    option.palette = palette();
    style()->drawPrimitive(QStyle::PE_IndicatorToolBarSeparator, &option, &painter, this);
}
