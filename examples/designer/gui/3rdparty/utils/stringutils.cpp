/**************************************************************************
**
** This file is part of Qt Creator
**
** Copyright (c) 2012 Nokia Corporation and/or its subsidiary(-ies).
**
** Contact: Nokia Corporation (qt-info@nokia.com)
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

#include <gui/3rdparty/utils/stringutils.h>

#include <QString>
#include <QStringList>
#include <QFileInfo>
#include <QDir>

#include <limits.h>

namespace Utils {

/*QTCREATOR_UTILS_EXPORT*/ QString settingsKey(const QString &category)
{
    QString rc(category);
    const QChar underscore = QLatin1Char('_');
    // Remove the sort category "X.Category" -> "Category"
    if (rc.size() > 2 && rc.at(0).isLetter() && rc.at(1) == QLatin1Char('.'))
        rc.remove(0, 2);
    // Replace special characters
    const int size = rc.size();
    for (int i = 0; i < size; i++) {
        const QChar c = rc.at(i);
        if (!c.isLetterOrNumber() && c != underscore)
            rc[i] = underscore;
    }
    return rc;
}

// Figure out length of common start of string ("C:\a", "c:\b"  -> "c:\"
static inline int commonPartSize(const QString &s1, const QString &s2)
{
    const int size = qMin(s1.size(), s2.size());
    for (int i = 0; i < size; i++)
        if (s1.at(i) != s2.at(i))
            return i;
    return size;
}

/*QTCREATOR_UTILS_EXPORT*/ QString commonPrefix(const QStringList &strings)
{
    switch (strings.size()) {
    case 0:
        return QString();
    case 1:
        return strings.front();
    default:
        break;
    }
    // Figure out common string part: "C:\foo\bar1" "C:\foo\bar2"  -> "C:\foo\bar"
    int commonLength = INT_MAX;
    const int last = strings.size() - 1;
    for (int i = 0; i < last; i++)
        commonLength = qMin(commonLength, commonPartSize(strings.at(i), strings.at(i + 1)));
    if (!commonLength)
        return QString();
    return strings.at(0).left(commonLength);
}

/*QTCREATOR_UTILS_EXPORT*/ QString commonPath(const QStringList &files)
{
    QString common = commonPrefix(files);
    // Find common directory part: "C:\foo\bar" -> "C:\foo"
    int lastSeparatorPos = common.lastIndexOf(QLatin1Char('/'));
    if (lastSeparatorPos == -1)
        lastSeparatorPos = common.lastIndexOf(QLatin1Char('\\'));
    if (lastSeparatorPos == -1)
        return QString();
#ifdef Q_OS_UNIX
    if (lastSeparatorPos == 0) // Unix: "/a", "/b" -> '/'
        lastSeparatorPos = 1;
#endif
    common.truncate(lastSeparatorPos);
    return common;
}

/*QTCREATOR_UTILS_EXPORT*/ QString withTildeHomePath(const QString &path)
{
#ifdef Q_OS_WIN
    QString outPath = path;
#else
    static const QString homePath = QDir::homePath();

    QFileInfo fi(QDir::cleanPath(path));
    QString outPath = fi.absoluteFilePath();
    if (outPath.startsWith(homePath))
        outPath = QLatin1Char('~') + outPath.mid(homePath.size());
    else
        outPath = path;
#endif
    return outPath;
}

int AbstractQtcMacroExpander::findMacro(const QString &str, int *pos, QString *ret)
{
    forever {
        int openPos = str.indexOf(QLatin1String("%{"), *pos);
        if (openPos < 0)
            return 0;
        int varPos = openPos + 2;
        int closePos = str.indexOf(QLatin1Char('}'), varPos);
        if (closePos < 0)
            return 0;
        int varLen = closePos - varPos;
        if (resolveMacro(str.mid(varPos, varLen), ret)) {
            *pos = openPos;
            return varLen + 3;
        }
        // An actual expansion may be nested into a "false" one,
        // so we continue right after the last %{.
        *pos = varPos;
    }
}

/*QTCREATOR_UTILS_EXPORT*/ void expandMacros(QString *str, AbstractMacroExpander *mx)
{
    QString rsts;

    for (int pos = 0; int len = mx->findMacro(*str, &pos, &rsts); ) {
        str->replace(pos, len, rsts);
        pos += rsts.length();
    }
}

/*QTCREATOR_UTILS_EXPORT*/ QString expandMacros(const QString &str, AbstractMacroExpander *mx)
{
    QString ret = str;
    expandMacros(&ret, mx);
    return ret;
}

} // namespace Utils
