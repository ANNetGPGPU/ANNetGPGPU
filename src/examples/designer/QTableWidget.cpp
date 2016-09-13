/*
 * QTableWidget.cpp
 *
 *  Created on: 06.07.2012
 *      Author: Daniel <dgrat> Frenzel
 */

#include <QTableWidget.h>
#include <iostream>


TableWidget::TableWidget() {
	// TODO Auto-generated constructor stub

}

TableWidget::~TableWidget() {
	// TODO Auto-generated destructor stub
}

void TableWidget::copy()
{
	QItemSelectionModel * selection = selectionModel();
	QModelIndexList indexes = selection->selectedIndexes();

	std::cout<<"indices: "<<indexes.size()<<std::endl;

	if(indexes.size() < 1)
		return;

	// QModelIndex::operator < sorts first by row, then by column.
	// this is what we need
	std::sort(indexes.begin(), indexes.end());

	// You need a pair of indexes to find the row changes
	QModelIndex previous = indexes.first();
	QString selected_text;
	QModelIndex current;
	if(indexes.size() > 1) {
		indexes.removeFirst();

		Q_FOREACH(current, indexes) {
			QVariant data = model()->data(previous);
			QString text = data.toString();
			// At this point `text` contains the text in one cell
			selected_text.append(text);
			// If you are at the start of the row the row number of the previous index
			// isn't the same.  Text is followed by a row separator, which is a newline.
			if (current.row() != previous.row()) {
				selected_text.append(QLatin1Char('\n'));
			}
			// Otherwise it's the same row, so append a column separator, which is a tab.
			else {
			  selected_text.append(QLatin1Char('\t'));
			}
			previous = current;
		}
	}
	else if(indexes.size() == 1) {
		QVariant data = model()->data(previous);
		QString text = data.toString();

		selected_text.append(text);
	}

	// add last element
	selected_text.append(model()->data(current).toString());
	selected_text.append(QLatin1Char('\n'));
	qApp->clipboard()->setText(selected_text);
}

void TableWidget::paste() {
	QString selected_text = qApp->clipboard()->text();
	QStringList cells = selected_text.split(QRegExp(QLatin1String("\n|\t")));

	while(!cells.empty() && cells.back().size() == 0) {
		cells.pop_back(); // strip empty trailing tokens
	}

	int rows = selected_text.count(QLatin1Char('\n'));
	int cols = cells.size() / rows;

	if(cells.size() % rows != 0) {
		// error, uneven number of columns, probably bad data
		QMessageBox::critical(this, tr("Error"),
		tr("Invalid clipboard data, unable to perform paste operation."));
		return;
	}
/*
	if(cols != columnCount()) {
		// error, clipboard does not match current number of columns
		QMessageBox::critical(this, tr("Error"), tr("Invalid clipboard data, incorrect number of columns."));
		return;
	}

	// don't clear the grid, we want to keep any existing headers
	setRowCount(rows);
	setColumnCount(cols);
*/
	int cell = 0;
	for(int row = 0; row < rows; ++row) {
		for(int col = 0; col < cols; ++col, ++cell) {
			QTableWidgetItem *newItem = new QTableWidgetItem(cells[cell]);
			if(row+currentRow() < rowCount() && col+currentColumn() < columnCount() ) {
				setItem(row+currentRow(), col+currentColumn(), newItem);
			}
		}
	}
}

void TableWidget::keyPressEvent(QKeyEvent * event)
{
	if(event->matches(QKeySequence::Copy) )	{
		copy();
	}
	else if(event->matches(QKeySequence::Paste) ) {
		paste();
	}
	else {
		QTableWidget::keyPressEvent(event);
	}
}
