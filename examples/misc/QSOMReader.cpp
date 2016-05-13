#include "QSOMReader.h"


SOMReader::SOMReader(const unsigned int &iWidth, const unsigned int iHeight,
                     const unsigned int &iFieldSize,
                     QWidget *parent) :
    QLabel(parent)
{
	m_pImage = NULL;
	Resize(iWidth, iHeight, iFieldSize);
	Fill(Qt::white);
}

SOMReader::~SOMReader() {
	if(m_pImage != NULL)
		delete m_pImage;
}

void SOMReader::SetField(const QPoint &pField, const QColor &color) {
	for(unsigned int y = pField.y()*m_iFieldSize+pField.y()+1; y < pField.y()*m_iFieldSize+pField.y()+m_iFieldSize+1; y++) {
		for(unsigned int x = pField.x()*m_iFieldSize+pField.x()+1; x < pField.x()*m_iFieldSize+pField.x()+m_iFieldSize+1; x++) {
			m_pImage->setPixel(x, y, color.rgb() );
		}
	}
	this->setPixmap(QPixmap::fromImage(*m_pImage));
}

void SOMReader::SetField(const QPoint &pField, const std::vector<float> &vColor) {
	for(unsigned int y = pField.y()*m_iFieldSize+pField.y()+1; y < pField.y()*m_iFieldSize+pField.y()+m_iFieldSize+1; y++) {
		for(unsigned int x = pField.x()*m_iFieldSize+pField.x()+1; x < pField.x()*m_iFieldSize+pField.x()+m_iFieldSize+1; x++) {
			m_pImage->setPixel(x, y, qRgb((vColor[0]*255.f), (vColor[1]*255.f), (vColor[2]*255.f)));
		}
	}
	this->setPixmap(QPixmap::fromImage(*m_pImage));
}

void SOMReader::Resize(	const unsigned int &iWidth, const unsigned int iHeight,
        				const unsigned int &iFieldSize)
{
	if(m_pImage != NULL)
		delete m_pImage;

	m_iWidth 	= iWidth;
	m_iHeight 	= iHeight;
	m_iFieldSize 	= iFieldSize;

	m_pImage = new QImage(m_iWidth*iFieldSize+m_iWidth+1, m_iHeight*iFieldSize+m_iHeight+1, QImage::Format_RGB888);
	m_pImage->fill(Qt::black);
	this->setPixmap(QPixmap::fromImage(*m_pImage));
}

void SOMReader::Fill(const QColor &color) {
	for(unsigned int y = 0; y < m_iHeight; y++) {
		for(unsigned int x = 0; x < m_iWidth; x++) {
			SetField(QPoint(x, y), color);
		}
	}
}

void SOMReader::Save(const QString &sFileName){
	m_pImage->save(sFileName);
}
