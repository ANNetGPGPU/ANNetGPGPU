#include <QEdge.h>
#include <QNode.h>
#include <cmath>

static const double Pi = 3.14159265358979323846264338327950288419717;
static double TwoPi = 2.0 * Pi;


Edge::Edge(Node *pSourceNode, Node *pDestNode) : m_ArrowSize(10)
{
    m_Color = Qt::black;
    setAcceptedMouseButtons(0);
    m_pSource = pSourceNode;
    m_pDest = pDestNode;
    m_pSource->addEdgeO(this);
    m_pDest->addEdgeI(this);
    adjust();
    setZValue(0);
}

Edge::~Edge()
{
}

void Edge::setColor(QColor color) {
    m_Color = color;
}

Node *Edge::sourceNode() const
{
    return m_pSource;
}

Node *Edge::destNode() const
{
    return m_pDest;
}

void Edge::adjust()
{
    if (!m_pSource || !m_pDest)
        return;

    QLineF line(mapFromItem(m_pSource, 0, 0), mapFromItem(m_pDest, 0, 0));
    qreal length = line.length();

    prepareGeometryChange();

    if (length > qreal(20.)) {
        QPointF edgeOffset((line.dx() * 10) / length, (line.dy() * 10) / length);
        m_SourcePoint = line.p1() + edgeOffset;
        m_DestPoint = line.p2() - edgeOffset;
    } else {
        m_SourcePoint = m_DestPoint = line.p1();
    }
}

QRectF Edge::boundingRect() const
{
    if (!m_pSource || !m_pDest)
        return QRectF();

    qreal penWidth = 1;
    qreal extra = (penWidth + m_ArrowSize) / 2.0;

    return QRectF(m_SourcePoint, QSizeF(m_DestPoint.x() - m_SourcePoint.x(),
                                      m_DestPoint.y() - m_SourcePoint.y()))
        .normalized()
        .adjusted(-extra, -extra, extra, extra);
}

void Edge::paint(QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget) {
    if (!m_pSource || !m_pDest)
        return;

    QLineF line(m_SourcePoint, m_DestPoint);
    if (qFuzzyCompare(line.length(), qreal(0.)))
        return;

    // Draw the line itself
    painter->setPen(QPen(m_Color, 1, Qt::SolidLine, Qt::RoundCap, Qt::RoundJoin));
    painter->drawLine(line);

    // Draw the arrows
    double angle = ::acos(line.dx() / line.length());
    if (line.dy() >= 0)
        angle = TwoPi - angle;
/*
    QPointF sourceArrowP1 = m_SourcePoint + QPointF(sin(angle + Pi / 3) * m_ArrowSize,
                                                  cos(angle + Pi / 3) * m_ArrowSize);
    QPointF sourceArrowP2 = m_SourcePoint + QPointF(sin(angle + Pi - Pi / 3) * m_ArrowSize,
                                                  cos(angle + Pi - Pi / 3) * m_ArrowSize);
*/
    QPointF destArrowP1 = m_DestPoint + QPointF(sin(angle - Pi / 3) * m_ArrowSize,
                                              cos(angle - Pi / 3) * m_ArrowSize);
    QPointF destArrowP2 = m_DestPoint + QPointF(sin(angle - Pi + Pi / 3) * m_ArrowSize,
                                              cos(angle - Pi + Pi / 3) * m_ArrowSize);

    painter->setBrush(m_Color);
    //painter->drawPolygon(QPolygonF() << line.p1() << sourceArrowP1 << sourceArrowP2);
    painter->drawPolygon(QPolygonF() << line.p2() << destArrowP1 << destArrowP2);
}
