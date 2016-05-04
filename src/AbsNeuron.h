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

#ifndef ABSNEURON_H_
#define ABSNEURON_H_

#ifndef SWIG
#include <vector>
#include <string>
#include <bzlib.h>
#endif

namespace ANN {
  
// containers
class F2DArray;
class F3DArray;
class TrainingSet;
class ConTable;
// math
class TransfFunction;
// net
class AbsLayer;
class AbsNeuron;
class Edge;

/**
 * @class AbsNeuron
 * @brief Abstract class describing a basic neuron in a network.
 * Pure virtual functions must get implemented if deriving from this class.
 * These functions a doing the back-/propagation jobs.
 * You can modify the behavior of the complete net by overloading them.
 * @author Daniel "dgrat" Frenzel
 */
class AbsNeuron {
protected:
	std::vector<float> m_vPosition;			// x, y, z, .. coordinates of the neuron (e.g. SOM)
	float m_fValue;					// value of the neuron in the net

	float m_fErrorDelta;				// Current error delta of this neuron

	AbsLayer *m_pParentLayer;			// layer which is inheriting this neuron
	int m_iNeuronID;				// ID of this neuron in the layer

	Edge *m_pBias;					// Pointer to the bias edge (or connection to bias neuron)

	//ANN::list<Edge*> m_lOutgoingConnections;
	//ANN::list<Edge*> m_lIncomingConnections;

	std::vector<Edge*> m_lOutgoingConnections;
	std::vector<Edge*> m_lIncomingConnections;

	TransfFunction *m_ActFunction;

public:
	/**
	 * @brief Creates a new neuron with parent layer: *pParentLayer
	 */
	AbsNeuron(AbsLayer *pParentLayer = NULL);
	/**
	 * @brief Copy constructor for creation of a new neuron with the "same" properties like *pNeuron. 
	 * This constructor can't copy connections (edges), because they normally have dependencies to other neurons.
	 * @param pNeuron object to copy properties from
	 */
	AbsNeuron(const AbsNeuron *pNeuron);
	virtual ~AbsNeuron();

	/**
	 * @brief Deletes all weights of this neuron
	 */
	void EraseAllEdges();

	/**
	 * @brief Pointer to the layer inherting this neuron.
	 */
	AbsLayer *GetParent() const;

	/**
	 * @brief Appends an edge to the list of incoming edges.
	 */
	virtual void AddConI(Edge *ANEdge);
	/**
	 * @brief Appends an edge to the list of outgoing edges.
	 */
	virtual void AddConO(Edge *ANEdge);

	virtual void SetConO(Edge *Edge, const unsigned int iID);
	virtual void SetConI(Edge *Edge, const unsigned int iID);

	/**
	 * @brief Returns a pointer to an incoming weight
	 * @return Pointer to an incoming edge
	 * @param iID Index of edge in m_lIncomingConnections
	 */
	virtual Edge* GetConI(const unsigned int &iID) const;
	/**
	 * @brief Returns a pointer to an outgoing weight
	 * @return Pointer to an outgoing edge
	 * @param iID Index of edge in m_lOutgoingConnections
	 */
	virtual Edge* GetConO(const unsigned int &iID) const;
	/**
	 * @brief Returns all incoming weights
	 * @return Array of pointers of all incoming edges
	 */
	virtual std::vector<Edge*> GetConsI() const;
	/**
	 * @brief Returns all outgoing weights
	 * @return Array of pointers of all outgoing edges
	 */
	virtual std::vector<Edge*> GetConsO() const;
	/**
	 * @brief Sets the index of this neuron to a certain value.
	 * @param iID New index of this neuron.
	 */
	virtual void SetID(const int iID);
	/**
	 * @brief Returns the index of this neuron.
	 * @return Index of this neuron.
	 */
	virtual unsigned int GetID() const;
	/**
	 * @brief Set the neuron to a certain value.
	 * @param fValue New value of this neuron.
	 */
	virtual void SetValue(const float &fValue);
	/**
	 * @brief Returns the value of this neuron.
	 * @return Returns the value of this neuron.
	 */
	virtual const float &GetValue() const;
	/**
	 * @brief Returns the postion coordinates of this neuron
	 * @return x, y, z, .. coordinates of the neuron (e.g. SOM)
	 */
	virtual const std::vector<float> GetPosition() const;
	/**
	 * @brief Sets the current position of the neuron in the net.
	 * @param vPos Vector with Cartesian coordinates
	 */
	virtual void SetPosition(const std::vector<float> &vPos);
	/**
	 * @brief Sets the error delta of this neuron to a certain value.
	 * @param fValue New error delts of this neuron.
	 */
	virtual void SetErrorDelta(const float &fValue);
	/**
	 * @brief Returns the current error delta of this neuron.
	 * @return Returns the error delta of this neuron.
	 */
	virtual const float &GetErrorDelta() const;
	/**
	 * @brief Defines a bias weight for this neuron.
	 * @param pANEdge Pointer to edge connecting this neuron with bias neuron.
	 */
	virtual void SetBiasEdge(Edge *pANEdge);
	/**
	 * @brief Returns the current bias weight for this neuron.
	 * @return Returns pointer to edge connecting this neuron with bias neuron.
	 */
	virtual Edge *GetBiasEdge() const;
	/**
	 * @brief Sets the transfer function for this neuron.
	 * @param pFCN Kind of function the net has to use while back-/propagating.
	 */
	virtual void SetTransfFunction (const TransfFunction *pFCN);
	/**
	 * @brief Returns the used transfer function of this neuron.
	 * @return The transfer function of the net.
	 */
	virtual const TransfFunction *GetTransfFunction() const;

	/**
	 * @brief Overload to define how the net has to act while propagating back.
	 * I.e. how to modify the edges after calculating the error deltas.
	 */
	virtual void AdaptEdges() 	= 0;
	/**
	 * @brief Overload to define how the net has to act while propagating.
	 * I.e. which neurons/edges to use for calculating the new value of the neuron
	 */
	virtual void CalcValue() 	= 0;

	/**
	 * @brief Save neuron's content to filesystem
	 */
	virtual void ExpToFS(BZFILE* bz2out, int iBZ2Error);
	/**
	 * @brief Load neuron's content to filesystem
	 * @return The connections table of this neuron.
	 */
	virtual void ImpFromFS(BZFILE* bz2in, int iBZ2Error, ConTable &Table);

	/* QUASI STATIC:*/

	/**
	 * @brief standard output of the net. Only usable if input/output layer was already set.
	 */
	friend std::ostream& operator << (std::ostream &os, AbsNeuron &op);
	/**
	 * @brief standard output of the net. Only usable if input/output layer was already set.
	 */
	friend std::ostream& operator << (std::ostream &os, AbsNeuron *op);

	/**
	 * @brief Connects a neuron with another neuron
	 * @param bAdaptState indicates whether the connection is changeable
	 */
	friend void Connect(AbsNeuron *pSrcNeuron, AbsNeuron *pDstNeuron, const bool &bAdaptState);
	/**
	 * @brief Connects a neuron with a complete layer
	 * @param bAdaptState indicates whether the connections are changeable
	 */
	friend void Connect(AbsNeuron *pSrcNeuron, AbsLayer  *pDestLayer, const bool &bAdaptState);
	/**
	 * @brief Connects a neuron with another neuron
	 * Allows to set the value of the connection and the current momentum
	 * @param bAdaptState indicates whether the connection is changeable
	 */
	friend void Connect(AbsNeuron *pSrcNeuron, AbsNeuron *pDstNeuron, const float &fVal, const float &fMomentum, const bool &bAdaptState);
	/**
	 * @brief Connects a neuron with another layer
	 */
	friend void Connect(AbsNeuron *srcNeuron, AbsLayer *destLayer, const std::vector<float> &vValues, const std::vector<float> &vMomentums, const bool &bAdaptState);

	/*
	 * Returns the value of the neuron.
	 */
	operator float() const;
};

}
#endif /* ABSNEURON_H_ */
