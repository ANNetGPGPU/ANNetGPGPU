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

#ifndef NETCONNECTIONTABLE_H_
#define NETCONNECTIONTABLE_H_

#ifndef SWIG
#include <string>
#include <stdint.h>
#endif

namespace ANN {
  
typedef uint32_t LayerTypeFlag;
typedef uint32_t NetTypeFlag;

/**
 * \brief Represents a container for a connection (edge/weight) in the network.
 *
 * @author Daniel "dgrat" Frenzel
 */
struct ConDescr {
	int m_iSrcLayerID;
	int m_iDstLayerID;

	int m_iSrcNeurID;
	int m_iDstNeurID;

	float m_fVal;
};

struct NeurDescr {
	int m_iLayerID;
	int m_iNeurID;

	std::string m_sTransFunction;	// TODO implement
	std::string m_sDistFunction;	// TODO .. as well

	std::vector<float> m_vPos;
};

/**
 * \brief Represents a container for all connections (edges/weights) in the network.
 *
 * @author Daniel "dgrat" Frenzel
 */
struct ConTable {
	NetTypeFlag 				NetType;
	unsigned int 				NrOfLayers;

	std::vector<unsigned int> 	SizeOfLayer;
	std::vector<int> 			ZValOfLayer;
	std::vector<LayerTypeFlag> 	TypeOfLayer;

	std::vector<NeurDescr> 		Neurons;

	std::vector<ConDescr> 		BiasCons;		// TODO not elegant
	std::vector<ConDescr> 		NeurCons;
};

}
#endif /* NETCONNECTIONTABLE_H_ */
