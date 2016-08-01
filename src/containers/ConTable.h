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
template <class T>
struct ConDescr {
	int m_iSrcLayerID;
	int m_iDstLayerID;

	int m_iSrcNeurID;
	int m_iDstNeurID;

	T m_fVal;
	std::vector<T> m_vMisc; // information for addons
	
#ifdef __ConDescr_ADDON
	#include __ConDescr_ADDON
#endif
};

template <class T>
struct NeurDescr {
	int m_iLayerID;
	int m_iNeurID;

	std::string m_sTransFunction;
	std::string m_sDistFunction;

	std::vector<T> m_vMisc; // information for addons
	
#ifdef __NeurDescr_ADDON
	#include __NeurDescr_ADDON
#endif
};

/**
 * \brief Represents a container for all connections (edges/weights) in the network.
 *
 * @author Daniel "dgrat" Frenzel
 */
template <class T>
struct ConTable {
	NetTypeFlag NetType;
	unsigned int NrOfLayers;

	std::vector<unsigned int> SizeOfLayer;
	std::vector<int> ZValOfLayer;
	std::vector<LayerTypeFlag> TypeOfLayer;
	std::vector<NeurDescr<T> > Neurons;
	std::vector<ConDescr<T> > NeurCons;
	std::vector<T> m_vMisc; // information for addons
	
#ifdef __ConTable_ADDON
	#include __ConTable_ADDON
#endif
};

}
