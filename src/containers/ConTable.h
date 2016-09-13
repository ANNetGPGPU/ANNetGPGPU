/// -*- tab-width: 8; Mode: C++; c-basic-offset: 8; indent-tabs-mode: t -*-
/*
   This program is free software: you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.
   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.
   You should have received a copy of the GNU General Public License
   along with this program.  If not, see <http://www.gnu.org/licenses/>.
   
   Author: Daniel Frenzel (dgdanielf@gmail.com)
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
