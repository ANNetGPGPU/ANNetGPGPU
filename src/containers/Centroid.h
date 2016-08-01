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
#include <iostream>
#include <vector>
#endif

namespace ANN {
/*
 * Representation of the centroid as N-dimensional structure of the size of the input vector(s) and
 * with the euclidean distance(s) to the input vector(s)
 */
template <class Type>
struct Centroid {
	std::vector<Type> m_vInput;
	std::vector<Type> m_vCentroid;
	
	unsigned int m_iBMUID;
	Type m_fEucDist;

	bool operator<(const Centroid &rhs) const {
		return m_iBMUID < rhs.m_iBMUID;
	}
	bool operator>(const Centroid &rhs) const {
		return m_iBMUID > rhs.m_iBMUID;
	}
	bool operator<=(const Centroid &rhs) const {
		return m_iBMUID <= rhs.m_iBMUID;
	}
	bool operator>=(const Centroid &rhs) const {
		return m_iBMUID >= rhs.m_iBMUID;
	}
	bool operator==(const Centroid &rhs) const {
		return m_iBMUID == rhs.m_iBMUID;
	}
	bool operator!=(const Centroid &rhs) const {
		return m_iBMUID != rhs.m_iBMUID;
	}
	
#ifdef __Centroid_ADDON
	#include __Centroid_ADDON
#endif
}; 

}

template <class T>
std::ostream& operator << (std::ostream &os, ANN::Centroid<T> &op) {
	for(unsigned int i = 0; i < op.m_vCentroid.size(); i++) {
		std::cout<<"Centroid["<<i<<"]: "<<op.m_vCentroid[i]<<std::endl;
	}
	std::cout<<"Euclidean distance: "<<op.m_fEucDist<<std::endl;
	return os;     // Ref. auf Stream
}
