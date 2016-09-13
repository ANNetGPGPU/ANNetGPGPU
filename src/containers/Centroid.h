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
