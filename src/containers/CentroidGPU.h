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
#include <type_traits>
#include <iostream>
#include <vector>
#include <thrust/host_vector.h>
#endif


namespace ANNGPGPU {
	template <class Type>
	class Centroid
	{
	public:
		Type _distance = static_cast<Type>(0);
		int32_t _unitID = -1;
		int32_t _deviceID = -1;
		thrust::host_vector<Type> _position;
		thrust::host_vector<Type> _edges;
		std::vector<Type> _input;
		
		Centroid(Type fDist = 0, int32_t iUnit = -1, int32_t iDev = -1) {
			this->_distance = fDist;
			this->_deviceID = iDev;
			this->_unitID = iUnit;
		}

		bool operator<(const Centroid &rhs) const {
			return this->_distance < rhs._distance;
		}
		bool operator>(const Centroid &rhs) const {
			return this->_distance > rhs._distance;
		}
		bool operator<=(const Centroid &rhs) const {
			return this->_distance <= rhs._distance;
		}
		bool operator>=(const Centroid &rhs) const {
			return this->_distance >= rhs._distance;
		}
		bool operator==(const Centroid &rhs) const {
			return this->_distance == rhs._distance;
		}
		bool operator!=(const Centroid &rhs) const {
			return this->_distance != rhs._distance;
		}
		
		#ifdef __Centroid_ADDON
			#include __Centroid_ADDON
		#endif
	};
};

template <class T>
std::ostream& operator << (std::ostream &os, ANNGPGPU::Centroid<T> &op) {
	for(unsigned int i = 0; i < op._edges.size(); i++) {
		std::cout<<"Centroid["<<i<<"]: "<<op._edges[i]<<std::endl;
	}
	std::cout<<"Euclidean distance: "<<op._distance<<std::endl;
	return os;
}
