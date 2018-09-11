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

template <class Type, bool OnGPU>
void F2DArray<Type, OnGPU>::SetValue(const size_t &iX, const size_t &iY, const Type &fVal) {
	assert(iY < this->m_iY);
	assert(iX < this->m_iX);

	this->m_data[iY*this->m_iX+iX] = fVal;
}

template <class Type, bool OnGPU>
Type F2DArray<Type, OnGPU>::GetValue(const size_t &iX, const size_t &iY) const {
	assert(iY < this->m_iY);
	assert(iX < this->m_iX);

	return this->m_data[iY*this->m_iX+iX];
}

template <class Type, bool OnGPU>
void F2DArray<Type, OnGPU>::GetOutput() {
	for(size_t y = 0; y < GetH(); y++) {
		for(size_t x = 0; x < GetW(); x++) {
			std::cout << "Array["<<x<<"]["<<y<<"]=" << GetValue(x, y) << std::endl;
		}
	}
}

