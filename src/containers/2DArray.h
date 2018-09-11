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
	#include <cassert>
	#include <vector>
	#include <iostream>
	#include <cstring>
#endif

#ifdef __CUDACC__
	template <class T> using iterator = thrust::detail::normal_iterator<thrust::device_ptr<T>>;
	template <class T> using const_iterator = thrust::detail::normal_iterator<thrust::device_ptr<const T>>;
#endif

namespace hidden {
#ifdef __CUDACC__
	template <class T> using dev_vector_t = thrust::device_vector<T>;
#endif
	template <class T> using vector_t = std::vector<T>;
	
	template <class T, class Arr> class F1DArray;
	template <class T, bool> class F2DArray;
	
	
	template <class Type, class Arr>
	class F1DArray {
	private:
		Arr _arr = nullptr;
		size_t m_iWX = 0;	// nr. of neurons in layer m_iY
		size_t m_iWY = 0;	// nr. of layer in net
		size_t m_iY = 0;
		
	public:
		F1DArray(Arr ref, const size_t wx, const size_t wy, const size_t y) : _arr(ref), m_iWX(wx), m_iWY(wy), m_iY(y) {}

		Type & operator[] (size_t x) {
			assert(_arr != nullptr);
			assert(x < m_iWX);
			assert(m_iY*m_iWX+x < m_iWX*m_iWY);
			return (*_arr)[m_iY*m_iWX+x];
		}

		const Type & operator[] (size_t x) const {
			assert(_arr != nullptr);
			assert(x < m_iWX);
			assert(m_iY*m_iWX+x < m_iWX*m_iWY);
			return (*_arr)[m_iY*m_iWX+x];
		}
	};
	
	template<typename Type, bool OnGPU, typename Enable = void>
	class F2DArrayBase;
	
	template<typename Type, bool OnGPU>
	class F2DArrayBase<Type, OnGPU, typename std::enable_if<!OnGPU>::type> {
	public:
		size_t m_iX = 0;	// nr. of neurons in layer m_iY
		size_t m_iY = 0;	// nr. of layer in net
		vector_t<Type> m_data;
		
		F2DArrayBase() {}
		
		F2DArrayBase(const size_t x, const size_t y) : m_iX(x), m_iY(y) {
			this->m_data.resize(x * y);
		}
		
		// copy ctor from device to host vector
		F2DArrayBase(const F2DArrayBase<Type, true> &mat) {
			this->m_iX = mat.m_iX;
			this->m_iY = mat.m_iY;
			this->m_data.resize(m_iX * m_iY);

			for(unsigned int y = 0; y < m_iY; y++)
			for(unsigned int x = 0; x < m_iX; x++) {
				this->m_data[y*m_iX+x] = mat.m_data[y*m_iX+x];
			}
		}
		
		vector_t<Type> GetSubArrayX(const size_t &iY) const {
			assert(iY < this->m_iY);
			
			vector_t<Type> row(m_iX);
			for(size_t x = 0; x < this->m_iX; x++) {
				row[x] = this->m_data[iY*this->m_iX+x];
			}
			return row;
		}

		vector_t<Type> GetSubArrayY(const size_t &iX) const {
			assert(iX < this->m_iX);

			vector_t<Type> col(m_iY);
			for(size_t y = 0; y < this->m_iY; y++) {
				col[y] = this->m_data[y*this->m_iX+iX];
			}
			return col;
		}
	};
	
#ifdef __CUDACC__
	// my favourite type :D
	template<typename Type, bool OnGPU>
	class F2DArrayBase<Type, OnGPU, typename std::enable_if<OnGPU>::type> {
	public:
		size_t m_iX = 0;	// nr. of neurons in layer m_iY
		size_t m_iY = 0;	// nr. of layer in net
		dev_vector_t<Type> m_data;

		F2DArrayBase() {}
		
		F2DArrayBase(const size_t x, const size_t y) : m_iX(x), m_iY(y) {
			this->m_data.resize(x * y);
		}
		
		// copy ctor from host to device vector
		F2DArrayBase(const F2DArrayBase<Type, false> &mat) {
			this->m_iX = mat.m_iX;
			this->m_iY = mat.m_iY;
			this->m_data.resize(m_iX * m_iY);

			for(unsigned int y = 0; y < m_iY; y++)
			for(unsigned int x = 0; x < m_iX; x++) {
				this->m_data[y*m_iX+x] = mat.m_data[y*m_iX+x];
			}
		}
		
		dev_vector_t<Type> GetSubArrayX(const size_t &iY) const {
			assert(iY < this->m_iY);
			
			dev_vector_t<Type> row(m_iX);
			for(size_t x = 0; x < this->m_iX; x++) {
				row[x] = this->m_data[iY*this->m_iX+x];
			}
			return row;
		}

		dev_vector_t<Type> GetSubArrayY(const size_t &iX) const {
			assert(iX < this->m_iX);

			dev_vector_t<Type> col(m_iY);
			for(size_t y = 0; y < this->m_iY; y++) {
				col[y] = this->m_data[y*this->m_iX+iX];
			}
			return col;
		}

		iterator<Type> GetRowBegin(const unsigned int &y) {
			assert(y < m_iY);
			return this->m_data.begin()+y*m_iX;
		}

		iterator<Type> GetRowEnd(const unsigned int &y) {
			assert(y < m_iY);
			return this->m_data.begin()+y*m_iX+m_iX;
		}

		const_iterator<Type> GetRowBegin(const unsigned int &y) const {
			assert(y < m_iY);
			return this->m_data.begin()+y*m_iX;
		}

		const_iterator<Type> GetRowEnd(const unsigned int &y) const {
			assert(y < m_iY);
			return this->m_data.begin()+y*m_iX+m_iX;
		}
	};
#endif

	template <class Type, bool OnGPU = false>
	class F2DArray : public F2DArrayBase<Type, OnGPU> {		
	public:
		using hidden::F2DArrayBase<Type, OnGPU>::F2DArrayBase;
		virtual ~F2DArray() {}

		Type &at(const size_t id) {
			return this->m_data.at(id);
		}
		const Type &at(const size_t id) const {
			return this->m_data.at(id);
		}
		
		void GetOutput();
		
		size_t GetW() const { return this->m_iX; }
		size_t GetH() const { return this->m_iY; }
		size_t GetTotalSize() const { return this->m_data.size(); }

		void SetValue(const size_t &x, const size_t &y, const Type &fVal);
		Type GetValue(const size_t &x, const size_t &y) const;

		F1DArray<Type, vector_t<Type> *> operator[] (size_t y) {
			return F1DArray<Type, vector_t<Type> *>(&(this->m_data), this->m_iX, this->m_iY, y);
		}

		const F1DArray<Type, const vector_t<Type> *> operator[] (size_t y) const {
			return F1DArray<Type, const vector_t<Type> *>(&(this->m_data), this->m_iX, this->m_iY, y);
		}

	#ifdef __F2DArray_ADDON
		#include __F2DArray_ADDON
	#endif
	};

	#include "2DArray.tpp"
};


namespace ANN {
	template<class T> using F2DArray = hidden::F2DArray<T, false>;
}
namespace ANNGPGPU {
	template<class T> using F2DArray = hidden::F2DArray<T, true>;
}


