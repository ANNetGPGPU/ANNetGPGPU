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
#include <utility>
#include <vector>
#include <bzlib.h>
#endif

namespace ANN {

/**
 * \brief Storage of simple input/output samples usable for training.
 *
 * Data must get converted to simple float arrays to get used with this storage format.
 *
 * @author Daniel "dgrat" Frenzel
 */
template <class Type>
class TrainingSet {
private:
	std::vector<std::vector<Type> > m_vInputList;
	std::vector<std::vector<Type> > m_vOutputList;

public:
	TrainingSet();
	~TrainingSet();

	void AddInput(const std::vector<Type> &vIn);
	void AddOutput(const std::vector<Type> &vOut);
	void AddInput(Type *pIn, const unsigned int &iSize);
	void AddOutput(Type *pOut, const unsigned int &iSize);

	unsigned int GetNrElements() const;

	std::vector<Type> GetInput(const unsigned int &iID) const;
	std::vector<Type> GetOutput(const unsigned int &iID) const;

	void Clear();

	void ExpToFS(BZFILE* bz2out, int iBZ2Error);
	void ImpFromFS(BZFILE* bz2in, int iBZ2Error);
	
#ifdef __TrainingSet_ADDON
	#include __TrainingSet_ADDON
#endif
};

#include "TrainingSet.tpp"

}

