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

