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

#ifndef TRAININGDATA_H_
#define TRAININGDATA_H_

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

class TrainingSet {
private:
	std::vector<std::vector<float> > m_vInputList;
	std::vector<std::vector<float> > m_vOutputList;

public:
	TrainingSet();
	~TrainingSet();

	void AddInput(const std::vector<float> &vIn);
	void AddOutput(const std::vector<float> &vOut);
	void AddInput(float *pIn, const unsigned int &iSize);
	void AddOutput(float *pOut, const unsigned int &iSize);

	unsigned int GetNrElements() const;

	std::vector<float> GetInput(const unsigned int &iID) const;
	std::vector<float> GetOutput(const unsigned int &iID) const;

	void Clear();

	void ExpToFS(BZFILE* bz2out, int iBZ2Error);
	void ImpFromFS(BZFILE* bz2in, int iBZ2Error);
};

}

#endif /* TRAININGDATA_H_ */
