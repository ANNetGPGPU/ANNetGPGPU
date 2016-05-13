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

#ifndef PLAINNETARRAY_H_
#define PLAINNETARRAY_H_

#ifndef SWIG
#include <vector>
#endif 

namespace ANN {

class F2DArray;

/**
 * \brief Pseudo 3D-array.
 * @author Daniel "dgrat" Frenzel
 */
class F3DArray {
	friend class F2DArray;

private:
	unsigned int m_iX;	// nr. of neuron in layer m_iY
	unsigned int m_iY;	// nr. of layer in net
	unsigned int m_iZ;	// nr. of axon/weight of neuron m:iX in layer m_iY
	float *m_pArray;

public:
	// Standard C++ "conventions"
	F3DArray();
	F3DArray(const unsigned int &iSizeX, const unsigned int &iSizeY, const unsigned int &iSizeZ, const float &fVal);
	F3DArray(const unsigned int &iSizeX, const unsigned int &iSizeY, const unsigned int &iSizeZ, float *pArray);
	virtual ~F3DArray();

	void Alloc(const unsigned int &iX, const unsigned int &iY, const unsigned int &iZ);

	unsigned int GetW() const;	// X
	unsigned int GetH() const;	// Y
	unsigned int GetD() const;	// Z

	unsigned int GetTotalSize() const; 	// X*Y*Z

	void SetSubArrayYZ(const unsigned int &iX, const F2DArray &mYZ);
	void SetSubArrayXZ(const unsigned int &iY, const F2DArray &mXZ);
	void SetSubArrayXY(const unsigned int &iZ, const F2DArray &mXY);
	
	/* return a pointer to the subarray at: Y,X */
	F2DArray GetSubArrayYZ(const unsigned int &iX) const;
	F2DArray GetSubArrayXZ(const unsigned int &iY) const;
	F2DArray GetSubArrayXY(const unsigned int &iZ) const;

	void SetValue(const int &iX, const int &iY, const int &iZ, const float &fVal);
	float GetValue(const int &iX, const int &iY, const int &iZ) const;

//OPERATORS
	operator float*();
	F2DArray operator[] (const int &iX) const;
};

}

#endif /* PLAINNETARRAY_H_ */
