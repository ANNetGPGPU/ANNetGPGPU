%{
#include "containers/2DArray.h"
%}

%ignore ANN::F2DArray::operator [];

%inline %{
	template <class T>
	struct AN2DRow {
		ANN::F2DArray<T> *g;
		int    y;

		// These functions are used by Python to access sequence types (lists, tuples, ...)
		T __getitem__(int x) {
			return g->GetValue(x, y);
		}

		void __setitem__(int x, T val) {
			g->SetValue(x, y, val);
		}
	};
%}

%include "containers/2DArray.h"  

%extend ANN::F2DArray<float> {
	AN2DRow<float> __getitem__(int y) {
		AN2DRow<float> r;
		r.g = self;
		r.y = y;
		return r;
	}
};
