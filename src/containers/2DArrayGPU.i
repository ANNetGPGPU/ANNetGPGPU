%{
#include "containers/2DArrayGPU.h"
%}

%inline %{
	template <class T>
	struct AN2DRow<T> {
		ANNGPGPU::F2DArray<T> *g;
		int y;

		// These functions are used by Python to access sequence types (lists, tuples, ...)
		T __getitem__(int x) {
			return g->GetValue(x, y);
		}

		void __setitem__(int x, T val) {
			g->SetValue(x, y, val);
		}
	};
%}

%include "containers/2DArrayGPU.h"  

%extend ANNGPGPU::F2DArray<float> {
	AN2DRow<float> __getitem__(int y) {
		AN2DRow<float> r;
		r.g = self;
		r.y = y;
		return r;
	}
};
