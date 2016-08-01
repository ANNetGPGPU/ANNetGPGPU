%{
#include "containers/ConTable.h"
%}

%include "containers/ConTable.h"   

namespace ANN {
	%template(ConDescrF) ConDescr<float>;
	%template(NeurDescrF) NeurDescr<float>;
	%template(ConTableF) ConTable<float>;
}
