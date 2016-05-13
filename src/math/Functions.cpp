//#include <iostream>
#include "Functions.h"

using namespace ANN;

/*
 * BP
 */
TransfFunction Functions::fcn_tanh = {
	(char*)"tanh",
	fcn_tanh_normal,
	fcn_tanh_derivate
};

TransfFunction Functions::fcn_log = {
	(char*)"log",
	fcn_log_normal,
	fcn_log_derivate
};

TransfFunction Functions::fcn_linear = {
	(char*)"linear",
	fcn_linear_normal,
	fcn_linear_derivate
};

TransfFunction Functions::fcn_binary = {
	(char*)"binary",
	fcn_binary_normal,
	fcn_binary_derivate
};


