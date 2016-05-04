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

/*
 * SOM
 */
DistFunction Functions::fcn_bubble = {
	(char*)"bubble",
	fcn_bubble_nhood,
	fcn_rad_decay,
	fcn_lrate_decay
};

DistFunction Functions::fcn_gaussian = {
	(char*)"gaussian",
	fcn_gaussian_nhood,
	fcn_rad_decay,
	fcn_lrate_decay
};

DistFunction Functions::fcn_cutgaussian = {
	(char*)"cutgaussian",
	fcn_cutgaussian_nhood,
	fcn_rad_decay,
	fcn_lrate_decay
};

DistFunction Functions::fcn_epanechicov = {
	(char*)"epanechicov",
	fcn_epanechicov_nhood,
	fcn_rad_decay,
	fcn_lrate_decay
};

DistFunction Functions::fcn_mexican = {
	(char*)"mexican",
	fcn_mexican_nhood,
	fcn_rad_decay,
	fcn_lrate_decay
};

TransfFunction* Functions::ResolveTransfFByName (const char *name) {
	if (strcmp (name, "tanh") == 0) {
		//std::cout<<"fcn_tanh"<<std::endl;
		return (&fcn_tanh);
	}
	if (strcmp (name, "log") == 0) {
		//std::cout<<"fcn_log"<<std::endl;
		return (&fcn_log);
	}
	if (strcmp (name, "linear") == 0) {
		//std::cout<<"fcn_linear"<<std::endl;
		return (&fcn_linear);
	}
	if (strcmp (name, "binary") == 0) {
		//std::cout<<"fcn_binary"<<std::endl;
		return (&fcn_binary);
	}
	//std::cout<<"NULL"<<std::endl;
	return (NULL);
}

DistFunction* Functions::ResolveDistFByName (const char *name) {
	if (strcmp (name, "gaussian") == 0) {
		//std::cout<<"fcn_gaussian"<<std::endl;
		return (&fcn_gaussian);
	}
	if (strcmp (name, "mexican") == 0) {
		//std::cout<<"fcn_mexican"<<std::endl;
		return (&fcn_mexican);
	}
	if (strcmp (name, "bubble") == 0) {
		//std::cout<<"fcn_mexican"<<std::endl;
		return (&fcn_bubble);
	}
	if (strcmp (name, "cutgaussian") == 0) {
		//std::cout<<"fcn_mexican"<<std::endl;
		return (&fcn_cutgaussian);
	}
	if (strcmp (name, "epanechicov") == 0) {
		//std::cout<<"fcn_mexican"<<std::endl;
		return (&fcn_epanechicov);
	}
	return (NULL);
}

