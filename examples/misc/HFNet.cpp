/*
 * main.cpp
 *
 *  Created on: 12.04.2010
 *      Author: dgrat
 */

#include <ANNet>
#include <ANContainers>
#include <ANMath>

#include "Samples.h"

#include <ctime>
#include <iostream>


int main(int argc, char *argv[]) {
	ANN::TrainingSet input;
	input.AddInput(TR1, 16);
	input.AddInput(TR2, 16);
	input.AddInput(TR3, 16);

	ANN::HFNet net;
	net.Resize(16,1);
	net.SetTrainingSet(input);
	net.PropagateBW();

	net.SetInput(fInpHF1);
	for(int k = 0; k < 1; k++) {
		net.PropagateFW();

		for(int k = 0; k < net.GetOutput().size(); k++) {
			std::cout<<"outp: "<<net.GetOutput().at(k)<<std::endl;
		}
		std::cout<<std::endl;
	}
	net.SetInput(fInpHF2);
	for(int k = 0; k < 1; k++) {
		net.PropagateFW();

		for(int k = 0; k < net.GetOutput().size(); k++) {
			std::cout<<"outp: "<<net.GetOutput().at(k)<<std::endl;
		}
		std::cout<<std::endl;
	}
	net.SetInput(fInpHF3);
	for(int k = 0; k < 1; k++) {
		net.PropagateFW();

		for(int k = 0; k < net.GetOutput().size(); k++) {
			std::cout<<"outp: "<<net.GetOutput().at(k)<<std::endl;
		}
		std::cout<<std::endl;
	}

	net.ExpToFS("hf.foo.bar");
	net.ImpFromFS("hf.hftest");

	net.SetInput(fInpHF1);
	for(int k = 0; k < 1; k++) {
		net.PropagateFW();

		for(int k = 0; k < net.GetOutput().size(); k++) {
			std::cout<<"outp: "<<net.GetOutput().at(k)<<std::endl;
		}
		std::cout<<std::endl;
	}
	net.SetInput(fInpHF2);
	for(int k = 0; k < 1; k++) {
		net.PropagateFW();

		for(int k = 0; k < net.GetOutput().size(); k++) {
			std::cout<<"outp: "<<net.GetOutput().at(k)<<std::endl;
		}
		std::cout<<std::endl;
	}
	net.SetInput(fInpHF3);
	for(int k = 0; k < 1; k++) {
		net.PropagateFW();

		for(int k = 0; k < net.GetOutput().size(); k++) {
			std::cout<<"outp: "<<net.GetOutput().at(k)<<std::endl;
		}
		std::cout<<std::endl;
	}

	return 0;
}
