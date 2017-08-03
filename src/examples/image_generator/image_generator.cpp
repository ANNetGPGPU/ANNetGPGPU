/*
 * SOM trained on pixels in OpenCV Image
 */

#include <iostream>
#include <iomanip> // manipulators
#include <vector> // For dynamic arrays.
#include <ANNet>
#include <ANGPGPU>
#include <ANContainers>
#include <ANMath>

#include <chrono>

using namespace std;
using namespace std::chrono;

#include "opencv2/highgui.hpp"
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"

using namespace cv;

// Convert Mat to input data
ANN::TrainingSet<float> mat2Input(cv::Mat inputImage) {
    ANN::TrainingSet<float> input;
    for (int y = 0; y < inputImage.rows; y++) {

        // Loop over columns
        cv::Vec3b* pixel = inputImage.ptr<cv::Vec3b>(y); // point to first pixel in row
        for (int x = 0; x < inputImage.cols; x++) {

            // Loop over dims
            // Assumes 3 channel BGR image.
            vector<float> thisVector;
            for (int k=0; k < 3; k++) {
                float value = int(pixel[x][k])/255.0; // Assumes 8bpp image.
                // Append this value to a training vector;
                thisVector.push_back(value);
            }

            // Add the training vector to the training set.
            input.AddInput(thisVector);
        }
    }

    return input;
}

// Convert the SOM weights to a Mat.
void getWeights(ANNGPGPU::SOMNetGPU<float, ANN::functor_bubble<float>> *network, cv::Mat outputImage) {
    for(int neuron = 0; neuron <outputImage.cols*outputImage.rows; neuron++) {
        // Convert neuron index to x/y position.
        int x = floor(neuron/outputImage.cols); // Fills rows first
        int y = neuron%outputImage.cols; // Cols (remainder)

        ANN::SOMNeuron<float> *pNeur = (ANN::SOMNeuron<float>*)((ANN::SOMLayer<float>*)network->GetOPLayer())->GetNeuron(neuron);
        uchar B = int(pNeur->GetConI(0)->GetValue()*255);
        uchar G = int(pNeur->GetConI(1)->GetValue()*255);
        uchar R = int(pNeur->GetConI(2)->GetValue()*255);
        outputImage.at<cv::Vec3b>(x,y) = cv::Vec3b(B,G,R);
    }
}

int main(int argc, char* argv[]) {
    if (argc < 3) { // We expect 3 arguments: the program name, the source path and the destination path
        std::cerr << "Usage: " << argv[0] << " <input image> <output image>" << std::endl;
        return 1;
    }
    std::string source = "";
    std::string destination = "";
    for (int i = 1; i < argc; ++i) { // Remember argv[0] is the path to the program, we want from argv[1] onwards
        if (i + 1 < argc) {
            source = argv[i];
        }
        else {
            destination = argv[i];
        }
    }
    
    std::cout << "Started program with input image " << source << ", output image: " << destination << std::endl;
    
    // Load the image
    Mat myImage = imread(source);

    // Make sure image is loaded
    if( !myImage.data ) {
        cout << "Could not load image." << endl;
        return 1;
    }

    // Create SOM
    ANNGPGPU::SOMNetGPU<float, ANN::functor_bubble<float>> network;
    network.CreateSOM(3, 1, myImage.cols, myImage.rows); // numInputs, numOutputs, width, height

    // Use image to populate training set.
    ANN::TrainingSet<float> trainingData;
    trainingData = mat2Input(myImage);

    // Set network parameters
    network.SetTrainingSet(trainingData);
    network.SetLearningRate(1.0f);
    network.SetSigma0(myImage.cols/2);

    high_resolution_clock::time_point t1 = high_resolution_clock::now();
    // Train the network on image
    network.Training(5000, ANN::ANRandomMode); // samples are randomly ordered.
    high_resolution_clock::time_point t2 = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>( t2 - t1 ).count();
    std::cout << "time passed: " << duration << std::endl;
    
    // Convert SOM weights to Mat
    Mat outputImage = cv::Mat(myImage.rows, myImage.cols, myImage.type(), cv::Scalar::all(0)); // Create output image
    getWeights(&network, outputImage);

    // Save output file.
    imwrite(destination, outputImage);

    return 0;
}
