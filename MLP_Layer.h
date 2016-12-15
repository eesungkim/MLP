#ifndef MLP_Layer_H
#define MLP_Layer_H

#include <iostream>
#include <ctime>
#include <cmath>
#include <vector>
#include <fstream>

class  MLP_Layer {
    int nPreviousUnit;
    int nCurrentUnit;
    
    float* inputLayer;
    float* outputLayer;
    float* weight;
    float* gradient;
    float* delta;
    
    float* biasWeight;    
    float* biasGradient;
    
public:
    MLP_Layer(){};
    ~MLP_Layer()    {   Delete();   }
    
    void Allocate(int previous_node_num, int nCurrentUnit);
    void Delete();
    
    float* ForwardPropagate(float* inputLayer);
    void BackwardPropagateHiddenLayer(MLP_Layer* previousLayer);
    void BackwardPropagateOutputLayer(float* desiredValues);
    
    void UpdateWeight(float learningRate);
    
	float* GetOutput()  {   return outputLayer; }
    float* GetWeight()  {   return weight;      }
    float* GetDelta()   {   return delta;       }
    int GetNumCurrent() {   return nCurrentUnit;}
	int GetMaxOutputIndex();
    // Sigmoid
    float ActivationFunction(float net)		{ return 1.F/(1.F + (float)exp(-net)); }
    //float DerivativeActivation(int preNode){return (1 - outputLayer[preNode]) *outputLayer[preNode]; }
    
    // ReLU
    //float ActivationFunction(float net)  {if (net <= 0) net=0; return net;}
    //float DerivativeActivation(int preNode){ if(outputLayer[preNode] <= 0) return 0.01; else return 1;}
    
    
    float DerActivationFromOutput(float output){ return output * (1.F-output); }
    float DerActivation(float net)	{ return DerActivationFromOutput(ActivationFunction(net)); }
    
    
    
};

#endif