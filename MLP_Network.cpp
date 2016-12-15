#include "MLP_Network.h"

void MLP_Network::Allocate(int nInputUnit,   int nHiddenUnit, int nOutputUnit, int nHiddenLayer,
                           int nTrainingSet)
{
    this->nTrainingSet  = nTrainingSet;
    this->nInputUnit    = nInputUnit;
    this->nHiddenUnit   = nHiddenUnit;
    this->nOutputUnit   = nOutputUnit;
    this->nHiddenLayer  = nHiddenLayer;
    
    layerNetwork = new MLP_Layer[nHiddenLayer+1]();
    
    layerNetwork[0].Allocate(nInputUnit, nHiddenUnit);
    for (int i = 1; i < nHiddenLayer; i++)
    {
        layerNetwork[i].Allocate(nHiddenUnit, nHiddenUnit);
    }
    layerNetwork[nHiddenLayer].Allocate(nHiddenUnit, nOutputUnit);
}

void MLP_Network::Delete()
{
    for (int i = 0; i < nHiddenLayer+1; i++)
    {
        layerNetwork[i].Delete();
    }
}

void MLP_Network::ForwardPropagateNetwork(float* inputNetwork)
{
    float* outputOfHiddenLayer=NULL;
    
    outputOfHiddenLayer=layerNetwork[0].ForwardPropagate(inputNetwork);
    for (int i=1; i < nHiddenLayer ; i++)
    {
        outputOfHiddenLayer=layerNetwork[i].ForwardPropagate(outputOfHiddenLayer);                  //hidden forward
    }
    layerNetwork[nHiddenLayer].ForwardPropagate(outputOfHiddenLayer);      // output forward
}

void MLP_Network::BackwardPropagateNetwork(float* desiredOutput)
{
    layerNetwork[nHiddenLayer].BackwardPropagateOutputLayer(desiredOutput);  // back_propa_output
    for (int i= nHiddenLayer-1; i >= 0  ; i--)
        layerNetwork[i].BackwardPropagateHiddenLayer(&layerNetwork[i+1]);              // back_propa_hidden
}

void MLP_Network::UpdateWeight(float learningRate)
{
        for (int i = 0; i < nHiddenLayer; i++)
            layerNetwork[i].UpdateWeight(learningRate);
        
        layerNetwork[nHiddenLayer].UpdateWeight(learningRate);
}

float MLP_Network::CostFunction(float* inputNetwork, float* desiredOutput)
{
    float *outputNetwork = layerNetwork[nHiddenLayer].GetOutput();
    float err=0.F;
    for (int j = 0; j < nOutputUnit; ++j)
        err += (desiredOutput[j] - outputNetwork[j])*(desiredOutput[j] - outputNetwork[j]);
    
    err /= 2;
        
    return err;
}

 
float MLP_Network::CalculateResult(float* inputNetwork,float* desiredOutput)
{
    int maxIdx = 0;
    
    maxIdx = layerNetwork[nHiddenLayer].GetMaxOutputIndex();
    
    if(desiredOutput[maxIdx] == 1.0f)
        return 1;
    return 0;
}
