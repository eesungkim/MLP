
#include "MLP_Layer.h"


void MLP_Layer::Allocate(int previous_num, int current_num)
{
    this->nPreviousUnit   =  previous_num;
    this->nCurrentUnit    =  current_num;
    
    weight          = new float[nPreviousUnit * nCurrentUnit];
    gradient       = new float[nPreviousUnit * nCurrentUnit];
    inputLayer     = new float[nPreviousUnit];
    outputLayer    = new float[nCurrentUnit];
    delta          = new float[nCurrentUnit];
    biasWeight    = new float[nCurrentUnit]; 
    biasGradient  = new float[nCurrentUnit];
    
    srand((unsigned)time(NULL));
    for (int j = 0; j < nCurrentUnit; j++)
    {
        outputLayer[j]=0.0;
        delta[j]=0.0;
        for (int i = 0; i < nPreviousUnit; i++)
        {
            weight[j*nPreviousUnit+i]   = 0.2 * rand() / RAND_MAX - 0.1;
            gradient[j*nPreviousUnit+i]= 0.0;
        }
        biasWeight[j] = 0.2 * rand() / RAND_MAX - 0.1;                             
        biasGradient[j] = 0;
    }
}



void MLP_Layer::Delete(){

    delete [] weight;
    delete [] gradient;
    delete [] delta;
    delete [] outputLayer;
    delete [] biasGradient;
    delete [] biasWeight;
}

float* MLP_Layer::ForwardPropagate(float* inputLayers)      // f( sigma(weights * inputs) + bias )
{
    this->inputLayer=inputLayers;
    for(int j = 0 ; j < nCurrentUnit ; j++)
    {
        float net= 0;
        for(int i = 0 ; i < nPreviousUnit ; i++)
        {
            net += inputLayer[i] * weight[j*nPreviousUnit+i];
        }
        net+=biasWeight[j];
        
        outputLayer[j] = ActivationFunction(net);
    }
    return outputLayer;
}

void MLP_Layer::BackwardPropagateOutputLayer(float* desiredValues)
{
    for (int k = 0; k < nCurrentUnit; k++){
        float fnet_derivative = outputLayer[k] * (1 - outputLayer[k]);
        delta[k] = fnet_derivative * (desiredValues[k] - outputLayer[k]);
        //delta[k] = DerivativeActivation(k) * (desiredValues[k] - outputLayer[k]);
    }
    
    for (int k = 0 ; k < nCurrentUnit ; k++)
        for (int j = 0 ; j < nPreviousUnit; j++)
            gradient[k*nPreviousUnit + j] += - (delta[k] * inputLayer[j]);
    
    for (int k = 0 ; k < nCurrentUnit   ; k++)
            biasGradient[k] += - delta[k] ;
    
    
}

void MLP_Layer::BackwardPropagateHiddenLayer(MLP_Layer* previousLayer)
{
    
    float* previousLayer_weight = previousLayer->GetWeight();
    float* previousLayer_delta = previousLayer->GetDelta();
    int previousLayer_node_num = previousLayer->GetNumCurrent();

    for (int j = 0; j < nCurrentUnit; j++)
    {
        float previous_sum=0;
        for (int k = 0; k < previousLayer_node_num; k++)
        {
            previous_sum += previousLayer_delta[k] * previousLayer_weight[k*nCurrentUnit + j];
        }
        delta[j] =  outputLayer[j] * (1 - outputLayer[j])* previous_sum;
        //delta[j] =  DerivativeActivation(j)* previous_sum;
    }
    
    for (int j = 0; j < nCurrentUnit; j++)
        for (int i = 0; i < nPreviousUnit ; i++)
            gradient[j*nPreviousUnit + i] +=  -delta[j] * inputLayer[i];
    
    for (int j = 0 ; j < nCurrentUnit   ; j++)
        biasGradient[j] += -delta[j] ;
}

void MLP_Layer::UpdateWeight(float learningRate)
{
    for (int j = 0; j < nCurrentUnit; j++)
        for (int i = 0; i < nPreviousUnit; i++)
            weight[j*nPreviousUnit + i] +=  -learningRate *gradient[j*nPreviousUnit + i];
    
    for (int j = 0; j < nCurrentUnit; j++)
        biasWeight[j] += -biasGradient[j];
    
    for (int j = 0; j < nCurrentUnit; j++)           
        for (int i = 0; i < nPreviousUnit; i++)
            gradient[j*nPreviousUnit + i] = 0;
    
    for (int j = 0; j < nCurrentUnit; j++)
        biasGradient[j]=0;
}


int MLP_Layer::GetMaxOutputIndex()
{
    int maxIdx = 0;
    for(int o = 1; o < nCurrentUnit; o++){
        if(outputLayer[o] > outputLayer[maxIdx])
            maxIdx = o;
    }
    
    return maxIdx;
}


