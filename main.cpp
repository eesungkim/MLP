#include "MLP_Network.h"
#include "MLP_Layer.h"
#include "MNIST.h"



int main()
{
    
    int nInputUnit      = 784;
    int nHiddenUnit     = 512;
    int nOutputUnit     = 10;
    
    int nHiddenLayer    = 1;
    int nMiniBatch      = 10;
    float learningRate     = 0.1;
    
    int nTrainingSet    = 60000;
    int nTestSet        = 10000;
    
    float errMinimum = 0.01;    
    int maxEpoch = 1000;
    
    //Allocate
    float **inputTraining			= new float*[nTrainingSet];
    float **desiredOutputTraining	= new float*[nTrainingSet];
    
    for(int i = 0;i < nTrainingSet;i++){
        inputTraining[i]			= new float[nInputUnit];
        desiredOutputTraining[i]	= new float[nOutputUnit];
    }
    float **inputTest			= new float*[nTestSet];
    float **desiredOutputTest	= new float*[nTestSet];
    
    for(int i = 0;i < nTestSet;i++){
        inputTest[i]			= new float[nInputUnit];
        desiredOutputTest[i]	= new float[nOutputUnit];
    }
    


    
    //MNIST Input Array Allocation and Initialization
    MNIST mnist;
    mnist.ReadMNIST_Input("/Users/MLP_160713/train-images-idx3-ubyte", nTrainingSet, inputTraining);
    mnist.ReadMNIST_Label("/Users/MLP_160713/train-labels-idx1-ubyte",nTrainingSet, desiredOutputTraining);
    
    mnist.ReadMNIST_Input("/Users/MLP_160713/t10k-images-idx3-ubyte",nTestSet, inputTest);
    mnist.ReadMNIST_Label("/Users/MLP_160713/t10k-labels-idx1-ubyte",nTestSet, desiredOutputTest);
    
    

    MLP_Network mlp;
    
    mlp.Allocate(nInputUnit,nHiddenUnit,nOutputUnit,nHiddenLayer,nTrainingSet);


    
    //Start clock
    clock_t start, finish;
    double elapsed_time;
    start = clock();
    
    
    float initialLR = learningRate;
   
    int epoch = 0;
    while (epoch < maxEpoch)
    {
        float sumError=0;
        int batchCount=0;
        for (int i = 0; i < nTrainingSet; i++)
        {
            mlp.ForwardPropagateNetwork(inputTraining[i]);

            mlp.BackwardPropagateNetwork( desiredOutputTraining[i]);
            
            sumError += mlp.CostFunction(inputTraining[i],desiredOutputTraining[i]);
            
            
            if( ((batchCount+1) % nMiniBatch) == 0)
            {
                mlp.UpdateWeight(learningRate);
                batchCount=0;
            }
            batchCount++;
        }
        
        sumError /= nTrainingSet;
        
        cout<<epoch<<" | "<<sumError<<" | "<<errMinimum<<endl;
        
        if (sumError < errMinimum)
            break;
        
        learningRate = initialLR/(1+epoch*learningRate);    // learning rate progressive decay
        ++epoch;
    }

    
    
    //Finish clock
    finish = clock();
    elapsed_time = (double)(finish-start)/CLOCKS_PER_SEC;
    cout<<"time: "<<elapsed_time<<" sec"<<endl;

    
    // Test Set Result
    cout<<"[Result]"<<endl<<endl;

    int sums=0;
    float accuracyRate=0.F;
    for (int i = 0; i < nTrainingSet; i++)
    {
        mlp.ForwardPropagateNetwork(inputTraining[i]);
        
        sums += mlp.CalculateResult(inputTraining[i],desiredOutputTraining[i]);
        
        
    }
    
    accuracyRate = (sums / (float)nTrainingSet) * 100;
    
    cout << "[Training Set]\t"<<"Accuracy Rate: " << accuracyRate << " %"<<endl;
    
    // Test Set Result
    sums=0;
    accuracyRate=0.F;
    for (int i = 0; i < nTestSet; i++)
    {
        mlp.ForwardPropagateNetwork(inputTest[i]);
        
        sums += mlp.CalculateResult(inputTest[i], desiredOutputTest[i]);
    }
    accuracyRate = (sums / (float)nTestSet) * 100;
    
    cout << "[Test Set]\t"<<"Accuracy Rate: " << accuracyRate << " %"<<endl;
    
    
    for (int i = 0; i < nTrainingSet; i++)
    {
        delete [] desiredOutputTraining[i];
        delete [] inputTraining[i];
        delete [] desiredOutputTest[i];
        delete [] inputTest[i];
    }

    delete[] inputTraining;
    delete[] desiredOutputTraining;
    delete[] inputTest;
    delete[] desiredOutputTest;
    
 
    return 0;
}
