#ifndef DBN_H
#define DBN_H
#include "RBM.h"
#include <iostream>
#include <ctime>
#include <cmath>
#include <vector>
#include <fstream>
#include <cstdlib>
#include <string>
#include "MLP_Network.h"
#include "MLP_Layer.h"



class DBN: public MLP_Network
{
    
private:
    RBM* rbmNetwork;
    
    float** input_DBN;
    
    int nTrainingSet;
    int nInputUnit;
    int nHiddenUnit;
    int nOutputUnit;
    int nHiddenLayer;
    int nMiniBatch;
    
public:
    DBN(float** input,int nTrainingSet,int nInputUnit,int nHiddenUnit,  int nHiddenLayer, int nMiniBatch, int num_output_size, float** desired_outputs);
    ~DBN();
    void Allocate_DBN();
    void Deallocate_DBN();
    
    void Set_Input_In_Range();
    
    int Sample_Binary_State_DBN(float probability);
    void Pretrain(int cd_k,float lr_RBM, int DBN_EPOCH);
    void Finetune();
    void Print_Result(float** test,int nTestSet);
};

#endif
