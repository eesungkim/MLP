//
//  DBN.cpp
//  RBM
//
//  Created by Eesung Kim on 8/10/15.
//  Copyright (c) 2015 Eesung Kim. All rights reserved.
//

#include "DBN.h"
#include "MLP_Network.h"


DBN::DBN(float** input,int nTrainingSet,int nInputUnit,int nHiddenUnit,  int nHiddenLayer, int nMiniBatch, int nOutputUnit, float** desiredOutput) : MLP_Network(input,nTrainingSet,nInputUnit,nHiddenUnit,nHiddenLayer,nMiniBatch, nOutputUnit, desiredOutput){
    
    this->nTrainingSet  = nTrainingSet;
    this->nInputUnit    = nInputUnit;
    this->nHiddenUnit   = nHiddenUnit;
    this->nOutputUnit   = nOutputUnit;
    this->nHiddenLayer  = nHiddenLayer;   //2단계
    this->nMiniBatch    = nMiniBatch;
    
    Allocate_DBN();
    
    for (int i=0; i < nTrainingSet; i++)
        for (int j=0; j < nInputUnit; j++)
            this->input_DBN[i][j]=input[i][j];
    
    Set_Input_In_Range();
}

void DBN::Set_Input_In_Range()          // 각 변수마다 0~ 1 사이로 input값 설정
{
    for(int k=0 ; k < nInputUnit ; k++)
    {
        float max=0.0001,min=10000, range=0;
        for(int i=0 ; i < nTrainingSet ; i++)
        {
            if(max < input_DBN[i][k]) //만약 max가 num[i]보다 작으면 max는num[i]의 값이 된다.
                max = input_DBN[i][k];
        
            if(min > input_DBN[i][k]) //생략
                min = input_DBN[i][k];
        }
        range = max-min;
        for(int j=0 ; j < nTrainingSet ; j++)
            input_DBN[j][k]= input_DBN[j][k]/range;
    }
}

int DBN::Sample_Binary_State_DBN(float probability)
{
    if(probability < 0 || probability > 1)
        return 0;
    else
    {
        int c = 0;
        float num_random;
        
        num_random = (float)rand() / (RAND_MAX + 1.0);
        if (num_random < probability)
            c=1;
        return c;
    }
}

void DBN::Allocate_DBN()
{
    this->input_DBN = new float*[nTrainingSet];
    for (int i=0; i<nTrainingSet; i++)
        this->input_DBN[i] = new float[nInputUnit];
    
    rbmNetwork = new RBM[nHiddenLayer+1]();
    
    rbmNetwork[0].Allocate_RBM(nInputUnit, nHiddenUnit);
    for (int i = 1; i < nHiddenLayer; i++)
    {
        rbmNetwork[i].Allocate_RBM(nHiddenUnit, nHiddenUnit);
    }
    rbmNetwork[nHiddenLayer].Allocate_RBM(nHiddenUnit, nOutputUnit);
}

DBN::~DBN()
{
    Deallocate_DBN();
}

void DBN::Deallocate_DBN()
{
    for (int i=0; i<nTrainingSet; i++) {
            delete input_DBN[i];
    }
    delete [] input_DBN;
    input_DBN=NULL;
}

void DBN::Pretrain(int cd_k, float lr_RBM, int DBN_EPOCH)
{
    float *layer_input = NULL;
    float *prev_layer_input;
    int prev_layer_input_size;
    //cout<<"pretraing...."<<endl;
    
    //Start clock
    clock_t start, finish;
    double elapsed_time;
    start = clock();
    
    
    
    float *train_X = new float[nInputUnit];

    for (int i=0; i < nHiddenLayer; i++)
    {
        for( int epoch =0 ; epoch < DBN_EPOCH ; epoch++)
        {
            /*--------------------------------------*/
            double percentage= epoch/(double)DBN_EPOCH*100;
            cout<<"DBN "<<i+1<<"/"<<nHiddenLayer<<" | "<<percentage<<" %"<<endl;
            if (percentage == 0 ||percentage == 20 || percentage == 40 || percentage == 60 || percentage == 80)
            {
                clock_t check = clock();
                elapsed_time = (double)(check-start)/CLOCKS_PER_SEC;
                cout<<"pretraining.... [ "<<percentage<<" % ] / [ "<<elapsed_time/60<<" ] min"<<endl;
            }
            /*--------------------------------------*/
            
    
            int batchCount=0;
            for (int m=0; m < nTrainingSet; m++)
            {
                for(int n=0 ; n <nInputUnit ; n++)
                    train_X[n] = input_DBN[m][n];  // 랜덤하게 0~1로 샘플링
    
                for(int l=0 ; l <= i ; l++)
                {
                    if(l == 0)
                    {
                        layer_input = new float[nInputUnit];
                        for (int j=0; j< nInputUnit; j++)
                            layer_input[j]  = train_X[j];
                    }
                    else
                    {
                        if(l == 1)
                            prev_layer_input_size = nInputUnit;
                        else
                            prev_layer_input_size = nHiddenUnit;
                        
                        prev_layer_input = new float[prev_layer_input_size];
                        
                        for(int j=0; j<prev_layer_input_size; j++)
                            prev_layer_input[j] = layer_input[j];
                        
                        delete[] layer_input;
                        
                        layer_input = new float[nHiddenUnit];
                        
                        rbmNetwork[l-1].Positive_Phase_DBN(prev_layer_input, layer_input);
                        delete[] prev_layer_input;
                    }
                }
                batchCount++;
                
                rbmNetwork[i].Contrastive_Divergence(layer_input, cd_k);
                
                rbmNetwork[i].Update_Weight(lr_RBM);
                
                /*
                rbmNetwork[i].Update_Weight_Batch(lr_RBM);
                
                if ( nMiniBatch == batchCount)
                {
                    rbmNetwork[i].Update_Weight(lr_RBM);
                    batchCount=0;
                }
                 */
            }
        }
    }
    
    //Finish clock
    
    finish = clock();
    elapsed_time = (double)(finish-start)/CLOCKS_PER_SEC;
    cout<<"pretraining time: "<<elapsed_time/60<<" min"<<endl;
    delete[] layer_input;
    delete[] train_X;

}

void DBN::Finetune()
{
    for (int i=0; i < nHiddenLayer; i++)
    {
        layerNetwork[i].Set_Weight_MLP(rbmNetwork[i].Get_Weight());
        layerNetwork[i].Set_Bias_Weight_MLP(rbmNetwork[i].Get_Bias_Weight());
    }
}

