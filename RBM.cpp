//
//  RBM.cpp
//  RBM
//
//  Created by Eesung Kim on 8/7/15.
//  Copyright (c) 2015 Eesung Kim. All rights reserved.
//

#include "RBM.h"



void RBM::Allocate_RBM(int num_visible_unit, int num_hidden_unit)
{
    this->num_visible_unit   = num_visible_unit;
    this->num_hidden_unit    = num_hidden_unit;
    
    weight_RBM          = new float[num_visible_unit * num_hidden_unit];
    if (weight_RBM == NULL)
        cout <<"[RBM] Error: Memory Allocation [weight_RBM]"<<endl;
    else
        cout<<"[RBM] Success: Memory Allocation [weight_RBM]"<<endl;
    
    
    first_sample_h        = new float[num_hidden_unit];
    if (first_sample_h == NULL)
        cout <<"[RBM] Error: Memory Allocation [first_sample_h]"<<endl;
    else
        cout<<"[RBM] Success: Memory Allocation [first_sample_h]"<<endl;
    first_probability_h   = new float[num_hidden_unit];
    if (first_probability_h == NULL)
        cout <<"[RBM] Error: Memory Allocation [first_probability_h]"<<endl;
    else
        cout<<"[RBM] Success: Memory Allocation [first_probability_h]"<<endl;
    
    nth_sample_v          = new float[num_visible_unit];
    if (nth_sample_v == NULL)
        cout <<"[RBM] Error: Memory Allocation [nth_sample_v]"<<endl;
    else
        cout<<"[RBM] Success: Memory Allocation [nth_sample_v]"<<endl;
    nth_probability_v     = new float[num_visible_unit];
    if (nth_probability_v == NULL)
        cout <<"[RBM] Error: Memory Allocation [nth_probability_v]"<<endl;
    else
        cout<<"[RBM] Success: Memory Allocation [nth_probability_v]"<<endl;
    
    nth_sample_h          = new float[num_hidden_unit];
    if (nth_sample_h == NULL)
        cout <<"[RBM] Error: Memory Allocation [nth_sample_h]"<<endl;
    else
        cout<<"[RBM] Success: Memory Allocation [nth_sample_h]"<<endl;
    nth_probability_h     = new float[num_hidden_unit];
    if (nth_probability_h == NULL)
        cout <<"[RBM] Error: Memory Allocation [nth_probability_h]"<<endl;
    else
        cout<<"[RBM] Success: Memory Allocation [nth_probability_h]"<<endl;
    
    hBias_weight     = new float[num_hidden_unit];
    if (hBias_weight == NULL)
        cout <<"[RBM] Error: Memory Allocation [hBias_weight]"<<endl;
    else
        cout<<"[RBM] Success: Memory Allocation [hBias_weight]"<<endl;
    
    vBias_weight     = new float[num_visible_unit];
    if (vBias_weight == NULL)
        cout <<"[RBM] Error: Memory Allocation [vBias_weight]"<<endl;
    else
        cout<<"[RBM] Success: Memory Allocation [vBias_weight]"<<endl;
    
    input_RBM        = new float[num_visible_unit];
    if (input_RBM == NULL)
        cout <<"[RBM] Error: Memory Allocation [input_RBM]"<<endl;
    else
        cout<<"[RBM] Success: Memory Allocation [input_RBM]"<<endl;
    
    accum_weight = new float[num_visible_unit * num_hidden_unit];
    if (accum_weight == NULL)
        cout <<"[RBM] Error: Memory Allocation [accum_weight]"<<endl;
    else
        cout<<"[RBM] Success: Memory Allocation [accum_weight]"<<endl;
    
    accum_hBias_weight     = new float[num_hidden_unit];
    if (accum_hBias_weight == NULL)
        cout <<"[RBM] Error: Memory Allocation [accum_hBias_weight]"<<endl;
    else
        cout<<"[RBM] Success: Memory Allocation [accum_hBias_weight]"<<endl;
    
    accum_vBias_weight = new float[num_visible_unit];
    if (accum_vBias_weight == NULL)
        cout <<"[RBM] Error: Memory Allocation [accum_vBias_weight]"<<endl;
    else
        cout<<"[RBM] Success: Memory Allocation [accum_vBias_weight]"<<endl;
    
    cout<<endl<<endl<<endl;
    
    Init_RBM();
}

void RBM::Init_RBM()
{
    srand((unsigned)time(NULL));
    for (int h = 0; h < num_hidden_unit; h++)
    {
        for (int v = 0; v < num_visible_unit; v++)
        {
            weight_RBM[h*num_visible_unit+v]   = 0.2 * rand() / RAND_MAX - 0.1;
            //weight_RBM[h*num_visible_unit+v]   = 0;
            accum_weight[h*num_visible_unit+v]   = 0;
            
            input_RBM[v]=0;
            
            nth_sample_v[v]=0;
            nth_probability_v[v]=0;
            
            vBias_weight[v] = 0.2 * rand() / RAND_MAX - 0.1;                // bias weights are 1
            accum_vBias_weight[v] =0;
        }
        
        
        first_probability_h[h] =0;
        first_sample_h[h] =0;
        
        nth_sample_h[h]=0;
        nth_probability_h[h]=0;
        
        hBias_weight[h] = 0.2 * rand() / RAND_MAX - 0.1;                                        // bias weights are 1
        accum_hBias_weight[h]=0;
    }
}

RBM::~RBM()
{
    Deallocate_RBM();
};

void RBM::Deallocate_RBM(){
    
    delete [] weight_RBM;
    delete [] first_sample_h;
    //delete [] input_RBM;
    delete [] first_probability_h;
    delete [] nth_probability_v;
    delete [] nth_probability_h;
    delete [] nth_sample_h;
    delete [] nth_sample_v;
    delete [] hBias_weight;
    delete [] vBias_weight;
    delete [] accum_weight;
    delete [] accum_vBias_weight;
    delete [] accum_hBias_weight;
    
    weight_RBM =NULL;
    first_sample_h=NULL;
    first_probability_h=NULL;
    nth_probability_h=NULL;
    nth_probability_v=NULL;
    nth_sample_h=NULL;
    nth_sample_v=NULL;
    hBias_weight=NULL;
    vBias_weight=NULL;
    accum_weight=NULL;
    accum_vBias_weight=NULL;
    accum_hBias_weight=NULL;
}
////////////////////////////////////////////////////////////////////////////////////////




/*
float* RBM::Positive_Phase_DBN(float* vInput)
{
    //float* point = new float[num_visible_unit];
    
    for(int h = 0 ; h < num_hidden_unit ; h++)
    {
        float dot= 0;
        for(int v = 0 ; v < num_visible_unit ; v++)
        {
            dot += vInput[v] * weight_RBM[h*num_visible_unit+v];
        }
        dot += hBias_weight[h];
        
        first_probability_h[h] = Activation(dot);
        first_sample_h[h] = Sample_Binary_State(first_probability_h[h]);
    }
    return first_sample_h;
}
*/
void RBM::Positive_Phase_DBN(float* vInput, float *sample)
{
    float* output   = new float[num_hidden_unit];
    
    for(int h = 0 ; h < num_hidden_unit ; h++)
    {
        float dot= 0;
        for(int v = 0 ; v < num_visible_unit ; v++)
        {
            dot += vInput[v] * weight_RBM[h*num_visible_unit+v];
        }
        dot += hBias_weight[h];
        
        output[h] = Activation(dot);
        sample[h] = Sample_Binary_State(output[h]);
    }
    delete [] output;
}



float* RBM::Positive_Phase_first(float* vInput)
{
    for(int h = 0 ; h < num_hidden_unit ; h++)
    {
        float dot= 0;
        for(int v = 0 ; v < num_visible_unit ; v++)
        {
            dot += vInput[v] * weight_RBM[h*num_visible_unit+v];
        }
        dot += hBias_weight[h];
        
        first_probability_h[h] = Activation(dot);
        first_sample_h[h] = Sample_Binary_State(first_probability_h[h]);
    }
    return first_sample_h;
}

void RBM::Positive_Phase(float* vInput)
{
    for(int h = 0 ; h < num_hidden_unit ; h++)
    {
        float dot= 0;
        for(int v = 0 ; v < num_visible_unit ; v++)
        {
            dot += vInput[v] * weight_RBM[h*num_visible_unit+v];
        }
        dot += hBias_weight[h];
        
        nth_probability_h[h] = Activation(dot);
        nth_sample_h[h] = Sample_Binary_State(nth_probability_h[h]);
    }
}
void RBM::Negative_Phase(float* hInput)
{
    for(int v = 0 ; v < num_visible_unit ; v++)
    {
        float dot= 0;
        for(int h = 0 ; h < num_hidden_unit ; h++)
        {
            dot += hInput[h] * weight_RBM[h*num_visible_unit+v];             // check error
        }
        dot += vBias_weight[v];
        
        nth_probability_v[v] = Activation(dot);
        nth_sample_v[v] = Sample_Binary_State(nth_probability_v[v]);
    }
}

float* RBM::Negative_Phase_DBN(float* hInput)   // for test
{
    //float* point = new float[num_visible_unit];
    
    for(int v = 0 ; v < num_visible_unit ; v++)
    {
        float dot= 0;
        for(int h = 0 ; h < num_hidden_unit ; h++)
        {
            dot += hInput[h] * weight_RBM[h*num_visible_unit+v];             // check error
        }
        dot += vBias_weight[v];
        
        nth_probability_v[v] = Activation(dot);
        nth_probability_v[v] = Sample_Binary_State(nth_probability_v[v]);
    }
    return nth_probability_v;
}

////////////////////////////////////////////////////////////////////////////////////////


int RBM::Sample_Binary_State(float probability) {
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
////////////////////////////////////////////////////////////////////////////////////////

void RBM::Gibbs_Sampling(float* hInput)
{
    Negative_Phase(hInput);
    Positive_Phase(nth_sample_h);
}

void RBM::Contrastive_Divergence(float* v0Input, int cd_k)
{
    for (int i=0; i < num_visible_unit; i++) {
        input_RBM[i] = v0Input[i];
    }
    
    
    Positive_Phase_first(input_RBM);
    
    for(int step=0; step < cd_k; step++)
    {
        if(step == 0) {
            Gibbs_Sampling(first_sample_h);
        } else {
            Gibbs_Sampling(nth_sample_h);
        }
    }
    //Update_Weight();
}
////////////////////////////////////////////////////////////////////////////////////////

void RBM::Update_Weight(float learningRate)
{
    for(int h=0; h<num_hidden_unit; h++)
    {
        for(int v=0; v<num_visible_unit; v++)
        {
            weight_RBM[h*num_visible_unit+v] += learningRate*(first_sample_h[h] * input_RBM[v] - nth_probability_h[h] * nth_sample_v[v]);
        }
        hBias_weight[h] += learningRate* (first_sample_h[h] - nth_probability_h[h]);   //c : bias for hidden units
    }
    
    for(int v=0; v<num_visible_unit; v++)
        //vBias_weight[v] += learningRate*(input_RBM[v] - nth_sample_v[v]);             // b : bias for input units
        vBias_weight[v] += learningRate*(input_RBM[v] - nth_probability_v[v]);             // b : bias for input units
}

void RBM::Update_Weight_Batch(float learningRate)
{
    
    for(int h=0; h<num_hidden_unit; h++)                            // batch updating
    {
        for(int v=0; v<num_visible_unit; v++)
            weight_RBM[h*num_visible_unit+v] += learningRate * accum_weight[h*num_visible_unit+v];
        hBias_weight[h] += learningRate * accum_hBias_weight[h];
    }
    for(int v=0; v<num_visible_unit; v++)
        vBias_weight[v] += learningRate * accum_vBias_weight[v];
    
    for(int h=0; h<num_hidden_unit; h++)                            // iniciate accum_ after batch updating
    {
        for(int v=0; v<num_visible_unit; v++)
            accum_weight[h*num_visible_unit+v] =0;
        accum_hBias_weight[h] =0;
    }
    for(int v=0; v<num_visible_unit; v++)
        accum_vBias_weight[v] =0;
}
/*
 void RBM::Update_Weight()
 {
 for(int h=0; h<num_hidden_unit; h++)
 {
 for(int v=0; v<num_visible_unit; v++)
 {
 accum_weight[h*num_visible_unit+v] += (first_sample_h[h] * input_RBM[v] - nth_probability_h[h] * nth_sample_v[v]);
 }
 accum_hBias_weight[h] +=  (first_sample_h[h] - nth_probability_h[h]);   //c : bias for hidden units
 }
 
 for(int v=0; v<num_visible_unit; v++)
 accum_vBias_weight[v] += (input_RBM[v] - nth_sample_v[v]);             // b : bias for input units
 }
 */
////////////////////////////////////////////////////////////////////////////////////////

void RBM::Reconstruct(float *vInput,float* reconstructed_v)     // for test
{
    
    float* proba_v = new float[num_visible_unit];
    
    Positive_Phase_first(vInput);
    
    for(int v = 0 ; v < num_visible_unit ; v++)
    {
        float dot= 0;
        for(int h = 0 ; h < num_hidden_unit ; h++)
        {
            dot += first_probability_h[h] * weight_RBM[h*num_hidden_unit+v];
        }
        dot += vBias_weight[v];
        
        
        proba_v[v]= Activation(dot);
        reconstructed_v[v] = Sample_Binary_State(proba_v[v]);
    }
}