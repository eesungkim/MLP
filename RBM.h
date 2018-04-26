#ifndef RBM_H
#define RBM_H

#include <iostream>
#include <ctime>
#include <cmath>
#include <vector>
#include <fstream>
#include <cstdlib>
#include <string>
using namespace std;

class RBM{
private:
    int num_visible_unit;
    int num_hidden_unit;
    
    
    float* input_RBM;
    
    float* first_sample_h;
    float* first_probability_h;
    
    float* nth_sample_v;
    float* nth_probability_v;
    
    float* nth_sample_h;
    float* nth_probability_h;
    
    float* weight_RBM;
    
    float* hBias_weight;
    float* vBias_weight;
    
    float* accum_weight;
    float* accum_hBias_weight;
    float* accum_vBias_weight;
    
public:
    RBM(){};
    ~RBM();
    
    float* Get_Weight(){ return weight_RBM;}
    float* Get_Bias_Weight() {return hBias_weight;}
    
    void Allocate_RBM(int num_visible_unit, int num_hidden_unit);
    void Init_RBM();
    void Deallocate_RBM();
    
    void Contrastive_Divergence(float* v0Input, int cd_k);
    void Gibbs_Sampling(float* hInput);
    
    
    void Positive_Phase_DBN(float* vInput, float* sample);
    float* Negative_Phase_DBN(float* hInput);   // for test
    
    float* Positive_Phase_first(float* vInput);
    void Positive_Phase(float* vInput);
    void Negative_Phase(float* hInput);
    
    float Activation(float net){return 1.0 / (1.0 + exp(-net));}    // sigmoid
    
    void Update_Weight(float learningRate);
    void Update_Weight_Batch(float learningRate);
    
    int Sample_Binary_State(float probability);
    void Reconstruct(float* vInput, float* reconstructed_v); // for test
};

#endif