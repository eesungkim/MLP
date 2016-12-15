#ifndef MNIST_PARSER_H
#define MNIST_PARSER_H


#include <iostream>
#include <fstream>

using namespace std;

class MNIST {
public:
    void ReadMNIST_Input(string filename, int num_images, float** inputs);
    void ReadMNIST_Label(string filename, int num_labels, float** outputs);
private:
	int BytetoInt(int byte); // convert Byte to Int
};

#endif
