#include "MNIST.h"

void MNIST::ReadMNIST_Input(string filename,int num_images, float** inputs)
{
    ifstream image_file(filename, ios::binary); // read MNIST-image database
    if (image_file.is_open())
    {
        cout << "MNIST : IMAGE OPENED" << endl;
        int magic_number = 0;
        int num_images_from_file = 0;
        int image_rows = 0;
        int image_cols = 0;
        image_file.read((char*)&magic_number, sizeof(magic_number)); // get magic number : don't use
        magic_number = BytetoInt(magic_number);
        image_file.read((char*)&num_images_from_file, sizeof(num_images_from_file)); // get num_images : don't use
        num_images_from_file = BytetoInt(num_images_from_file);
        image_file.read((char*)&image_rows, sizeof(image_rows)); // get rows
        image_rows = BytetoInt(image_rows);
        image_file.read((char*)&image_cols, sizeof(image_cols)); // get cols
        image_cols = BytetoInt(image_cols);
        for (int i = 0; i < num_images; i++)
        {
            for (int r = 0; r < image_rows; r++)
            {
                for (int c = 0; c < image_cols; c++)
                {
                    unsigned char temp = 0;
                    image_file.read((char*)&temp, sizeof(temp));
                    if ((float)temp > 0)
                        inputs[i][(image_rows*r) + c] = 1.f;
                    else
                        inputs[i][(image_rows*r) + c] = 0.f;
                }
            }
        }
    }
    else
        cout << "MNIST : IMAGE NOT OPENED" << endl;
}

void MNIST::ReadMNIST_Label(string filename,int num_labels, float** outputs)
{
    ifstream label_file(filename, ios::binary); // read MNIST-label database
    if (label_file.is_open()){
        cout << "MNIST : LABEL OPENED" << endl;
        int magic_number = 0;
        int num_labels_from_file = 0;
        label_file.read((char*)&magic_number, sizeof(magic_number)); // get magic number : don't use
        magic_number = BytetoInt(magic_number);
        label_file.read((char*)&num_labels_from_file, sizeof(num_labels_from_file)); // get num_images : don't use
        num_labels_from_file = BytetoInt(num_labels_from_file);
        for (int i = 0; i < num_labels; i++){
            unsigned char temp = 0;
            label_file.read((char*)&temp, sizeof(temp));
            for (int j = 0; j < 10; j++){
                if ((int)temp == j)
                    outputs[i][j] = 1.f;
                else
                    outputs[i][j] = 0.f;
            }
        }
    }
    else
        cout << "MNIST : LABEL NOT OPENED" << endl;
}


int MNIST::BytetoInt(int byte) // Byte(size by 4) to Int
{
	unsigned char byte0, byte1, byte2, byte3; // MSB(byte0) / byte1 / byte2 / LSB(byte3)
	byte0 = byte & 0xFF; // 0xFF : 00000000 00000000 00000000 11111111
	byte1 = (byte >> 8) & 0xFF;
	byte2 = (byte >> 16) & 0xFF;
	byte3 = (byte >> 24) & 0xFF;
	return ((int)byte0 << 24) | ((int)byte1 << 16) | ((int)byte2 << 8) | byte3; // by Endian problem, we have to change order and calculate
}
