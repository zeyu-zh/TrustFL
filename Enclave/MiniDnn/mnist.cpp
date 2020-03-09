#include "./mnist.h"
#include "Enclave.h"
#include "Enclave_t.h"  /* print_string */
//#include <memory.h>

int ReverseInt(int i) {
  unsigned char ch1, ch2, ch3, ch4;
  ch1 = i & 255;
  ch2 = (i >> 8) & 255;
  ch3 = (i >> 16) & 255;
  ch4 = (i >> 24) & 255;
  return ((int)ch1 << 24) + ((int)ch2 << 16) + ((int)ch3 << 8) + ch4;
}

void MNIST::read_mnist_data(std::string filename, Matrix& data) {
  struct file_info info;
  int ret;
  ocall_open_file(&ret, filename.c_str(), &info);
  void * p_data = info.p_file;
  if (ret == 0) {
    int magic_number = 0;
    int number_of_images = 0;
    int n_rows = 0;
    int n_cols = 0;
    unsigned char label;
    int * temp_int = (int*)p_data;
    magic_number = temp_int[0];
    number_of_images = temp_int[1];
    n_rows = temp_int[2];
    n_cols = temp_int[3];
    magic_number = ReverseInt(magic_number);
    number_of_images = ReverseInt(number_of_images);
    n_rows = ReverseInt(n_rows);
    n_cols = ReverseInt(n_cols);
    temp_int++;
    temp_int++;
    temp_int++;
    temp_int++;
    unsigned char * temp_char = (unsigned char *)temp_int;
    unsigned int index = 0;
    data.resize(n_cols * n_rows, number_of_images);
    for (int i = 0; i < number_of_images; i++) {
      for (int r = 0; r < n_rows; r++) {
        for (int c = 0; c < n_cols; c++) {
          unsigned char image = 0;
          image = temp_char[index];
          index++;
          data(r * n_cols + c, i) = (float)image;
        }
      }
    }
  }
  ocall_close_file(&ret, &info);
}

void MNIST::read_mnist_label(std::string filename, Matrix& labels) {
  struct file_info info;
  int ret;
  ocall_open_file(&ret, filename.c_str(), &info);
  void * p_data = info.p_file;
  if (ret == 0) {
    int magic_number = 0;
    int number_of_images = 0;
    int * temp_int = (int*)p_data;
    magic_number = temp_int[0];
    temp_int++;
    number_of_images = temp_int[0];
    temp_int++;
    magic_number = ReverseInt(magic_number);
    number_of_images = ReverseInt(number_of_images);
    unsigned int index = 0;
    unsigned char * temp_char = (unsigned char *)temp_int;
    labels.resize(1, number_of_images);
    for (int i = 0; i < number_of_images; i++) {
      unsigned char label = 0;
      label = temp_char[index];
      index++;
      labels(0, i) = (float)label;
    }
  }
  ocall_close_file(&ret, &info);
}

void MNIST::read() {
  read_mnist_data(data_dir + "train-images.idx3-ubyte", train_data);
  //read_mnist_data(data_dir + "t10k-images-idx3-ubyte", test_data);
  read_mnist_label(data_dir + "train-labels.idx1-ubyte", train_labels);
  //read_mnist_label(data_dir + "t10k-labels-idx1-ubyte", test_labels);
}
