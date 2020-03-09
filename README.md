Trusted Federated Learning Guarded by Intel SGX
===============================================

# TrustFL

Introduction
------------




本项目代码主要依托linux下的[Intel SGX SDK](https://github.com/intel/linux-sgx)，基于[Mini-Dnn](https://github.com/iamhankai/mini-dnn-cpp)以及可在SGX下运行的[Eigen3](https://github.com/ftramer/slalom/tree/master/Include/eigen3_sgx)开发。代码由Enlave(C++)及Untrusted(Python)两部分构成。Enclave完成上文中的初始化及训练验证工作，Untrusted主要执行在不可信环境的训练。此处为了展示方案有效性，我们构建了一个LeNet-5用于完成MNIST手写数字数据集的分类任务。



Running Environment
-------------------
- OS: Ubuntu 16.04
- Driver: [Intel-SGX-Driver](https://github.com/01org/linux-sgx-driver)
- SDK: [linux-sgx](https://github.com/intel/linux-sgx/blob/master/README.md)
- python:
```shell
$ sudo pip install --upgrade pip
$ sudo pip install torch
$ sudo pip install torchvision
```

Running steps
-------------------
```shell
$ source <path_of_SGX_SDK>/environment cd <path_of_TrustFL>
$ make
$ ./trust_fl
```
# Publications

Xiaoli Zhang, Fengting Li, Zeyu Zhang, Qi Li, Cong Wang, and Jianping Wu, "Enabling Execution Assurance of Federated Learning at Untrusted Participants", in the 39th International Conference on Computer Communications (INFOCOMM'20)
