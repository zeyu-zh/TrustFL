#include <stdint.h>
#include <iostream>
#include <unordered_set>
#include <vector>
#include <string>
#include <map>
#include <iterator>
#include <stdio.h>
#include "verify.h"
#include "Enclave.h"
#include "Enclave_t.h"
#include "sgx_trts.h"
#include "sgx_tcrypto.h"

using namespace std;
/* 
 * Global variants 
 */
uint8_t hmac_key[HMAC_KEY_LENGTH]; // The key of hmac 
int training_num = 0, rows = 0, columns = 0; // The metadata of training data
uint32_t seed; // Real random data
sgx_sha256_hash_t *p_parameter_hash; // The buffer for storing hash of parameters, located in enclave
sha256_hmac *p_hmacu; // The buffer for storing hmac of training data, located in untrusted memroy


/*
 * Operator Overloading of struct _sha256_hmac.
 */
bool _sha256_hmac::operator == (const _sha256_hmac &rhs){ return (0 == memcmp(_hmac, rhs._hmac, 32)) ? true : false; }
bool _sha256_hmac::operator != (const _sha256_hmac &rhs){ return (0 == memcmp(_hmac, rhs._hmac, 32)) ? false : true; }


/*
 * Generate hmac_key with trusted real random generator inside CPU.
 */
void ecall_init(){
    sgx_status_t ret = SGX_SUCCESS;

    /* init hmac key */
    ret = sgx_read_rand(hmac_key, HMAC_KEY_LENGTH);
}

/*
 * Preprocess training parameters -> generate hash.
 */
void ecall_param_preprocess(int num_param){
    float *p_conv1, *p_conv2, *p_fc1, *p_fc2, *p_fc3;
    sgx_sha_state_handle_t handle;
    sgx_sha256_hash_t hash;
    int ret_val;

    /* init hash of parameters */
    p_parameter_hash = (sgx_sha256_hash_t*)malloc(sizeof(sgx_sha256_hash_t) * num_param);
    for(int i = 0; i < num_param; i++){
        ret_val = sgx_get_parameters(i, &p_conv1, &p_conv2, &p_fc1, &p_fc2, &p_fc3);
        sgx_sha256_init(&handle);
        sgx_sha256_update((uint8_t*)p_conv1, sizeof(float) * CONV1_SIZE, handle);
        sgx_sha256_update((uint8_t*)p_conv2, sizeof(float) * CONV2_SIZE, handle);
        sgx_sha256_update((uint8_t*)p_fc1, sizeof(float) * FC1_SIZE, handle);
        sgx_sha256_update((uint8_t*)p_fc2, sizeof(float) * FC2_SIZE, handle);
        sgx_sha256_update((uint8_t*)p_fc3, sizeof(float) * FC3_SIZE, handle);
        sgx_sha256_get_hash(handle, &hash);
        memcpy(&(p_parameter_hash[i]), &hash, sizeof(sgx_sha256_hash_t));
        sgx_free_parameters(p_conv1, p_conv2, p_fc1, p_fc2, p_fc3);

    }
}

/*
 * Generatre seed with sgx_read_rand.
 */
unsigned int ecall_get_seed(){
    sgx_status_t ret = SGX_SUCCESS;

    /* get random data */
    ret = sgx_read_rand((uint8_t*)&seed, 8);
    return seed;
}

/*
 * Preprocess training data -> generate hmac and initialize the metadata of training data.
 */
int ecall_data_preprocess(unsigned char* p_datau, unsigned int data_length, unsigned char* p_labelu, unsigned int label_length, unsigned long p_training_hmacu){
    uint8_t* p_data_label = nullptr;
    sha256_hmac temp;
    
    sgx_status_t ret = SGX_SUCCESS;

    p_hmacu = (sha256_hmac*)p_training_hmacu;
    /* alloc memory for hmac */    
    training_num = 60000;
    rows = 28;
    columns = 28;
    p_data_label = (uint8_t*)malloc(rows * columns + 1);

    /* update data pointer and label pointer */
    p_datau = p_datau + 16;
    p_labelu = p_labelu + 8;
    for(int i = 0; i < training_num; i++){
        memcpy(p_data_label, p_datau, rows*columns);
        p_data_label[rows*columns] = *p_labelu;
        ret = sgx_hmac_sha256_msg(p_data_label, rows * columns + 1, hmac_key, HMAC_KEY_LENGTH, (uint8_t*)&temp, SGX_SHA256_HASH_SIZE);
        if (ret != SGX_SUCCESS)
            return -2;
        p_hmacu[i] = temp;
        p_datau = p_datau + rows * columns;
        p_labelu = p_labelu + 1;
    }


    return 0;
}

/*
 * Read parameters from disk, return float array.
 */
int sgx_get_parameters(int round, float** p_conv1, float** p_conv2, float** p_fc1, float** p_fc2, float** p_fc3){
    string path;
    int ret;

    path = "./Parameter/" + to_string(round) + "/conv1.txt";
    *p_conv1  = (float*)malloc(sizeof(float) * CONV1_SIZE);
    ocall_get_parameter(&ret, path.c_str(), *p_conv1, CONV1_SIZE);
    if(ret == -1){
        free(*p_conv1);
        *p_conv1 = nullptr;
        return -1;
    }

    path = "./Parameter/" + to_string(round) + "/conv2.txt";
    *p_conv2 = (float*)malloc(sizeof(float) * CONV2_SIZE);
    ocall_get_parameter(&ret, path.c_str(), *p_conv2, CONV2_SIZE);
    if(ret == -1){
        free(*p_conv1); free(*p_conv2);
        *p_conv1 = nullptr; *p_conv2 = nullptr;
        return -1;
    }

    path = "./Parameter/" + to_string(round) + "/fc1.txt";
    *p_fc1 = (float*)malloc(sizeof(float) * FC1_SIZE);
    ocall_get_parameter(&ret, path.c_str(), *p_fc1, FC1_SIZE);
    if(ret == -1){
        free(*p_conv1); free(*p_conv2); free(*p_fc1);
        *p_conv1 = nullptr; *p_conv2 = nullptr; *p_fc1 = nullptr;
        return -1;
    }

    path = "./Parameter/" + to_string(round) + "/fc2.txt";
    *p_fc2 = (float*)malloc(sizeof(float) * FC2_SIZE+1);
    ocall_get_parameter(&ret, path.c_str(), *p_fc2, FC2_SIZE);
    if(ret == -1){
        free(*p_conv1); free(*p_conv2); free(*p_fc1); free(*p_fc2);
        *p_conv1 = nullptr; *p_conv2 = nullptr; *p_fc1 = nullptr; *p_fc2 = nullptr;
        return -1;
    }

    path = "./Parameter/" + to_string(round) + "/fc3.txt";
    *p_fc3 = (float*)malloc(sizeof(float) * FC3_SIZE);
    ocall_get_parameter(&ret, path.c_str(), *p_fc3, FC3_SIZE);
    if(ret == -1){
        free(*p_conv1); free(*p_conv2); free(*p_fc1); free(*p_fc2); free(*p_fc3);
        *p_conv1 = nullptr; *p_conv2 = nullptr; *p_fc1 = nullptr; *p_fc2 = nullptr; *p_fc3 = nullptr;
        return -1;
    }

    return 0;
}

/*
 * Release the memory allocated for parameters.
 */
void sgx_free_parameters(float* p_conv1, float* p_conv2, float* p_fc1, float* p_fc2, float* p_fc3){
    if(p_conv1 != nullptr) free(p_conv1);
    if(p_conv2 != nullptr) free(p_conv2);
    if(p_fc1 != nullptr) free(p_fc1);
    if(p_fc2 != nullptr) free(p_fc2);
    if(p_fc3 != nullptr) free(p_fc3);
}

/*
 * Double-check the input parameters before verification.
 */
int sgx_check_parameter(int round, float* p_conv1, float* p_conv2, float* p_fc1, float* p_fc2, float* p_fc3){
    sgx_sha_state_handle_t handle;
    sgx_sha256_hash_t hash;

    sgx_sha256_init(&handle);
    sgx_sha256_update((uint8_t*)p_conv1, sizeof(float) * CONV1_SIZE, handle);
    sgx_sha256_update((uint8_t*)p_conv2, sizeof(float) * CONV2_SIZE, handle);
    sgx_sha256_update((uint8_t*)p_fc1, sizeof(float) * FC1_SIZE, handle);
    sgx_sha256_update((uint8_t*)p_fc2, sizeof(float) * FC2_SIZE, handle);
    sgx_sha256_update((uint8_t*)p_fc3, sizeof(float) * FC3_SIZE, handle);
    sgx_sha256_get_hash(handle, &hash);

    return memcmp(&hash, &(p_parameter_hash[round]), sizeof(sgx_sha256_hash_t));
}

/*
 * Double-check the input data before verification.
 */
int sgx_check_data(int index, uint8_t* p_data, uint8_t label){
    uint8_t check[28 * 28 + 1];
    sha256_hmac temp;
    sgx_status_t ret;

    memcpy(check, p_data, 28 * 28);
    check[28 * 28] = label;
    ret = sgx_hmac_sha256_msg(check, 28 * 28 + 1, hmac_key, HMAC_KEY_LENGTH, (uint8_t*)&temp, SGX_SHA256_HASH_SIZE);
    if (ret != SGX_SUCCESS)
        return -2;

    return temp == p_hmacu[index] ? 0 : -1;
}

    


/* How to use sgx_get_paramters and sgx_free_parameters
 * float *p_conv1, *p_conv2, *p_fc1, *p_fc2, *p_fc3;
 * sgx_get_parameters(0, &p_conv1, &p_conv2, &p_fc1, &p_fc2, &p_fc3);
 * sgx_free_parameters(p_conv1, p_conv2, p_fc1, p_fc2, p_fc3);
 */