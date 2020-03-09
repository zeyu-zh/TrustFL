#ifndef ENCLAVE_VERIFY_H_
#define ENCLAVE_VERIFY_H_


#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <iostream>
#include <sgx_tcrypto.h>
#include <sgx_trts.h>

#define HMAC_KEY_LENGTH 128
#define CONV1_SIZE (30 * 5 + 6)
#define CONV2_SIZE (480 * 5 + 16)
#define FC1_SIZE (400 * 120 + 120)
#define FC2_SIZE (120 * 84 + 84)
#define FC3_SIZE (84 * 10 + 10)

typedef struct _sha256_hmac{
    uint8_t _hmac[SGX_SHA256_HASH_SIZE];
    bool operator == (const _sha256_hmac &rhs);
    bool operator != (const _sha256_hmac &rhs);
}sha256_hmac;

int sgx_get_parameters(int round, float** p_conv1, float** p_conv2, float** p_fc1, float** p_fc2, float** p_fc3);
void sgx_free_parameters(float* p_conv1, float* p_conv2, float* p_fc1, float* p_fc2, float* p_fc3);
int sgx_check_parameter(int round, float* p_conv1, float* p_conv2, float* p_fc1, float* p_fc2, float* p_fc3);
int sgx_check_data(int index, uint8_t* p_data, uint8_t label);
#endif