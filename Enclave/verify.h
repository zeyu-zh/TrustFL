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

/*
 * Layout of the IDX files that carry the training set.
 * The enclave works on a fixed-size training set, so the sizes are constants
 * instead of being derived from untrusted host input.
 */
#define TRAIN_NUM        60000
#define IMG_ROWS         28
#define IMG_COLS         28
#define IMG_PIXELS       (IMG_ROWS * IMG_COLS)
#define IDX_DATA_HEADER  16
#define IDX_LABEL_HEADER 8

/* Minimal number of bytes the host has to provide for a full preprocessing run. */
#define DATA_LENGTH_MIN  ((size_t)IDX_DATA_HEADER + (size_t)TRAIN_NUM * IMG_PIXELS)
#define LABEL_LENGTH_MIN ((size_t)IDX_LABEL_HEADER + (size_t)TRAIN_NUM)

/* Upper bound for the number of parameter rounds accepted by the enclave. */
#define MAX_PARAM_ROUNDS 65536

/* Error codes returned by the ecalls. */
#define E_VERIFY_INVALID    (-1)   /* bad/untrusted argument, or call ordering error */
#define E_VERIFY_CRYPTO     (-2)   /* HMAC/SHA operation failed */
#define E_VERIFY_NOMEM      (-3)   /* enclave-side allocation failed */

typedef struct _sha256_hmac{
    uint8_t _hmac[SGX_SHA256_HASH_SIZE];
    bool operator == (const _sha256_hmac &rhs);
    bool operator != (const _sha256_hmac &rhs);
}sha256_hmac;

#define HMAC_BYTES (TRAIN_NUM * sizeof(sha256_hmac))

int sgx_get_parameters(int round, float** p_conv1, float** p_conv2, float** p_fc1, float** p_fc2, float** p_fc3);
void sgx_free_parameters(float* p_conv1, float* p_conv2, float* p_fc1, float* p_fc2, float* p_fc3);
int sgx_check_parameter(int round, float* p_conv1, float* p_conv2, float* p_fc1, float* p_fc2, float* p_fc3);
int sgx_check_data(int index, uint8_t* p_data, uint8_t label);
#endif