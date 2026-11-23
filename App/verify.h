#ifndef APP_VERIFY_H_
#define APP_VERIFY_H_

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stddef.h>

#define HMAC_KEY_LENGTH 128
#define SGX_SHA256_HASH_SIZE 32
#define ENCLAVE_FILENAME "enclave.signed.so"

/*
 * Fixed layout of the MNIST IDX files, mirrored from Enclave/verify.h so that
 * the untrusted side can size-check its own buffers as well.
 */
#define TRAIN_NUM        60000
#define IMG_ROWS         28
#define IMG_COLS         28
#define IMG_PIXELS       (IMG_ROWS * IMG_COLS)
#define IDX_DATA_HEADER  16
#define IDX_LABEL_HEADER 8
#define DATA_LENGTH_MIN  ((size_t)IDX_DATA_HEADER + (size_t)TRAIN_NUM * IMG_PIXELS)
#define LABEL_LENGTH_MIN ((size_t)IDX_LABEL_HEADER + (size_t)TRAIN_NUM)
#define MAX_PARAM_ROUNDS 65536

typedef struct _sha256_hmac{
    uint8_t _hmac[SGX_SHA256_HASH_SIZE];
    bool operator == (const _sha256_hmac &rhs);
    bool operator != (const _sha256_hmac &rhs);
}sha256_hmac;

#define HMAC_BYTES (TRAIN_NUM * sizeof(sha256_hmac))


int sgx_data_preprocess();
sgx_status_t sgx_get_seed(uint32_t* seed);
sgx_status_t sgx_init(int num_data);
unsigned long int initialize_enclave(void);
void destroy_enclave();
void initialize_trustfl(const int training_rounds);


#endif