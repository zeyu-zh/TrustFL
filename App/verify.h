#ifndef APP_VERIFY_H_
#define APP_VERIFY_H_

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#define HMAC_KEY_LENGTH 128
#define SGX_SHA256_HASH_SIZE 32
#define ENCLAVE_FILENAME "enclave.signed.so"

typedef struct _sha256_hmac{
    uint8_t _hmac[SGX_SHA256_HASH_SIZE];
    bool operator == (const _sha256_hmac &rhs);
    bool operator != (const _sha256_hmac &rhs);
}sha256_hmac;


sgx_status_t sgx_data_preprocess();
sgx_status_t sgx_get_seed(uint32_t* seed);
sgx_status_t sgx_init(int num_data);
unsigned long int initialize_enclave(void);
void destroy_enclave();
void initialize_trustfl(const int training_rounds);


#endif