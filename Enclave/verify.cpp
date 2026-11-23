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
#include "sgx_lfence.h"
#include "sgx_tcrypto.h"

using namespace std;
/* 
 * Global variants 
 */
static bool hmac_key_ready = false;   // hmac_key is only usable after a successful ecall_init
uint8_t hmac_key[HMAC_KEY_LENGTH]; // The key of hmac 
int training_num = 0, rows = 0, columns = 0; // The metadata of training data
static int param_rounds = 0; // Number of rounds stored in p_parameter_hash
uint32_t seed; // Real random data
sgx_sha256_hash_t *p_parameter_hash = nullptr; // The buffer for storing hash of parameters, located in enclave
sha256_hmac *p_hmacu = nullptr; // Host-side buffer that receives the hmac of the training data
static sha256_hmac *p_hmac_store = nullptr; // Enclave-side copy of the hmac, used by sgx_check_data


/*
 * Operator Overloading of struct _sha256_hmac.
 */
bool _sha256_hmac::operator == (const _sha256_hmac &rhs){ return (0 == memcmp(_hmac, rhs._hmac, 32)) ? true : false; }
bool _sha256_hmac::operator != (const _sha256_hmac &rhs){ return (0 == memcmp(_hmac, rhs._hmac, 32)) ? false : true; }


/*
 * Generate hmac_key with trusted real random generator inside CPU.
 */
void ecall_init(){
    hmac_key_ready = false;
    memset(hmac_key, 0, sizeof(hmac_key));

    /* init hmac key */
    if (sgx_read_rand(hmac_key, HMAC_KEY_LENGTH) != SGX_SUCCESS){
        memset(hmac_key, 0, sizeof(hmac_key));
        return;
    }
    hmac_key_ready = true;
}

/*
 * Preprocess training parameters -> generate hash.
 */
void ecall_param_preprocess(int num_param){
    float *p_conv1 = nullptr, *p_conv2 = nullptr, *p_fc1 = nullptr, *p_fc2 = nullptr, *p_fc3 = nullptr;
    sgx_sha_state_handle_t handle;
    sgx_sha256_hash_t hash;

    /* num_param comes from the untrusted host: bound it before using it. */
    if (num_param <= 0 || num_param > MAX_PARAM_ROUNDS)
        return;

    if (p_parameter_hash != nullptr){
        free(p_parameter_hash);
        p_parameter_hash = nullptr;
        param_rounds = 0;
    }

    /* init hash of parameters */
    p_parameter_hash = (sgx_sha256_hash_t*)malloc(sizeof(sgx_sha256_hash_t) * (size_t)num_param);
    if (p_parameter_hash == nullptr)
        return;
    memset(p_parameter_hash, 0, sizeof(sgx_sha256_hash_t) * (size_t)num_param);
    param_rounds = num_param;

    for(int i = 0; i < num_param; i++){
        if (sgx_get_parameters(i, &p_conv1, &p_conv2, &p_fc1, &p_fc2, &p_fc3) != 0){
            /* leave the zero hash in place: the corresponding round will fail to verify */
            p_conv1 = p_conv2 = p_fc1 = p_fc2 = p_fc3 = nullptr;
            continue;
        }
        if (sgx_sha256_init(&handle) != SGX_SUCCESS){
            sgx_free_parameters(p_conv1, p_conv2, p_fc1, p_fc2, p_fc3);
            continue;
        }
        sgx_sha256_update((uint8_t*)p_conv1, sizeof(float) * CONV1_SIZE, handle);
        sgx_sha256_update((uint8_t*)p_conv2, sizeof(float) * CONV2_SIZE, handle);
        sgx_sha256_update((uint8_t*)p_fc1, sizeof(float) * FC1_SIZE, handle);
        sgx_sha256_update((uint8_t*)p_fc2, sizeof(float) * FC2_SIZE, handle);
        sgx_sha256_update((uint8_t*)p_fc3, sizeof(float) * FC3_SIZE, handle);
        sgx_sha256_get_hash(handle, &hash);
        sgx_sha256_close(handle);
        memcpy(&(p_parameter_hash[i]), &hash, sizeof(sgx_sha256_hash_t));
        sgx_free_parameters(p_conv1, p_conv2, p_fc1, p_fc2, p_fc3);
    }
}

/*
 * Generatre seed with sgx_read_rand.
 */
unsigned int ecall_get_seed(){
    uint32_t new_seed = 0;

    /* get random data: seed is a uint32_t, only read as many bytes as it holds */
    if (sgx_read_rand((uint8_t*)&new_seed, sizeof(new_seed)) != SGX_SUCCESS)
        return 0;
    seed = new_seed;
    return (unsigned int)seed;
}

/*
 * Preprocess training data -> generate hmac and initialize the metadata of training data.
 *
 * The three addresses are plain host pointers ([user_check] in the EDL), so
 * everything is validated first: the lengths, the fact that every span lies
 * outside the enclave, and that the key has been initialized.  The training set
 * is then copied into enclave-owned memory (single snapshot) so that the
 * untrusted host cannot change it while the loop is running.
 */
int ecall_data_preprocess(unsigned char* p_datau, unsigned int data_length,
                          unsigned char* p_labelu, unsigned int label_length,
                          unsigned long p_training_hmacu){
    uint8_t* p_data_label = nullptr;
    uint8_t* data_snapshot = nullptr;
    uint8_t* label_snapshot = nullptr;
    sha256_hmac *p_out_hmac = (sha256_hmac*)p_training_hmacu;
    int rc = 0;

    /* The HMAC key must be initialized before any tag can be produced. */
    if (!hmac_key_ready)
        return E_VERIFY_INVALID;
    if (p_datau == nullptr || p_labelu == nullptr || p_training_hmacu == 0)
        return E_VERIFY_INVALID;

    /* The enclave works on a fixed-size training set: demand the full buffers. */
    if ((size_t)data_length < DATA_LENGTH_MIN || (size_t)label_length < LABEL_LENGTH_MIN)
        return E_VERIFY_INVALID;

    /* No span may overlap the enclave: the host only owns plain memory. */
    if (!sgx_is_outside_enclave((const void*)p_datau, DATA_LENGTH_MIN) ||
        !sgx_is_outside_enclave((const void*)p_labelu, LABEL_LENGTH_MIN) ||
        !sgx_is_outside_enclave((const void*)p_training_hmacu, HMAC_BYTES))
        return E_VERIFY_INVALID;
    sgx_lfence();

    if (p_hmac_store == nullptr){
        p_hmac_store = (sha256_hmac*)malloc(HMAC_BYTES);
        if (p_hmac_store == nullptr)
            return E_VERIFY_NOMEM;
    }
    p_data_label = (uint8_t*)malloc(IMG_PIXELS + 1);
    data_snapshot = (uint8_t*)malloc(DATA_LENGTH_MIN);
    label_snapshot = (uint8_t*)malloc(LABEL_LENGTH_MIN);
    if (p_data_label == nullptr || data_snapshot == nullptr || label_snapshot == nullptr){
        free(p_data_label);
        free(data_snapshot);
        free(label_snapshot);
        return E_VERIFY_NOMEM;
    }

    /* Snapshot the untrusted inputs, then only ever touch the copies. */
    memcpy(data_snapshot, p_datau, DATA_LENGTH_MIN);
    memcpy(label_snapshot, p_labelu, LABEL_LENGTH_MIN);

    /* update data pointer and label pointer (skip the IDX headers) */
    const uint8_t* p_data = data_snapshot + IDX_DATA_HEADER;
    const uint8_t* p_label = label_snapshot + IDX_LABEL_HEADER;

    for(int i = 0; i < TRAIN_NUM; i++){
        sha256_hmac temp;
        memcpy(p_data_label, p_data, IMG_PIXELS);
        p_data_label[IMG_PIXELS] = *p_label;
        if (sgx_hmac_sha256_msg(p_data_label, IMG_PIXELS + 1, hmac_key, HMAC_KEY_LENGTH,
                                (uint8_t*)&temp, SGX_SHA256_HASH_SIZE) != SGX_SUCCESS){
            rc = E_VERIFY_CRYPTO;
            break;
        }
        p_hmac_store[i] = temp;
        p_out_hmac[i] = temp;
        p_data += IMG_PIXELS;
        p_label += 1;
    }

    free(p_data_label);
    free(data_snapshot);
    free(label_snapshot);

    if (rc != 0){
        /* do not publish a partially computed state */
        p_hmacu = nullptr;
        return rc;
    }

    training_num = TRAIN_NUM;
    rows = IMG_ROWS;
    columns = IMG_COLS;
    p_hmacu = p_out_hmac;
    return 0;
}

/*
 * Check a single training sample against the hmac computed by ecall_data_preprocess.
 * Added so that the verification path can be exercised (and tested) on its own.
 */
int ecall_verify_sample(unsigned int index, unsigned char* p_data, unsigned char label){
    if (p_data == nullptr || index >= (unsigned int)TRAIN_NUM)
        return E_VERIFY_INVALID;
    if (!sgx_is_outside_enclave((const void*)p_data, IMG_PIXELS))
        return E_VERIFY_INVALID;
    sgx_lfence();
    return sgx_check_data((int)index, p_data, label);
}

/*
 * Read parameters from disk, return float array.
 */
int sgx_get_parameters(int round, float** p_conv1, float** p_conv2, float** p_fc1, float** p_fc2, float** p_fc3){
    string path;
    int ret;

    if (round < 0 || p_conv1 == nullptr || p_conv2 == nullptr || p_fc1 == nullptr ||
        p_fc2 == nullptr || p_fc3 == nullptr)
        return -1;

    path = "./Parameter/" + to_string(round) + "/conv1.txt";
    *p_conv1  = (float*)malloc(sizeof(float) * CONV1_SIZE);
    if (*p_conv1 == nullptr) return -1;
    ocall_get_parameter(&ret, path.c_str(), *p_conv1, CONV1_SIZE);
    if(ret == -1){
        free(*p_conv1);
        *p_conv1 = nullptr;
        return -1;
    }

    path = "./Parameter/" + to_string(round) + "/conv2.txt";
    *p_conv2 = (float*)malloc(sizeof(float) * CONV2_SIZE);
    if (*p_conv2 == nullptr){
        free(*p_conv1); *p_conv1 = nullptr;
        return -1;
    }
    ocall_get_parameter(&ret, path.c_str(), *p_conv2, CONV2_SIZE);
    if(ret == -1){
        free(*p_conv1); free(*p_conv2);
        *p_conv1 = nullptr; *p_conv2 = nullptr;
        return -1;
    }

    path = "./Parameter/" + to_string(round) + "/fc1.txt";
    *p_fc1 = (float*)malloc(sizeof(float) * FC1_SIZE);
    if (*p_fc1 == nullptr){
        free(*p_conv1); free(*p_conv2);
        *p_conv1 = nullptr; *p_conv2 = nullptr;
        return -1;
    }
    ocall_get_parameter(&ret, path.c_str(), *p_fc1, FC1_SIZE);
    if(ret == -1){
        free(*p_conv1); free(*p_conv2); free(*p_fc1);
        *p_conv1 = nullptr; *p_conv2 = nullptr; *p_fc1 = nullptr;
        return -1;
    }

    path = "./Parameter/" + to_string(round) + "/fc2.txt";
    *p_fc2 = (float*)malloc(sizeof(float) * FC2_SIZE+1);
    if (*p_fc2 == nullptr){
        free(*p_conv1); free(*p_conv2); free(*p_fc1);
        *p_conv1 = nullptr; *p_conv2 = nullptr; *p_fc1 = nullptr;
        return -1;
    }
    ocall_get_parameter(&ret, path.c_str(), *p_fc2, FC2_SIZE);
    if(ret == -1){
        free(*p_conv1); free(*p_conv2); free(*p_fc1); free(*p_fc2);
        *p_conv1 = nullptr; *p_conv2 = nullptr; *p_fc1 = nullptr; *p_fc2 = nullptr;
        return -1;
    }

    path = "./Parameter/" + to_string(round) + "/fc3.txt";
    *p_fc3 = (float*)malloc(sizeof(float) * FC3_SIZE);
    if (*p_fc3 == nullptr){
        free(*p_conv1); free(*p_conv2); free(*p_fc1); free(*p_fc2);
        *p_conv1 = nullptr; *p_conv2 = nullptr; *p_fc1 = nullptr; *p_fc2 = nullptr;
        return -1;
    }
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
    int ret_val;

    if (p_parameter_hash == nullptr || round < 0 || round >= param_rounds)
        return E_VERIFY_INVALID;
    if (p_conv1 == nullptr || p_conv2 == nullptr || p_fc1 == nullptr ||
        p_fc2 == nullptr || p_fc3 == nullptr)
        return E_VERIFY_INVALID;

    if (sgx_sha256_init(&handle) != SGX_SUCCESS)
        return E_VERIFY_CRYPTO;
    sgx_sha256_update((uint8_t*)p_conv1, sizeof(float) * CONV1_SIZE, handle);
    sgx_sha256_update((uint8_t*)p_conv2, sizeof(float) * CONV2_SIZE, handle);
    sgx_sha256_update((uint8_t*)p_fc1, sizeof(float) * FC1_SIZE, handle);
    sgx_sha256_update((uint8_t*)p_fc2, sizeof(float) * FC2_SIZE, handle);
    sgx_sha256_update((uint8_t*)p_fc3, sizeof(float) * FC3_SIZE, handle);
    ret_val = sgx_sha256_get_hash(handle, &hash);
    sgx_sha256_close(handle);
    if (ret_val != SGX_SUCCESS)
        return E_VERIFY_CRYPTO;

    return memcmp(&hash, &(p_parameter_hash[round]), sizeof(sgx_sha256_hash_t)) == 0 ? 0 : E_VERIFY_INVALID;
}

/*
 * Double-check the input data before verification.
 */
int sgx_check_data(int index, uint8_t* p_data, uint8_t label){
    uint8_t check[IMG_PIXELS + 1];
    sha256_hmac temp;

    if (!hmac_key_ready || p_hmac_store == nullptr || p_data == nullptr)
        return E_VERIFY_INVALID;
    if (index < 0 || index >= TRAIN_NUM)
        return E_VERIFY_INVALID;

    memcpy(check, p_data, IMG_PIXELS);
    check[IMG_PIXELS] = label;
    if (sgx_hmac_sha256_msg(check, IMG_PIXELS + 1, hmac_key, HMAC_KEY_LENGTH,
                            (uint8_t*)&temp, SGX_SHA256_HASH_SIZE) != SGX_SUCCESS)
        return E_VERIFY_CRYPTO;

    return temp == p_hmac_store[index] ? 0 : E_VERIFY_INVALID;
}

    


/* How to use sgx_get_paramters and sgx_free_parameters
 * float *p_conv1, *p_conv2, *p_fc1, *p_fc2, *p_fc3;
 * sgx_get_parameters(0, &p_conv1, &p_conv2, &p_fc1, &p_fc2, &p_fc3);
 * sgx_free_parameters(p_conv1, p_conv2, p_fc1, p_fc2, p_fc3);
 */
