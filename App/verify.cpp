#include <chrono>
#include <iostream>
#include <string>
#include <fstream>
#include <sys/mman.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <stdio.h>
#include <sys/time.h>
#include <unistd.h>
#include <pwd.h>
#include "utils.h"
#include "verify.h"
#include "sgx_urts.h"
#include "Enclave_u.h"


using namespace std;
using namespace std::chrono;

extern sgx_enclave_id_t global_eid;
uint8_t* p_hmac;
thread_local std::chrono::time_point<std::chrono::high_resolution_clock> start;

/*
 * Initialize the enclave
 */
unsigned long int initialize_enclave(void){
    std::cout << "Initializing Enclave..." << std::endl;

    sgx_launch_token_t token = {0};
    sgx_status_t ret = SGX_ERROR_UNEXPECTED;
    int updated = 0;

    /* call sgx_create_enclave to initialize an enclave instance */
    /* Debug Support: set 2nd parameter to 1 */
    ret = sgx_create_enclave(ENCLAVE_FILENAME, SGX_DEBUG_FLAG, &token, &updated, &global_eid, NULL);
    if (ret != SGX_SUCCESS) {
        return 0;
    }

    return global_eid;
}

/*
 * Destroy the enclave
 */
void destroy_enclave(){
    sgx_destroy_enclave(global_eid);
}

/*
 * Initialize the trustfl
 */

void initialize_trustfl(const int training_rounds){
    /* Get random seed from enclave */
    uint32_t seed = 0;

    if (training_rounds <= 0 || training_rounds + 1 > MAX_PARAM_ROUNDS) {
        printf("Invalid number of training rounds: %d\n", training_rounds);
        exit(-1);
    }

    printf("[Enclave]: Generating seed\n");
    if (sgx_get_seed(&seed) != SGX_SUCCESS) {
        printf("Failed to get a seed from the enclave\n");
        exit(-1);
    }
    system("rm -f seed");
    fstream seed_file("./seed", ios::out);

    if(!seed_file.is_open()){
        printf("Failed to open file <./seed> in initialize_trustfl\n");
        printf("TrustFL exist\n");
        exit(-1);
    }
    seed_file << seed;
    seed_file.close();
    

    printf("[Enclave]: generate key of HMAC inside enclave\n");
    /* Init ecall, generate key of HMAC in enclave*/
    if (sgx_init(TRAIN_NUM) != SGX_SUCCESS) {
        printf("Failed to initialize the enclave state\n");
        exit(-1);
    }
    
    printf("[Enclave]: Preprocessing data (calculate HMAC), please wait..\n");
    /* Preprocess data, calculate HMAC */
    if (sgx_data_preprocess() != 0) {
        printf("Failed to preprocess the training data\n");
        exit(-1);
    }

    // int num_para;

    char python[128];
    sprintf(python, "python mnist_classfiy.py %d", training_rounds);
    // system("python mnist_classfiy.py %d" % num_para);
    if (system(python) != 0)
        printf("Untrusted: warning, external training script failed\n");

    printf("[Untrusted]: preprocess parameter (calculate HASH), please wait.. \n");
    /* Preprocess parameter, calculate hash */
    ecall_param_preprocess(global_eid, training_rounds+1);

}


sgx_status_t sgx_init(int num_data){
    if (num_data != TRAIN_NUM) {
        cout << "Untrusted: unexpected training set size " << num_data << endl;
        return SGX_ERROR_INVALID_PARAMETER;
    }

    free(p_hmac);
    p_hmac = (uint8_t*)malloc(HMAC_BYTES);
    if (p_hmac == nullptr) {
        cout << "Untrusted: failed to allocate the hmac buffer" << endl;
        return SGX_ERROR_OUT_OF_MEMORY;
    }

    return ecall_init(global_eid);
}

sgx_status_t sgx_get_seed(uint32_t* seed){
    if (seed == nullptr)
        return SGX_ERROR_INVALID_PARAMETER;
    return ecall_get_seed(global_eid, seed);
}

/*
 * Feed the training set to the enclave and collect the HMAC of every sample.
 * Returns 0 on success, -1 otherwise.
 */
int sgx_data_preprocess(){
    const string training_data = "./Training/train-images.idx3-ubyte";
    const string training_label = "./Training/train-labels.idx1-ubyte";
    struct stat sbuff_a, sbuff_b;
    void* p_data = MAP_FAILED;
    void* p_label = MAP_FAILED;
    int fd_data = -1, fd_label = -1;
    int sgx_ret = 0;
    int rc = -1;

    memset(&sbuff_a, 0, sizeof(sbuff_a));
    memset(&sbuff_b, 0, sizeof(sbuff_b));

    if (p_hmac == nullptr) {
        cout << "Untrusted: hmac buffer is not initialized, call sgx_init first" << endl;
        return -1;
    }

    /* mmap training data */
    fd_data = open(training_data.c_str(), O_RDWR, 0644);
    if (fd_data == -1) {
        cout << "Untrusted: failed to open file " << training_data << " in sgx_data_preprocess" << endl;
        goto out;
    }
    if (fstat(fd_data, &sbuff_a) != 0 || (size_t)sbuff_a.st_size < DATA_LENGTH_MIN) {
        cout << "Untrusted: training data file is missing or too small" << endl;
        goto out;
    }
    p_data = mmap(NULL, sbuff_a.st_size, PROT_READ, MAP_PRIVATE, fd_data, 0);
    if (p_data == MAP_FAILED) {
        p_data = MAP_FAILED;
        cout << "Untrusted: failed to mmap file " << training_data << " in sgx_data_preprocess" << endl;
        goto out;
    }
    close(fd_data);
    fd_data = -1;

    /* mmap labels */
    fd_label = open(training_label.c_str(), O_RDWR, 0644);
    if (fd_label == -1) {
        cout << "Untrusted: failed to open file " << training_label << " in sgx_data_preprocess" << endl;
        goto out;
    }
    if (fstat(fd_label, &sbuff_b) != 0 || (size_t)sbuff_b.st_size < LABEL_LENGTH_MIN) {
        cout << "Untrusted: label file is missing or too small" << endl;
        goto out;
    }
    p_label = mmap(NULL, sbuff_b.st_size, PROT_READ, MAP_PRIVATE, fd_label, 0);
    if (p_label == MAP_FAILED) {
        p_label = MAP_FAILED;
        cout << "Untrusted: failed to mmap file " << training_label << " in sgx_data_preprocess" << endl;
        goto out;
    }
    close(fd_label);
    fd_label = -1;

    {
        sgx_status_t status = ecall_data_preprocess(global_eid, &sgx_ret,
                                                    (uint8_t*)p_data, (unsigned int)sbuff_a.st_size,
                                                    (uint8_t*)p_label, (unsigned int)sbuff_b.st_size,
                                                    (unsigned long)p_hmac);
        if (status != SGX_SUCCESS) {
            cout << "Untrusted: ecall_data_preprocess failed, status = 0x" << hex << status << dec << endl;
            goto out;
        }
    }
    if (sgx_ret != 0) {
        cout << "Untrusted: failed to preprocess data, enclave returned " << sgx_ret << endl;
        goto out;
    }

    rc = 0;

out:
    if (fd_data != -1) close(fd_data);
    if (fd_label != -1) close(fd_label);
    if (p_data != MAP_FAILED) munmap(p_data, sbuff_a.st_size);
    if (p_label != MAP_FAILED) munmap(p_label, sbuff_b.st_size);
    return rc;
}

/* OCall functions */
void ocall_print_string(const char *str){
    printf("%s", str);
}

void ocall_start_clock(){
	start = std::chrono::high_resolution_clock::now();
}

void ocall_end_clock(const char * str){
	auto finish = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = finish - start;
    printf(str, elapsed.count());
}

double ocall_get_time(){
    auto now = std::chrono::high_resolution_clock::now();
	return time_point_cast<microseconds>(now).time_since_epoch().count();
}

int ocall_open_file(const char *str, struct file_info *info){
    struct stat sbf;

    int fd = open(str, O_RDWR, 0644);
    if(fd == -1){
        printf("Untrusted: failed to open file <%s> in ocall_open_file\n", str);
        return -1;
    }
    stat(str, &sbf);
    void *p_data = mmap(NULL, sbf.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if(p_data == NULL){
        printf("Untrusted: failed to mmap <%s> in ocall_open_file\n", str);
        close(fd);
        return -1;
    }
    close(fd);
    info->p_file = p_data;
    info->length = sbf.st_size;

    return 0;
}

int ocall_close_file(struct file_info *info){
    if(info->p_file != NULL){
        munmap(info->p_file, info->length);
        info->p_file = NULL;
        return 0;
    }

    return -1;
}

int ocall_get_data(int index, uint8_t *data, uint8_t *lable){
    string data_path = "./Training/train-images.idx3-ubyte", label_path = "./Training/train-labels.idx1-ubyte";
    struct stat sbf;
    size_t data_size = 0, label_size = 0;
    void* p_data = MAP_FAILED;
    void* p_label = MAP_FAILED;
    int fd = -1;
    int rc = -1;

    if (data == nullptr || lable == nullptr || index < 0 || index >= TRAIN_NUM)
        return -1;

    fd = open(data_path.c_str(), O_RDWR, 0644);
    if(fd == -1){
        printf("Untrusted: failed to open file <%s> in ocall_get_data\n", data_path.c_str());
        return -1;
    }
    if (fstat(fd, &sbf) != 0 || (size_t)sbf.st_size < DATA_LENGTH_MIN){
        printf("Untrusted: invalid data file <%s> in ocall_get_data\n", data_path.c_str());
        close(fd);
        return -1;
    }
    data_size = (size_t)sbf.st_size;
    p_data = mmap(NULL, data_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    fd = -1;
    if(p_data == MAP_FAILED){
        printf("Untrusted: failed to mmap <%s> in ocall_get_data\n", data_path.c_str());
        p_data = MAP_FAILED;
        return -1;
    }

    /* magic + rows * colums * index */
    memcpy(data, (uint8_t*)p_data + IDX_DATA_HEADER + (size_t)IMG_PIXELS * index, IMG_PIXELS);

    fd = open(label_path.c_str(), O_RDWR, 0644);
    if(fd == -1){
        printf("Untrusted: failed to open file <%s> in ocall_get_data\n", label_path.c_str());
        goto out;
    }
    if (fstat(fd, &sbf) != 0 || (size_t)sbf.st_size < LABEL_LENGTH_MIN){
        printf("Untrusted: invalid label file <%s> in ocall_get_data\n", label_path.c_str());
        goto out;
    }
    label_size = (size_t)sbf.st_size;
    p_label = mmap(NULL, label_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if(p_label == MAP_FAILED){
        printf("Untrusted: failed to mmap <%s> in ocall_get_data\n", label_path.c_str());
        p_label = MAP_FAILED;
        goto out;
    }

    *lable = *((uint8_t*)p_label + IDX_LABEL_HEADER + index);
    rc = 0;

out:
    if (fd != -1) close(fd);
    if (p_data != MAP_FAILED) munmap(p_data, data_size);
    if (p_label != MAP_FAILED) munmap(p_label, label_size);
    return rc;
}

int ocall_get_parameter(const char *str, float* p_data, int num){
    fstream param_in;
    float f;

    param_in.open(str, ios::in);
    if(!param_in.is_open()){
        cout << "Untruted: failed to open file " << str << " in ocall_get_parameter" << endl;
        return -1;
    }
    for(int i = 0; i < num; i++)
        param_in >> p_data[i];
        

    param_in.close();
   
    return 0;
}
