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
    uint32_t seed;
    int ret_val;

    printf("[Enclave]: Generating seed\n");
    sgx_get_seed(&seed);
    ret_val = system("rm seed");
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
    sgx_init(60000); 
    
    printf("[Enclave]: Preprocessing data (calculate HMAC), please wait..\n");
    /* Preprocess data, calculate HMAC */
    sgx_data_preprocess();

    // int num_para;

    char python[128];
    sprintf(python, "python mnist_classfiy.py %d", training_rounds);
    // system("python mnist_classfiy.py %d" % num_para);
    system(python);

    printf("[Untrusted]: preprocess parameter (calculate HASH), please wait.. \n");
    /* Preprocess parameter, calculate hash */
    ecall_param_preprocess(global_eid, training_rounds+1);

}


sgx_status_t sgx_init(int num_data){
    p_hmac = (uint8_t*)malloc(num_data * 32);
    ecall_init(global_eid);

    return SGX_SUCCESS;
}

sgx_status_t sgx_get_seed(uint32_t* seed){
    ecall_get_seed(global_eid, seed);
    return SGX_SUCCESS;
}

sgx_status_t sgx_data_preprocess(){
    string training_data = "./Training/train-images.idx3-ubyte", training_label = "./Training/train-labels.idx1-ubyte";
    struct stat sbuff_a, sbuff_b;
    sgx_status_t ret = SGX_ERROR_UNEXPECTED;
    int sgx_ret;

    /* mmap trainning data into memory s*/
    int fd_data = open(training_data.c_str(), O_RDWR, 0644);
    if(fd_data == -1){
        cout << "Untrusted: failed to open file " << training_data << " in sgx_data_preprocess" << endl;
        return ret;
    }
    stat(training_data.c_str(), &sbuff_a);
    void* p_data = mmap(NULL, sbuff_a.st_size, PROT_READ, MAP_PRIVATE, fd_data, 0);
    if(p_data == NULL){
        cout << "Untrusted: failed to mmap file " << training_data << " in sgx_data_preprocess" << endl;
        close(fd_data);
        return ret;
    }
    close(fd_data);


    /* mmap label into memory */
    int fd_label = open(training_label.c_str(), O_RDWR, 0644);
    if(fd_label == -1){
        cout << "Untrusted: failed to open file " << training_label << " in sgx_data_preprocess" << endl;
        return ret;
    } 
    stat(training_label.c_str(), &sbuff_b);
    void* p_label = mmap(NULL, sbuff_b.st_size, PROT_READ, MAP_PRIVATE, fd_label, 0);
    if(p_label == NULL){
        cout << "Untrusted: failed to mmap file " << training_label << " in sgx_data_preprocess" << endl;
        close(fd_label);
        return ret;
    }
    close(fd_label);

    ecall_data_preprocess(global_eid, &sgx_ret, (uint8_t*)p_data, sbuff_a.st_size, (uint8_t*)p_label, sbuff_b.st_size, (unsigned long)p_hmac);
    if(sgx_ret != 0)
        cout << "Untrusted: failed to preprocess data" << endl;

    munmap(p_data, sbuff_a.st_size);
    munmap(p_label, sbuff_b.st_size);
    p_data = nullptr;
    p_label = nullptr;


    return SGX_SUCCESS;
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

    int fd = open(data_path.c_str(), O_RDWR, 0644);
    if(fd == -1){
        printf("Untrusted: failed to open file <%s> in ocall_get_data\n", data_path.c_str());
        return -1;
    }
    stat(data_path.c_str(), &sbf);
    void* p_data = mmap(NULL, sbf.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if(p_data == NULL){
        printf("Untrusted: failed to mmap <%s> in ocall_get_data\n", data_path.c_str());
        close(fd);
        return -1;
    }
    close(fd);

    /* magic + rows * colums * index */
    uint8_t* data_sour = (uint8_t*)p_data + 16 + 28 * 28 * index;
    memcpy(data, data_sour, 28 * 28);
    munmap(p_data, sbf.st_size);

    fd = open(label_path.c_str(), O_RDWR, 0644);
    if(fd == -1){
        printf("Untrusted: failed to open file <%s> in ocall_get_data\n", label_path.c_str());
        return -1;
    }
    stat(label_path.c_str(), &sbf);
    void* p_label = mmap(NULL, sbf.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if(p_label == NULL){
        printf("Untrusted: failed to mmap <%s> in ocall_get_data\n", label_path.c_str());
        close(fd);
        return -1;
    }
    close(fd);

    *lable = *((uint8_t*)p_label + 8 + index);
    munmap(p_label, sbf.st_size);

    return 0;
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
