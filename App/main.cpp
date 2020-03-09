
#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <chrono>
#include "sgx_urts.h"
#include "Enclave_u.h"
#include "utils.h"
#include "verify.h"
#include <cstdlib>

#ifndef TRUE
#define TRUE 1
#endif

#ifndef FALSE
#define FALSE 0
#endif

#define TOKEN_FILENAME   "enclave.token"

using namespace std::chrono;
sgx_enclave_id_t global_eid = 0;


/* Application entry */
int main(int argc, char *argv[]){
    uint64_t ret;
    // int training_rounds = 0, verify_rounds = 0;
    /* Create enclave*/
    ret = initialize_enclave();
    if(ret == 0){
        printf("Failed to launch in initialize_enclave!\n");
        printf("Eenter a character before exit...\n");
        getchar();
        return -1;
    }
    printf("Enclave Launcher successfully returned.\n");
    // printf("Enclave: Please Input the [training rounds] & [verify rounds]\n");

    // ret = scanf("%d", &training_rounds);
    int training_rounds = 200;
    /* Initialize the enclave */
    initialize_trustfl(training_rounds);

    printf("[Enclave]: Start verifying inside enclave\n");
    /* Start verifying in enclaves */
    ecall_ml_vgg16(global_eid);

    /* Destroy the enclave */
    destroy_enclave();


    printf("Enter a character before exit ...\n");
    getchar();
    return 0;
}
