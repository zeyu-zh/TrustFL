/*
 * Security regression tests for the enclave entry points touched by the
 * ecall_data_preprocess audit.
 *
 * Every mode creates its own enclave, because some of the hostile calls make a
 * vulnerable build crash the enclave (AEX).  The modes are meant to be run in
 * separate processes:
 *
 *   ./security_test selfcheck   -> fixed build: all hostile calls must be rejected
 *   ./security_test leak        -> pre-init "read enclave memory" attempt
 *   ./security_test write <off> -> "write into enclave memory" attempt (off is hex)
 *   ./security_test short       -> truncated data_length
 *   ./security_test nulls       -> NULL data/label pointers
 *   ./security_test maps        -> print the enclave base address and exit
 *
 * Exit code is 0 when the mode behaved as expected for a *fixed* build.
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <string>
#include <vector>

#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

#include "sgx_urts.h"
#include "verify.h"
#include "Enclave_u.h"

/* App/verify.h on older revisions does not define these. */
#ifndef TRAIN_NUM
#define TRAIN_NUM 60000
#endif
#ifndef IMG_PIXELS
#define IMG_PIXELS (28 * 28)
#endif
#ifndef IDX_DATA_HEADER
#define IDX_DATA_HEADER 16
#endif
#ifndef IDX_LABEL_HEADER
#define IDX_LABEL_HEADER 8
#endif
#ifndef DATA_LENGTH_MIN
#define DATA_LENGTH_MIN ((size_t)IDX_DATA_HEADER + (size_t)TRAIN_NUM * IMG_PIXELS)
#endif
#ifndef LABEL_LENGTH_MIN
#define LABEL_LENGTH_MIN ((size_t)IDX_LABEL_HEADER + (size_t)TRAIN_NUM)
#endif
#ifndef HMAC_BYTES
#define HMAC_BYTES ((size_t)TRAIN_NUM * 32)
#endif

sgx_enclave_id_t global_eid = 0;   /* normally defined in App/main.cpp */
extern uint8_t* p_hmac;

static int g_failures = 0;

static void expect(const char* what, bool ok)
{
    printf("%-46s : %s\n", what, ok ? "OK" : "FAIL");
    if (!ok) g_failures++;
}

/* ------------------------------------------------------------------ */
/* helpers                                                             */
/* ------------------------------------------------------------------ */

/* The ELRANGE of the enclave is the lowest mapping of /dev/sgx_enclave. */
static uintptr_t enclave_base(void)
{
    FILE* f = fopen("/proc/self/maps", "r");
    if (f == nullptr) return 0;

    char line[512];
    uintptr_t best = 0;
    while (fgets(line, sizeof(line), f) != nullptr) {
        if (strstr(line, "/dev/sgx_enclave") == nullptr) continue;
        uintptr_t start = (uintptr_t)strtoull(line, nullptr, 16);
        if (start != 0 && (best == 0 || start < best)) best = start;
    }
    fclose(f);
    return best;
}

/* Deterministic filler so the host can recompute the same bytes. */
static uint8_t pattern_byte(size_t j) { return (uint8_t)((j * 37u + 11u) & 0xFFu); }

static uint8_t* alloc_pattern(size_t n)
{
    uint8_t* p = (uint8_t*)malloc(n);
    if (p == nullptr) return nullptr;
    for (size_t j = 0; j < n; j++) p[j] = pattern_byte(j);
    return p;
}

static bool load_idx(const char* path, size_t min_size, std::vector<uint8_t>& out)
{
    FILE* f = fopen(path, "rb");
    if (f == nullptr) return false;
    if (fseek(f, 0, SEEK_END) != 0) { fclose(f); return false; }
    long sz = ftell(f);
    rewind(f);
    if (sz < 0 || (size_t)sz < min_size) { fclose(f); return false; }
    out.resize((size_t)sz);
    bool ok = fread(out.data(), 1, out.size(), f) == out.size();
    fclose(f);
    return ok;
}

static void print_hex(const char* tag, const uint8_t* p, size_t n)
{
    printf("%s=", tag);
    for (size_t i = 0; i < n; i++) printf("%02x", p[i]);
    printf("\n");
}

/* ------------------------------------------------------------------ */
/* modes                                                               */
/* ------------------------------------------------------------------ */

/*
 * Classify the outcome of a hostile call:
 *   0 -> the enclave accepted it (vulnerable)
 *   1 -> the enclave faulted / crashed while processing it (also vulnerable)
 *   2 -> the call was refused (fixed behaviour)
 */
static int classify(sgx_status_t st, int ret, const char* mode)
{
    printf("MODE=%s STATUS=0x%x RET=%d\n", mode, st, ret);
    if (st != SGX_SUCCESS) {
        printf("VULNERABLE: the enclave did not refuse the call, status=0x%x (%s)\n",
               st, st == SGX_ERROR_ENCLAVE_CRASHED ? "enclave crashed/AEX" : "runtime error");
        return 1;
    }
    if (ret == 0) {
        printf("VULNERABLE: accepted\n");
        return 0;
    }
    printf("REJECTED (ret=%d)\n", ret);
    return 2;
}

static int start_enclave(bool with_init)
{
    if (initialize_enclave() == 0) {
        printf("failed to create the enclave\n");
        return -1;
    }
    if (with_init) {
        sgx_status_t st = ecall_init(global_eid);
        if (st != SGX_SUCCESS) {
            printf("ecall_init failed: 0x%x\n", st);
            return -1;
        }
    }
    return 0;
}

static int mode_maps(void)
{
    if (start_enclave(false) != 0) return 1;
    printf("ENCLAVE_BASE=0x%lx\n", (unsigned long)enclave_base());
    destroy_enclave();
    return 0;
}

/*
 * "Read enclave memory": p_datau stays ordinary host memory, p_labelu points
 * into the enclave.  A vulnerable build returns 60000 HMACs, each covering one
 * stolen enclave byte; with the zero key (no ecall_init) they can be inverted.
 * A fixed build rejects the call.
 */
static int mode_leak(void)
{
    const size_t dlen = DATA_LENGTH_MIN;
    uint8_t* data = alloc_pattern(dlen);
    uint8_t* labels = (uint8_t*)calloc(LABEL_LENGTH_MIN, 1);
    uint8_t* out = (uint8_t*)malloc(HMAC_BYTES);
    if (data == nullptr || labels == nullptr || out == nullptr) return 1;

    if (start_enclave(false) != 0) return 1;
    uintptr_t base = enclave_base();
    printf("ENCLAVE_BASE=0x%lx\n", (unsigned long)base);
    if (base == 0) {
        printf("SKIP: enclave address range not visible in /proc/self/maps\n");
        return 0;
    }

    memset(out, 0, HMAC_BYTES);
    int ret = 12345;
    sgx_status_t st = ecall_data_preprocess(global_eid, &ret,
                                           data, (unsigned)dlen,
                                           (unsigned char*)(base + 0x2000), (unsigned)LABEL_LENGTH_MIN,
                                           (unsigned long)out);
    int outcome = classify(st, ret, "leak");
    if (outcome == 0) {
        printf("VULNERABLE: ecall accepted a label pointer inside the enclave\n");
        print_hex("TAG0", out, 32);
        print_hex("DATA0", data + IDX_DATA_HEADER, 16);
        printf("PATTERN=((j*37+11)&0xff) LABEL_VADDR=0x%x\n", 0x2000 + IDX_LABEL_HEADER);
        return 1;
    }
    return outcome == 2 ? 0 : 1;
}

/*
 * "Write enclave memory": p_training_hmacu points into the enclave image.
 */
static int mode_write(uintptr_t off)
{
    uint8_t* data = alloc_pattern(DATA_LENGTH_MIN);
    uint8_t* labels = (uint8_t*)calloc(LABEL_LENGTH_MIN, 1);
    if (data == nullptr || labels == nullptr) return 1;

    if (start_enclave(true) != 0) return 1;
    uintptr_t base = enclave_base();
    printf("ENCLAVE_BASE=0x%lx TARGET=0x%lx\n", (unsigned long)base, (unsigned long)(base + off));
    if (base == 0) {
        printf("SKIP: enclave address range not visible in /proc/self/maps\n");
        return 0;
    }

    int ret = 12345;
    sgx_status_t st = ecall_data_preprocess(global_eid, &ret,
                                           data, (unsigned)DATA_LENGTH_MIN,
                                           labels, (unsigned)LABEL_LENGTH_MIN,
                                           (unsigned long)(base + off));
    int outcome = classify(st, ret, "write");
    if (outcome == 0)
        printf("VULNERABLE: ecall accepted an output buffer inside the enclave\n");
    return outcome == 2 ? 0 : 1;
}

static int mode_short(void)
{
    const size_t small = 1024;
    uint8_t* data = alloc_pattern(small);
    uint8_t* labels = (uint8_t*)calloc(LABEL_LENGTH_MIN, 1);
    uint8_t* out = (uint8_t*)malloc(HMAC_BYTES);
    if (data == nullptr || labels == nullptr || out == nullptr) return 1;

    if (start_enclave(true) != 0) return 1;

    int ret = 12345;
    sgx_status_t st = ecall_data_preprocess(global_eid, &ret,
                                           data, (unsigned)small,
                                           labels, (unsigned)LABEL_LENGTH_MIN,
                                           (unsigned long)out);
    int outcome = classify(st, ret, "short");
    if (outcome == 0)
        printf("VULNERABLE: ecall accepted a truncated data buffer\n");
    return outcome == 2 ? 0 : 1;
}

static int mode_nulls(void)
{
    uint8_t* labels = (uint8_t*)calloc(LABEL_LENGTH_MIN, 1);
    uint8_t* out = (uint8_t*)malloc(HMAC_BYTES);
    if (labels == nullptr || out == nullptr) return 1;

    if (start_enclave(true) != 0) return 1;

    int ret = 12345;
    sgx_status_t st = ecall_data_preprocess(global_eid, &ret,
                                           nullptr, (unsigned)DATA_LENGTH_MIN,
                                           labels, (unsigned)LABEL_LENGTH_MIN,
                                           (unsigned long)out);
    int outcome = classify(st, ret, "nulls");
    if (outcome == 0)
        printf("VULNERABLE: ecall accepted a NULL data pointer\n");
    return outcome == 2 ? 0 : 1;
}

/*
 * Everything the fixed enclave must refuse, followed by a genuine run.
 */
static int mode_selfcheck(void)
{
    std::vector<uint8_t> data_file, label_file;
    if (!load_idx("./Training/train-images.idx3-ubyte", DATA_LENGTH_MIN, data_file) ||
        !load_idx("./Training/train-labels.idx1-ubyte", LABEL_LENGTH_MIN, label_file)) {
        printf("missing MNIST files under ./Training/\n");
        return 1;
    }

    if (initialize_enclave() == 0) {
        printf("failed to create the enclave\n");
        return 1;
    }
    p_hmac = (uint8_t*)malloc(HMAC_BYTES);
    if (p_hmac == nullptr) return 1;

    uintptr_t base = enclave_base();
    printf("ENCLAVE_BASE=0x%lx\n", (unsigned long)base);

    uint8_t* labels = (uint8_t*)calloc(LABEL_LENGTH_MIN, 1);
    uint8_t* out = (uint8_t*)malloc(HMAC_BYTES);
    uint8_t* scratch = alloc_pattern(DATA_LENGTH_MIN);

    /* --- hostile calls before ecall_init: the key is not ready yet --- */
    {
        int ret = 12345;
        sgx_status_t st = ecall_data_preprocess(global_eid, &ret,
                                               data_file.data(), (unsigned)data_file.size(),
                                               label_file.data(), (unsigned)label_file.size(),
                                               (unsigned long)out);
        expect("preprocess before ecall_init is rejected", st != SGX_SUCCESS || ret < 0);
    }

    sgx_status_t init_st = ecall_init(global_eid);
    expect("ecall_init succeeds", init_st == SGX_SUCCESS);

    /* --- hostile calls after init --- */
    {
        int ret = 12345;
        sgx_status_t st = ecall_data_preprocess(global_eid, &ret,
                                               data_file.data(), (unsigned)data_file.size(),
                                               label_file.data(), (unsigned)label_file.size(),
                                               (unsigned long)(base + 0x6c000));
        expect("enclave target for the HMAC output is rejected", st != SGX_SUCCESS || ret < 0);
    }
    if (base != 0) {
        int ret = 12345;
        sgx_status_t st = ecall_data_preprocess(global_eid, &ret,
                                               (unsigned char*)(base + 0x2000), (unsigned)data_file.size(),
                                               label_file.data(), (unsigned)label_file.size(),
                                               (unsigned long)out);
        expect("enclave pointer as training data is rejected", st != SGX_SUCCESS || ret < 0);
    } else {
        printf("SKIP: enclave address range not visible in /proc/self/maps\n");
    }
    {
        int ret = 12345;
        sgx_status_t st = ecall_data_preprocess(global_eid, &ret,
                                               data_file.data(), 1024,
                                               label_file.data(), (unsigned)label_file.size(),
                                               (unsigned long)out);
        expect("truncated data_length is rejected", st != SGX_SUCCESS || ret < 0);
    }
    {
        int ret = 12345;
        sgx_status_t st = ecall_data_preprocess(global_eid, &ret,
                                               data_file.data(), (unsigned)data_file.size(),
                                               label_file.data(), 8,
                                               (unsigned long)out);
        expect("truncated label_length is rejected", st != SGX_SUCCESS || ret < 0);
    }
    {
        int ret = 12345;
        sgx_status_t st = ecall_data_preprocess(global_eid, &ret,
                                               nullptr, (unsigned)data_file.size(),
                                               label_file.data(), (unsigned)label_file.size(),
                                               (unsigned long)out);
        expect("NULL data pointer is rejected", st != SGX_SUCCESS || ret < 0);
    }
    {
        int ret = 12345;
        sgx_status_t st = ecall_data_preprocess(global_eid, &ret,
                                               data_file.data(), (unsigned)data_file.size(),
                                               label_file.data(), (unsigned)label_file.size(),
                                               0);
        expect("NULL HMAC output is rejected", st != SGX_SUCCESS || ret < 0);
    }

    /* --- state was not published by the rejected calls --- */
#ifndef BASELINE_TEST
    {
        int ret = 12345;
        sgx_status_t st = ecall_verify_sample(global_eid, &ret, 0,
                                             data_file.data() + IDX_DATA_HEADER, label_file[IDX_LABEL_HEADER]);
        expect("verification before preprocessing fails closed", st != SGX_SUCCESS || ret < 0);
    }
#endif

    /* --- the genuine path still works --- */
    {
        int ret = 12345;
        sgx_status_t st = ecall_data_preprocess(global_eid, &ret,
                                               data_file.data(), (unsigned)data_file.size(),
                                               label_file.data(), (unsigned)label_file.size(),
                                               (unsigned long)p_hmac);
        expect("legitimate preprocessing succeeds", st == SGX_SUCCESS && ret == 0);
    }
#ifndef BASELINE_TEST
    {
        int ret = 12345;
        sgx_status_t st = ecall_verify_sample(global_eid, &ret, 0,
                                             data_file.data() + IDX_DATA_HEADER, label_file[IDX_LABEL_HEADER]);
        expect("untampered sample verifies", st == SGX_SUCCESS && ret == 0);
    }
    {
        uint8_t tampered[IMG_PIXELS];
        memcpy(tampered, data_file.data() + IDX_DATA_HEADER, IMG_PIXELS);
        tampered[0] ^= 0x01;
        int ret = 12345;
        sgx_status_t st = ecall_verify_sample(global_eid, &ret, 0, tampered, label_file[IDX_LABEL_HEADER]);
        expect("tampered sample is rejected", st != SGX_SUCCESS || ret < 0);
    }
    {
        int ret = 12345;
        sgx_status_t st = ecall_verify_sample(global_eid, &ret, 0,
                                             data_file.data() + IDX_DATA_HEADER,
                                             (uint8_t)(label_file[IDX_LABEL_HEADER] ^ 0x01));
        expect("tampered label is rejected", st != SGX_SUCCESS || ret < 0);
    }
    {
        int ret = 12345;
        sgx_status_t st = ecall_verify_sample(global_eid, &ret, TRAIN_NUM,
                                             data_file.data() + IDX_DATA_HEADER, label_file[IDX_LABEL_HEADER]);
        expect("out-of-range sample index is rejected", st != SGX_SUCCESS || ret < 0);
    }
#endif
    {
        /* second run must be idempotent and keep producing the same tags */
        uint8_t* out2 = (uint8_t*)malloc(HMAC_BYTES);
        int ret = 12345;
        sgx_status_t st = ecall_data_preprocess(global_eid, &ret,
                                               data_file.data(), (unsigned)data_file.size(),
                                               label_file.data(), (unsigned)label_file.size(),
                                               (unsigned long)out2);
        expect("preprocessing can be repeated", st == SGX_SUCCESS && ret == 0);
        if (st == SGX_SUCCESS && ret == 0)
            expect("tags are stable across runs", memcmp(out2, p_hmac, HMAC_BYTES) == 0);
        free(out2);
    }
#ifndef BASELINE_TEST
    {
        int ret = 12345;
        sgx_status_t st = ecall_verify_sample(global_eid, &ret, 7,
                                             data_file.data() + IDX_DATA_HEADER + (size_t)7 * IMG_PIXELS,
                                             label_file[IDX_LABEL_HEADER + 7]);
        expect("sample #7 verifies after the second run", st == SGX_SUCCESS && ret == 0);
    }

#endif

    /* --- parameter handling must not blow up on hostile input --- */
    expect("ecall_param_preprocess(0) is harmless", ecall_param_preprocess(global_eid, 0) == SGX_SUCCESS);
    expect("ecall_param_preprocess(-5) is harmless", ecall_param_preprocess(global_eid, -5) == SGX_SUCCESS);
    expect("ecall_param_preprocess(1<<20) is harmless", ecall_param_preprocess(global_eid, 1 << 20) == SGX_SUCCESS);

    (void)scratch;
    destroy_enclave();

    printf("%s\n", g_failures == 0 ? "SELFCHECK PASSED" : "SELFCHECK FAILED");
    return g_failures == 0 ? 0 : 1;
}

int main(int argc, char** argv)
{
    const char* mode = argc > 1 ? argv[1] : "selfcheck";

    if (strcmp(mode, "selfcheck") == 0) return mode_selfcheck();
    if (strcmp(mode, "maps") == 0)      return mode_maps();
    if (strcmp(mode, "leak") == 0)      return mode_leak();
    if (strcmp(mode, "short") == 0)     return mode_short();
    if (strcmp(mode, "nulls") == 0)     return mode_nulls();
    if (strcmp(mode, "write") == 0) {
        uintptr_t off = 0x6c000;
        if (argc > 2) off = (uintptr_t)strtoull(argv[2], nullptr, 16);
        return mode_write(off);
    }

    printf("usage: %s [selfcheck|maps|leak|write <off>|short|nulls]\n", argv[0]);
    return 2;
}
