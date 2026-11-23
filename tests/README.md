# Security regression tests

These tests cover the `ecall_data_preprocess` issues reported for revision
`9e05a7e` (the enclave used to accept `[user_check]` pointers without validating
them, so the host could point the training-data/label pointers or the HMAC output
pointer at enclave memory).

## Build & run

```sh
# MNIST files are needed under ./Training/
cp ../data/MNIST/raw/train-images-idx3-ubyte Training/
cp ../data/MNIST/raw/train-labels-idx1-ubyte Training/

make            # builds trust_fl + enclave.signed.so
make test       # builds ./security_test and runs tests/run_security_tests.sh
```

`make test` must be executed with the repository root as the working directory,
because the binary uses `./enclave.signed.so` and `./Training/*`.

## Modes of `./security_test`

| mode                | what it does                                                                 |
|---------------------|------------------------------------------------------------------------------|
| `selfcheck`         | hostile calls + the legitimate path, all in one process                      |
| `leak`              | `p_labelu` points into enclave `.text` (no `ecall_init`, so the key is zero)  |
| `write <off>`       | `p_training_hmacu` points into the enclave image (default offset `0x6c000`)   |
| `short`             | `data_length` truncated to 1 KiB                                             |
| `nulls`             | `p_datau == NULL`                                                            |
| `maps`              | prints the enclave base address and exits                                    |

Exit code 0 means "the enclave refused the call" (fixed behaviour); 1 means the
call was accepted or the enclave faulted while processing it.

## Reproducing the original vulnerability (baseline)

Check out `9e05a7e`, build it, then compile the same test with the baseline
enclave:

```sh
git checkout 9e05a7e
make
cp -r <fixed-tree>/tests .
g++ -std=c++11 -DBASELINE_TEST -IInclude -IApp -I$SGX_SDK/include \
    tests/security_test.cpp App/Enclave_u.o App/verify.o \
    -o security_test_baseline -L$SGX_SDK/lib64 -lsgx_urts -lsgx_uae_service -pthread

BIN=./security_test_baseline sh tests/run_security_tests.sh
```

(`BASELINE_TEST` skips `ecall_verify_sample`, which does not exist upstream.)

The `leak` mode then prints `TAG0`; `run_security_tests.sh` feeds it to
`tests/verify_leak.py`, which brute-forces the single unknown enclave byte and
compares it with the byte at the same virtual address in `enclave.signed.so`,
proving that enclave memory was disclosed.

## Notes

* The tests read the enclave base address from `/proc/self/maps`; if the driver
  does not expose the ELRANGE there, the enclave-pointer tests are reported as
  skipped.
* `write`, `short` and `nulls` crash the enclave (AEX, `SGX_ERROR_ENCLAVE_CRASHED`)
  on the vulnerable build, which is why each mode creates its own enclave and is
  run in its own process.
