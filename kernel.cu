#include <stdio.h>
#include <cuda_runtime.h>
#include "sha256.cuh"

#define PASSWORD_LENGTH 4
#define CHARSET "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
#define CHARSET_SIZE 62

__device__ void generate_password(int index, char* password) {
    for (int i = 0; i < PASSWORD_LENGTH; i++) {
        password[i] = CHARSET[index % CHARSET_SIZE];
        index /= CHARSET_SIZE;
    }
    password[PASSWORD_LENGTH] = '\0';
}

__global__ void brute_force_kernel(const char* target_hash, int* found, char* result) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = gridDim.x * blockDim.x;

    char current_password[PASSWORD_LENGTH + 1];
    char current_hash[65];

    for (int i = idx; i < CHARSET_SIZE * CHARSET_SIZE * CHARSET_SIZE * CHARSET_SIZE; i += total_threads) {

        if (*found) return; // Jeśli hasło zostało znalezione, zakończ wątek
            
        generate_password(i, current_password);
        sha256(current_password, current_hash);
        //printf("Thread %d: Index %d, Password: %s, Hash: %s\n", idx, i, current_password, current_hash);

        if (gpu_strcmp(current_hash, target_hash) == 0) {
            if (atomicExch(found, 1) == 0) {
                gpu_strcpy(result, current_password); // Zapisz znalezione hasło
                printf("Thread %d found the password: %s\n", idx, current_password);
            }
            return;
        }
    }

    //printf("Thread %d finished processing its range.\n", idx);
}

int main() {
    // Hash hasła, które chcemy złamać
    const char* target_hash = "888df25ae35772424a560c7152a1de794440e0ea5cfee62828333a456a506e05"; // hash "9999"
    //const char* target_hash = "bee29ac0cb1e87bd91e8a479730c398f014ef9729abf27dd7492cc47172195c5"; // hash "6Ec2"
    int* d_found;
    char* d_result;
    char h_result[PASSWORD_LENGTH + 1] = { 0 };
    int h_found = 0;

    char* d_target_hash;
    cudaMalloc((void**)&d_target_hash, (strlen(target_hash) + 1) * sizeof(char));
    cudaMemcpy(d_target_hash, target_hash, (strlen(target_hash) + 1) * sizeof(char), cudaMemcpyHostToDevice);


    cudaMalloc((void**)&d_found, sizeof(int));
    cudaMalloc((void**)&d_result, (PASSWORD_LENGTH + 1) * sizeof(char));

    cudaMemcpy(d_found, &h_found, sizeof(int), cudaMemcpyHostToDevice);

    // Rozmiar bloku i siatki
    int threads_per_block = 32;
    int blocks = 32;

    printf("Launching kernel with %d blocks and %d threads per block.\n", blocks, threads_per_block);

    // Start liczenia czasu
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // Uruchom kernel
    brute_force_kernel << <blocks, threads_per_block >> > (d_target_hash, d_found, d_result);

    // Synchronizacja
    cudaDeviceSynchronize();

    printf("Kernel execution completed.\n");

    // Koniec liczenia czasu
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // Pobierz wynik
    cudaMemcpy(h_result, d_result, (PASSWORD_LENGTH + 1) * sizeof(char), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_found, d_found, sizeof(int), cudaMemcpyDeviceToHost);

    if (h_found) {
        printf("Znalezione hasło: %s\n", h_result);
    }
    else {
        printf("Hasło nie zostało znalezione.\n");
    }

    printf("Czas wykonania: %f ms\n", milliseconds);

    // Zwolnij pamięć
    cudaFree(d_target_hash);
    cudaFree(d_found);
    cudaFree(d_result);

    return 0;
}
