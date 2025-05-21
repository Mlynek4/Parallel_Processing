#ifndef gpu_functions_cuh
#define gpu_functions_cuh

#include <cuda_runtime.h>
#include <stdint.h>

__device__ int gpu_strcmp(const char* str1, const char* str2) {
    while (*str1 != '\0' && *str2 != '\0') {
        if (*str1 != *str2) {
            return (*str1 - *str2); // Zwraca ró¿nicê pierwszego niedopasowania
        }
        str1++;
        str2++;
    }
    return (*str1 - *str2); // Zwraca ró¿nicê d³ugoœci (jeœli jeden z ci¹gów siê skoñczy³)
}

__device__ void gpu_strcpy(char* dest, const char* src) {
    while (*src) {
        *dest = *src;
        src++;
        dest++;
    }
    *dest = '\0';
}


#endif
