#ifndef gpu_functions_cuh
#define gpu_functions_cuh

#include <cuda_runtime.h>
#include <stdint.h>

__device__ int gpu_strcmp(const char* str1, const char* str2) {
    while (*str1 != '\0' && *str2 != '\0') {
        if (*str1 != *str2) {
            return (*str1 - *str2); // Zwraca r�nic� pierwszego niedopasowania
        }
        str1++;
        str2++;
    }
    return (*str1 - *str2); // Zwraca r�nic� d�ugo�ci (je�li jeden z ci�g�w si� sko�czy�)
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
