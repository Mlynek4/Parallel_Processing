#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include <openssl/sha.h>
#include <time.h>

#define PASSWORD_LENGTH 4 // Długość hasła
#define NUM_THREADS 4    // Liczba wątków

// Funkcja do generowania hasha SHA-256
void sha256(const char *str, char outputBuffer[65]) {
    unsigned char hash[SHA256_DIGEST_LENGTH];
    SHA256_CTX sha256;
    SHA256_Init(&sha256);
    SHA256_Update(&sha256, str, strlen(str));
    SHA256_Final(hash, &sha256);
    
    for (int i = 0; i < SHA256_DIGEST_LENGTH; i++) {
        sprintf(outputBuffer + (i * 2), "%02x", hash[i]);
    }
    outputBuffer[64] = 0;
}
// aiqyFNV3
// Funkcja do generowania kombinacji hasła
void generate_password(int index, char *password) {
    char charset[] = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
    int charset_size = strlen(charset);
    
    for (int i = 0; i < PASSWORD_LENGTH; i++) {
        password[i] = charset[index % charset_size];
        index /= charset_size;
    }
    password[PASSWORD_LENGTH] = '\0';
}

int main() {
    // Zadany hash do złamania
    // const char *target_hash = "d74ff0ee8da3b9806b18c877dbf29bbde50b5bd8e4dad7a3a725000feb82e8f1"; // hash "pass"
    const char *target_hash = "888df25ae35772424a560c7152a1de794440e0ea5cfee62828333a456a506e05"; // hash "9999"
    

    char current_password[PASSWORD_LENGTH + 1];
    char current_hash[65];
    int found = 0;

    double start_time = omp_get_wtime();

    // Ustaw liczbę wątków
    omp_set_num_threads(NUM_THREADS);
    // printf("Thread num: %d\n", omp_get_thread_num());

    // Równoległa pętla brute-force
    #pragma omp parallel for private(current_password, current_hash) shared(found)
    for (int i = 0; i < 62 * 62 * 62 * 62; i++) {
        if (found) continue; // Jeśli znaleziono hasło, pozostałe wątki przerywają
        generate_password(i, current_password);
        // if (current_password[0] == 97 && current_password[1] == 97 && current_password[2] == 97){
        //     printf("Thread num: %d, Password: %s\n", omp_get_thread_num(), current_password);
        // }
        sha256(current_password, current_hash);

        if (strcmp(current_hash, target_hash) == 0) {
            #pragma omp critical
            {
                if (!found) {
                    found = 1;
                    printf("Znalezione hasło: %s\n", current_password);
                }
            }
        }
    }

    if (!found) {
        printf("Hasło nie zostało znalezione.\n");
    }

    double end_time = omp_get_wtime();
    printf("Czas wykonania: %f sekund z %d wątkami\n", end_time - start_time, NUM_THREADS);


    return 0;
}