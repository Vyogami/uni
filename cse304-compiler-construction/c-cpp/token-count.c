#include <stdio.h>
#include <string.h>
#include <stdlib.h>

int main(void) {
    const char* fname = "token-count.c";
    FILE* fp = fopen(fname, "r");
    if(!fp) {
        perror("Couldn't open file.");
        return EXIT_FAILURE;
    }

    int count = 0;
    int c;
    while((c = fgetc(fp)) != EOF) {
        putchar(c);
        if (c == ' ' || c == '\n' || c == '\t'){
            count++;
        }
    }
    printf("Token Count: %d\n", count);
}
