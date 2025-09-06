/*

	Create classification and regression trees based on CART book by Breiman et al.

*/

#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

typedef struct {
	int n_cases;
	int n_features;
	float* y;
	float** x;
} data;	

float get_mean(float* vals, int len)
{
    if (len < 1)
        return 0;
    float sum = 0;

    for (int i = 0; i < len; i++) {
        sum += vals[i];
    }

    return sum / len;
}

void init_data(data* d, int cols, int rows)
{
	d->y = malloc(rows * sizeof(float));
	d->x = malloc(rows * sizeof(float) * cols);

	for (int i = 0; i < rows; i++) {
		d->x[i] = calloc(cols, sizeof(float));
	}
	d->n_cases = rows;
	d->n_features = cols - 1;
}

void free_data(data* d)
{
	if (!d) return;

	for (int i = 0; i < d->n_cases; i++) {
		free(d->x[i]);
	}
	free(d->y);
	free(d->x);
}

int get_n_columns(char* line)
{
    int n_cols = 0;
    char* tok;

    for (tok = strtok(line, ","); tok && *tok; tok = strtok(NULL, ",\n")) {
        n_cols++;
    }

    return n_cols;
}

int get_n_rows(FILE* fp)
{
    char* line = NULL;
    size_t line_len;
    int rows = 1;
    size_t read = 0;

    while (getline(&line, &line_len, fp) != -1) {
        rows++;
    }

    free(line);
    return rows;
}

const char* get_csv_element(char* line, int num)
{
    const char* tok;
    for (tok = strtok(line, ","); tok && *tok; tok = strtok(NULL, ",\n")) {
        if (--num == 0) {
            return tok;
        }
    }

    return NULL;
}

int main(int argc, char* argv[])
{
    if (argc < 2) {
        fprintf(stderr, "usage: %s file.csv\n", argv[0]);
        exit(EXIT_FAILURE);
    }

    char* home = getenv("HOME");
    FILE* fp;

    if (access(argv[1], F_OK) != 0) {
        fprintf(stderr, "error in access: %s\n", strerror(errno));
        exit(EXIT_FAILURE);
    }

    fp = fopen(argv[1], "r");
    if (fp == NULL) {
        fprintf(stderr, "error in fopen: %s\n", strerror(errno));
        exit(EXIT_FAILURE);
    }

    char* line = NULL;
    size_t line_len;

    int err = getline(&line, &line_len, fp);
    if (err == -1) {
        if (line) {
            free(line);
        }
        fclose(fp);
        fprintf(stderr, "error in getline: %s\n", strerror(errno));
        exit(EXIT_FAILURE);
    }

    // first col is y second is x
    int cols = get_n_columns(line);
    // number of observations
    int rows = get_n_rows(fp);

    data d;

    init_data(&d, cols, rows);

    /*
     *
     *
     *
     * read stuff from csv and save into x and y
     *
     *
     *
     */

    size_t read = 0;
    free(line);
    line = NULL;
    line_len = 0;
    rewind(fp);

    for (int i = 0; (read = getline(&line, &line_len, fp)) != -1; i++) {
		for (int j = 1; j <= cols; j++) {
			char* tmp = strdup(line);
            if (tmp == NULL) {
                free(line);
                fclose(fp);
                fprintf(stderr, "error in strdup: %s\n", strerror(errno));
                exit(EXIT_FAILURE);
            }

            float el = atof(get_csv_element(tmp, j));

            if (j == 1) {
				d.y[i] = el;
            } else {
			//	printf("case is %d and col is %d\n", i, j);
				printf("case is %d and col is %d and float is %f\n", i, j, el);
				printf("value is %f\n", d.x[i][j - 1]);
				d.x[i][j - 1] = el;
            }
            free(tmp);
		}
	}

    free(line);
    fclose(fp);

    if (errno != 0) {
        fprintf(stderr, "error in getline: %s\n", strerror(errno));
        exit(EXIT_FAILURE);
    }

	free_data(&d);
    return 0;
}
