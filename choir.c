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

enum signs { GREATER, LESS };

typedef struct {
	int n_cases;
	int n_features;
	float* y;
	float** x;
} data;	

typedef struct {
	struct node* parent;
	struct node* child_left;
	struct node* child_right;
	float splitting_value;
	enum signs splitting_sign;
	float* y;
	float** x;
} node;

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

float get_mean(float* vals, int n_cases)
{
    if (n_cases < 1)
        return 0;
    float sum = 0;

    for (int i = 0; i < n_cases; i++) {
        sum += vals[i];
    }

    return sum / n_cases;
}

int comparison(const void* a, const void* b)
{
	float f_a = * ( (float *) a);
	float f_b = * ( (float *) b);

	if ( f_a == f_b ) return 0;
	else if (f_a < f_b) return -1;
	else return 1;
}

void transpose_x(float** x, float** new_x, int n_cases, int n_features)
{
	for (int i = 0; i < n_cases; i++) {
		for (int j = 0; j < n_features; j++) {
			printf("%f, case %d feature %d\n", x[i][j], i, j);
			new_x[j][i] = x[i][j];
		}
	}
}

float get_node_misclassification_rate(float* y, float estimated_y, int n_cases)
{
	float acc = 0;

	for (int i = 0; i < n_cases; i++) {
		acc += powf(y[i] - estimated_y, 2);
	}

	return acc;
}

float get_best_split(float* y, float* x, int n_cases)
{
	// split_value | misclassification left | misclassification right | misclassification children
	float xs[n_cases];
	float splits[n_cases - 1][4];

	memcpy(&xs, &x, sizeof(float) * n_cases);
	qsort(xs, n_cases, sizeof(float), comparison);

	for (int i = 0; i <= n_cases - 1; i++) {
		splits[i][0] = (x[i] + x[i + 1]) / 2;

		float y_estimate_left = 0; 
		float y_estimate_right = 0; 
		int n_left = 0;
		int n_right = 0;
		float* y_left;
		float* y_right;

		for (int j = 0; j < n_cases; j++) {
			if (x[j] > splits[i][0]) {
				y_estimate_left += y[j];
				n_left++;
			} else {
				y_estimate_right += y[j];
				n_right++;
			}
		}

		y_estimate_left = y_estimate_left / n_left;
		y_estimate_right = y_estimate_right / n_right;

		y_left = calloc(n_left, sizeof(float));
		y_right = calloc(n_right, sizeof(float));

		for (int j = 0; j < n_cases; j++) {
			int l_c = 0;
			int r_c = 0;

			if (x[j] > splits[i][0]) {
				y_left[l_c] = y[j];
				l_c++;
			} else {
				y_right[r_c] = y[j];
				r_c++;
			}
		}

		splits[i][1] = get_node_misclassification_rate(y_left, y_estimate_left, n_left);
		splits[i][2] = get_node_misclassification_rate(y_right, y_estimate_right, n_right);
		splits[i][3] = splits[i][1] + splits[i][2];

		free(y_left);
		free(y_right);
	}

	int smallest_index = 0;
	float smallest = splits[0][3];
	for (int i = 0; i < n_cases - 1; i++) {
		if (splits[i][3] < smallest) {
			smallest = splits[i][0];
		}
	}

	return smallest;
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
				printf("case is %d and col is %d and float is %f\n", i, j, el);
				d.x[i][j - 2] = el;
            }
            free(tmp);
		}
	}

    free(line);
    fclose(fp);

	node root_node;

	root_node.y = d.y;
	root_node.x = malloc(sizeof(float) * d.n_cases * d.n_features);

	for (int i = 0; i < d.n_features; i++) {
		root_node.x[i] = calloc(d.n_cases, sizeof(float));
	}

	for (int i = 0; i < d.n_cases; i++) {
		printf("%f\n", root_node.x[0][i]);
	}

	transpose_x(d.x, root_node.x, d.n_cases, d.n_features);

	root_node.splitting_value = get_best_split(root_node.y, root_node.x[0], d.n_cases);

	printf("%f\n", root_node.splitting_value);

	// TODO: root_node.x has rows as case and each row has the measurement
	// vector for each case, how it should be structured is that each row
	// contains all of the values for an x_m ordered in the same way as
	// root_node.y

	// TODO: create init and free functions for nodes

	// maybe just store the data in d.x already transposed so no need for 
	// function

    if (errno != 0) {
        fprintf(stderr, "error in getline: %s\n", strerror(errno));
        exit(EXIT_FAILURE);
    }

	free_data(&d);
	for (int i = 0; i < d.n_features; i++) {
		free(root_node.x[i]);
	}
	free(root_node.x);
    return 0;
}
