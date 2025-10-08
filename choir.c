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
#include <stdbool.h>

typedef struct {
	int n_cases;
	int n_features;
	double* y;
	double** x;
} data;	

struct node {
	struct node* parent;
	struct node* child_left;
	struct node* child_right;
	double splitting_value;
	double y_estimate;
	double mse;
	double* y;
	double** x;
	int x_split;
	int* indices;
	int n_cases;
	int n_features;
};

void init_data(data* d, int cols, int rows)
{
	// TODO: fix memory allocation here
	d->y = calloc(rows, sizeof(double));
	if (errno != 0) {
		fprintf(stderr, "error in calloc: %s\n", strerror(errno));
		exit(EXIT_FAILURE);
	}
	d->x = malloc((cols - 1) * sizeof(double *));
	if (errno != 0) {
		fprintf(stderr, "error in malloc: %s\n", strerror(errno));
		exit(EXIT_FAILURE);
	}

	for (int i = 0; i < cols - 1; i++) {
		d->x[i] = calloc(rows, sizeof(double));
		if (errno != 0) {
			fprintf(stderr, "error in calloc: %s\n", strerror(errno));
			exit(EXIT_FAILURE);
		}
	}
	d->n_cases = rows;
	d->n_features = cols - 1;
}

void free_data(data* d)
{
	if (!d) return;

	for (int i = 0; i < d->n_features; i++) {
		free(d->x[i]);
	}
	free(d->y);
	free(d->x);
}

void init_node(struct node* n, double* y, double** x, int n_cases, int n_features)
{
	n->parent = NULL;
	n->child_left = NULL;
	n->child_right = NULL;
	n->splitting_value = 0;
	n->y_estimate = 0;
	n->y = y;
	n->x = x;
	n->n_cases = n_cases;
	n->n_features = n_features;
	
	n->indices = calloc(n_cases, sizeof(int));	
}

void free_node(struct node* n)
{
	if (!n) return;

	free(n->indices);
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

/*
double get_mean(double* vals, int n_cases)
{
    if (n_cases < 1) return 0;

    double sum = 0;

    for (int i = 0; i < n_cases; i++) {
        sum += vals[i];
    }

    return sum / n_cases;
}
*/

double get_y_estimate(struct node* n)
{
    double sum = 0;

    for (int i = 0; i < n->n_cases; i++) {
        sum += n->y[n->indices[i]];
    }

    return sum / n->n_cases;
}

double get_mse(struct node* n)
{
	double acc = 0;

	for (int i = 0; i < n->n_cases; i++) {
		acc += powf(n->y[n->indices[i]] - n->y_estimate, 2);
	}

	return acc / n->n_cases;
}

int comparison(const void* a, const void* b)
{
	double f_a = * ( (double *) a);
	double f_b = * ( (double *) b);

	if ( f_a == f_b ) return 0;
	else if (f_a < f_b) return -1;
	else return 1;
}

/*
double get_node_misclassification_rate(double* y, double estimated_y, int n_cases)
{
	double acc = 0;

	for (int i = 0; i < n_cases; i++) {
		acc += powf(y[i] - estimated_y, 2);
	}

	return acc / n_cases;
}
*/

void get_best_split_for_x(double* res, struct node* n, int x_i)
{
	double xs[n->n_cases];
	// split_value | misclassification left | misclassification right | total decrease
	double splits[n->n_cases - 1][4];

	for (int i = 0; i < n->n_cases; i++) {
		for (int j = 0; j < 4; j++) splits[i][j] = 0;
		xs[i] = n->x[x_i][n->indices[i]];
	}

	qsort(xs, n->n_cases, sizeof(double), comparison);

	double min_value = xs[0];
    double max_value = xs[n->n_cases - 1];
    double min_threshold = 0.05 * (max_value - min_value);

	for (int i = 0; i < n->n_cases - 1; i++) {
		memset(splits[i], 0, sizeof(double) * 4);
	}

	for (int i = 0; i < n->n_cases - 1; i++) {
		splits[i][0] = (xs[i] + xs[i + 1]) / 2;

		if (splits[i][0] - min_value < min_threshold || max_value - splits[i][0] < min_threshold) {
            continue;
        }

		double y_estimate_left = 0; 
		double y_estimate_right = 0; 
		int n_left = 0;
		int n_right = 0;

		for (int j = 0; j < n->n_cases; j++) {
			if (n->x[x_i][n->indices[j]] > splits[i][0]) {
				y_estimate_left += n->y[n->indices[j]];
				n_left++;
			} else {
				y_estimate_right += n->y[n->indices[j]];
				n_right++;
			}
		}

		y_estimate_left = y_estimate_left / n_left;
		y_estimate_right = y_estimate_right / n_right;

		double acc_l = 0;
		double acc_r = 0;

		for (int j = 0; j < n->n_cases; j++) {
			if (n->x[x_i][n->indices[j]] > splits[i][0]) {
				acc_l += powf(n->y[n->indices[j]] - y_estimate_left, 2);
			} else {
				acc_r += powf(n->y[n->indices[j]] - y_estimate_right, 2);
			}
		}
		
		if (acc_l > 0) splits[i][1] = acc_l / n_left;
		if (acc_r > 0) splits[i][2] = acc_r / n_right;
		splits[i][3] = n->mse - ((double)n_left / n->n_cases) * splits[i][1]
					   - ((double)n_right / n->n_cases) * splits[i][2];
	}

	int best_index = 0;
	double largest_decrease = splits[0][3];

	for (int i = 0; i < n->n_cases - 1; i++) {
		if (splits[i][3] > largest_decrease) {
			largest_decrease = splits[i][3];
			best_index = i;
		}
	}

	memcpy(res, splits[best_index], sizeof(double) * 4);
}

void set_best_split(struct node* n)
{
	// split_value | misclassification left | misclassification right | total error
	double splits[n->n_features][4];

	for (int i = 0; i < n->n_features; i++) {
		for (int j = 0; j < 4; j++) splits[i][j] = 0;
	}

	for (int i = 0; i < n->n_features; i++) {
		get_best_split_for_x(splits[i], n, i);
	}

	int best_index = 0;
	double largest_decrease = splits[0][3];
	for (int i = 1; i < n->n_features; i++) {
		if (splits[i][3] < largest_decrease && splits[i][3] > 0) {
			largest_decrease = splits[i][3];
			best_index = i;
		}
	}

	n->splitting_value = splits[best_index][0];
	n->x_split = best_index;
}

void grow(struct node* parent)
{
	if (parent->n_cases < 5) return;

    int n_left = 0, n_right = 0;

	for (int i = 0; i < parent->n_cases; i++) {
		if (parent->x[parent->x_split][parent->indices[i]] 
			< parent->splitting_value) {
			n_left++;
		} else {
			n_right++;
		}
	}

	printf("splitting value: %f\nx: %d\nn: %d\n", parent->splitting_value, parent->x_split, parent->n_cases);

	if (n_left > 5) {
		struct node child_left;

		init_node(&child_left, parent->y, parent->x, n_left, parent->n_features);
		child_left.parent = parent;
		child_left.n_cases = n_left;

		int n_l = 0;

		for (int i = 0; i < parent->n_cases; i++) {
			if (parent->x[parent->x_split][parent->indices[i]] 
				< parent->splitting_value) {
				child_left.indices[n_l++] = i;
			} 
		}

		child_left.y_estimate = get_y_estimate(&child_left);
		child_left.mse = get_mse(&child_left);
		set_best_split(&child_left);

		grow(&child_left);
	}

	if (n_right > 5) {
		struct node child_right;

		init_node(&child_right, parent->y, parent->x, n_right, parent->n_features);
		child_right.parent = parent;
		child_right.n_cases = n_right;

		int n_r = 0;

		for (int i = 0; i < parent->n_cases; i++) {
			if (parent->x[parent->x_split][parent->indices[i]] 
				> parent->splitting_value) {
				child_right.indices[n_r++] = i;
			}
		}

		child_right.y_estimate = get_y_estimate(&child_right);
		child_right.mse = get_mse(&child_right);
		set_best_split(&child_right);

		grow(&child_right);
	}
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

    init_data(&d, cols, rows - 1);
    size_t read = 0;
    free(line);
    line = NULL;
    line_len = 0;
    rewind(fp);

	getline(&line, &line_len, fp);
	if (errno != 0) {
		fprintf(stderr, "error in getline: %s\n", strerror(errno));
		exit(EXIT_FAILURE);
	}

    for (int i = 0; (read = getline(&line, &line_len, fp)) != -1; i++) {

		if (errno != 0) {
			fprintf(stderr, "error in getline: %s\n", strerror(errno));
			exit(EXIT_FAILURE);
		}

		for (int j = 1; j <= cols; j++) {
			char* tmp = strdup(line);
            if (tmp == NULL) {
                free(line);
                fclose(fp);
                fprintf(stderr, "error in strdup: %s\n", strerror(errno));
                exit(EXIT_FAILURE);
            }

            double el = atof(get_csv_element(tmp, j));

            if (j == 1) {
				d.y[i] = el;
            } else {
				d.x[j - 2][i] = el;
            }
            free(tmp);
		}
	}

    free(line);
    fclose(fp);

	struct node root_node;
	int ind[d.n_cases];

	for (int i = 0; i < d.n_cases; i++) {
		ind[i] = i;
	}

	root_node.y = d.y;
	root_node.x = d.x;
	root_node.n_cases = d.n_cases;
	root_node.n_features = d.n_features;
	root_node.indices = ind;
	root_node.y_estimate = get_y_estimate(&root_node);
	root_node.mse = get_mse(&root_node);
	set_best_split(&root_node);

	grow(&root_node);

	free_data(&d);
    return 0;
}

