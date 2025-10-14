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

struct alpha_pair {
	double value;
	struct node* node_ptr;
}

void init_data(data* d, int cols, int rows)
{
	// TODO: fix memory allocation here
	d->y = calloc(rows, sizeof(double));
	if (!d->y) {
		fprintf(stderr, "error in calloc: %s\n", strerror(errno));
		exit(EXIT_FAILURE);
	}
	d->x = malloc((cols - 1) * sizeof(double *));
	if (!d->x) {
		fprintf(stderr, "error in malloc: %s\n", strerror(errno));
		exit(EXIT_FAILURE);
	}

	for (int i = 0; i < cols - 1; i++) {
		d->x[i] = calloc(rows, sizeof(double));
		if (!d->x[i]) {
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
	n->mse = 0;
	n->y = y;
	n->x = x;
	n->x_split = 0;
	n->indices = calloc(n_cases, sizeof(int));	
	if (!n->indices) {
		fprintf(stderr, "error in calloc: %s\n", strerror(errno));
		exit(EXIT_FAILURE);
	}
	n->n_cases = n_cases;
	n->n_features = n_features;
}

void free_node(struct node* n)
{
	if (!n) return;

	free(n->indices);
	free(n);
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
		acc += (n->y[n->indices[i]] - n->y_estimate) *
			(n->y[n->indices[i]] - n->y_estimate);
	}

	return acc / n->n_cases;
}

int comparison(const void* a, const void* b)
{
	double f_a = * ((double *) a);
	double f_b = * ((double *) b);

	if (f_a == f_b) return 0;
	else if (f_a < f_b) return -1;
	else return 1;
}

int comparison_alpha(const void* a, const void* b)
{
	struct pair *p_a =  (struct alpha_pair *) a;
	struct pair *p_b =  (struct alpha_pair *) b;

	if (p_a->value == p_b->value) return 0;
	else if (p_a->value < p_b->value) return -1;
	else return 1;
}

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
				acc_l += (n->y[n->indices[j]] - y_estimate_left) * 
					(n->y[n->indices[j]] - y_estimate_left);
			} else {
				acc_r += (n->y[n->indices[j]] - y_estimate_right) *
					(n->y[n->indices[j]] - y_estimate_right);
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

	if (n_left == 0 || n_right == 0) return;

	printf("splitting value: %f\nx: %d\nn: %d\n", parent->splitting_value, parent->x_split, parent->n_cases);

	struct node *child_left = malloc(sizeof(struct node));
	struct node *child_right = malloc(sizeof(struct node));

	if (!child_left || !child_right) {
		fprintf(stderr, "error in malloc: %s\n", strerror(errno));
		exit(EXIT_FAILURE);
	}

	init_node(child_left, parent->y, parent->x, n_left, parent->n_features);
	child_left->parent = parent;
	parent->child_left = child_left;

	init_node(child_right, parent->y, parent->x, n_right, parent->n_features);
	child_right->parent = parent;
	parent->child_right = child_right;

	int n_l = 0;
	int n_r = 0;

	for (int i = 0; i < parent->n_cases; i++) {
		if (parent->x[parent->x_split][parent->indices[i]] 
			< parent->splitting_value) {
			child_left->indices[n_l++] = parent->indices[i];
		} else {
			child_right->indices[n_r++] = parent->indices[i];
		}
	}

	child_left->y_estimate = get_y_estimate(child_left);
	child_left->mse = get_mse(child_left);

	child_right->y_estimate = get_y_estimate(child_right);
	child_right->mse = get_mse(child_right);

	if (n_left > 5) {
		set_best_split(child_left);
		grow(child_left);
	}

	if (n_right > 5) {
		set_best_split(child_right);
		grow(child_right);
	}
}

void free_tree(struct node* parent)
{
	if (!parent) return;
	
	if (!parent->child_left && !parent->child_right) {
		free_node(parent);
	} else {
		free_tree(parent->child_left);
		free_tree(parent->child_right);
		free_node(parent);
	}
}

void get_total_t_nodes (struct node* parent, int* total)
{
	if (!parent) return;
	
	if (!parent->child_left && !parent->child_right) {
		*total++;
	} else {
		get_total_t_nodes(parent->child_left, total);
		get_total_t_nodes(parent->child_right, total);
	}
}

double get_alpha (struct node* parent, int t_nodes)
{
	if (!parent || (!parent->child_left && !parent->child_right)) return -1;

	return ((parent->mse - (parent->child_left->mse + parent->child_right->mse))
		/ t_nodes - 1);
}

void prune(struct node* parent)
{
	int n_trees = -1;
	int t_nodes = 0;
	get_total_t_nodes(parent, &t_nodes);
	double depth = log2(t_nodes);

	struct node*** terminal_nodes = malloc(depth * sizeof(struct node**));
	if (!terminal_nodes) {
		fprintf(stderr, "error in malloc: %s\n", strerror(errno));
		exit(EXIT_FAILURE);
	}

	for (int i = 0; i < depth; i++) {
		terminal_nodes[i] = malloc(pow(2, depth - i) * sizeof(struct node*));
		if (!terminal_nodes[i]) {
			fprintf(stderr, "error in malloc: %s\n", strerror(errno));
			exit(EXIT_FAILURE);
		}
	}

	int* n_terminal_nodes = calloc(depth, sizeof(int));
	if (!n_terminal_nodes) {
		fprintf(stderr, "error in calloc: %s\n", strerror(errno));
		exit(EXIT_FAILURE);
	}
	n_terminal_nodes[0] = t_nodes;

	for (int i = 0; !finished; i++) {
		int n_parents = n_terminal_nodes[i] / 2;
		struct alpha_pair *pairs = malloc(n_parents * sizeof(struct alpha_pair));
		if (!pairs) {
			fprintf(stderr, "error in calloc: %s\n", strerror(errno));
			exit(EXIT_FAILURE);
		}

		for (int j = 0; j < n_parents; j++) {
			pairs[j].value = get_alpha(terminal_nodes[i][j * 2], n_terminal_nodes[i]);
			pairs[j].node_ptr = terminal_nodes[0][j * 2];
		}

		qsort(pairs, n_parents, sizeof(double), comparison);

		int to_prune = 1;
		// to prune their children
		for (int j = 1; j < n_parents; j++) {
			if (pairs.value[0] - pairs.value[0] * 0.1 > pairs.value[j] && 
				pairs.value[0] + pairs.value[0] * 0.1 < pairs.value[j]) {
				to_prune++;
			}
		}

		struct node** will_prune = malloc(to_prune * 2 * sizeof(struct node*));

		for (int j = 0; j < to_prune; j++) {
			will_prune[j * 2] = pairs[j].node_ptr->child_left;
			will_prune[(j * 2) + 1] = pairs[j].node_ptr->child_right;
		}

		int index = 0;

		for (int j = 0; j < n_terminal_nodes[i]; j++) {
			bool need_to_prune = false;

			for (int k = 0; k < to_prune; k++) {
				if (terminal_nodes[i][j] == will_prune[k]) need_to_prune = true;
			}

			if (need_to_prune) {
				terminal_nodes[i + 1][index++] = terminal_nodes[i][j++]->parent;
			} else {
				terminal_nodes[i + 1][index++] = terminal_nodes[i][j];
			}

		}
	
	/*
		- Once alphas with node pointers are sorted, store the node pointers
		of the children in an array. 
		- Then, go through the terminal nodes in the previous tree and add those 
		that do not need to be pruned to the terminal nodes of the next tree.
		- When the ones that do need to be pruned are found, add their parents
		to the list of terminal nodes instead.
	*/

	/*
		- Calculate g1 for the parents of all the terminal nodes
		- Order the g1s from smallest to greatest
		- "Prune" the weakest link(s)
		- Save the new set of terminal nodes in the array of terminal
		nodes
	*/

	}
	
	/*

	- Find decreasing sequence of subtrees
		- How to store it

		Store array of array of node pointers. Each array of node pointers
		contains the pointers to the terminal nodes of the tree. As the se-
		quence of trees grows, the sequence of array grows too with the
		pointers to the terminal nodes of the subsequent trees.

		- Calculate cost complexity and find alpha

		Store array with number of terminal nodes per tree in the sequence.
		Calculate the g1 (see notes) for all of the parent nodes of the
		terminal nodes in the array of nodes declared above and find the
		weakest link or links.

	- Estimate error with test sample
		
		Abstract to another func?

	- Choose best tree

		Use 1SE rule

	*/
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

	struct node *root_node = malloc(sizeof(struct node));
	if (!root_node) {
		fprintf(stderr, "error in malloc: %s\n", strerror(errno));
		exit(EXIT_FAILURE);
	}

	init_node(root_node, d.y, d.x, d.n_cases, d.n_features);
	for (int i = 0; i < d.n_cases; i++) {
		root_node->indices[i] = i;
	}
	root_node->y_estimate = get_y_estimate(root_node);
	root_node->mse = get_mse(root_node);
	set_best_split(root_node);

	grow(root_node);

	free_tree(root_node);
	free_data(&d);
    return 0;
}

