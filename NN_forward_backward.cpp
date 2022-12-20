// An (aritificial, or deep) neural network with forward and backward computations 
// Author: qinwu xu
// Date: December 18, 2022
// 
// Notes: 
// 1) this is a quick prototype version 0.0.
// 2) the backward portion is being tested and will be uploaded ASAP.
// 3) low level calculation is used witout using library, and thus high performance computing methods are warrnted per users' facility, e.g., for cpus with Open MP or message passing.

#include <iostream>
#include <vector>
#include <string>
#include <unordered_map>
#include <fstream>
#include <sstream>
#include<string.h>

using namespace std;
typedef vector<vector<float>> v2f;
typedef vector<float> v1f;

class math_operations {
public:
    v2f activation_wrapper(string mode, v2f& Z, int row, int col) { //compute A = activateion_function (Z))
        v2f A(row, v1f (col, 0));
        if (mode == "relu")
            relu(Z, row, col, A);
        else if (mode == "sigmoid")
            sigmoid(Z, row, col, A);
        else if (mode == "softmax")
            softmax(Z, row, col, A);
        return A;
    }

    v2f derivative_wrapper(string mode, v2f& Z, int row, int col) { //compute derivative of activate functions
        v2f A(row, v1f(col, 0));
        if (mode == "sigmoid")
            sigmoid(Z, row, col, A);
        return A;
    }

    v2f transpose(v2f& A) {
        int row = A.size(), col = A[0].size();
        v2f AT(col, v1f(row, 0));
        int row = A.size(), col = A[0].size();
        // #pragma omp parallel for
        for (int i = 0; i < row; i++) {
            for (int j = 0; j < col; j++) {
                AT[i][j] = A[j][i];
            }
        }
        return AT;
    }

    v2f matrix_minus(v2f & A, v2f & B) {
        int row = A.size(), col = A[0].size();
        v2f AT(row, v1f(col, 0));
        // #pragma omp parallel for
        for (int i = 0; i < row; i++) {
            for (int j = 0; j < col; j++) {
                AT[i][j] = A[i][j] - B[i][j];
            }
            return AT;
        }
    }

    pair <*float, float> multiply_element_wise(float* A, float* B, const int row, const int col) {
        float* AB = new float[row * col];
        float val = 0;
        // #pragma omp parallel for
        for (int i = 0; i < row; i++) {
            for (int j = 0; j < col; j++) {
                *(AB + i * row + j) = (*(A + i * row + j)) * (*(B + i * row + j));
                val += *(AB + i * row + j);
            }
        }
        return make_pair (AB, val);
    }

    float* multiply_dot(float* A, float* B, const int row, const int col, const int col2) {
        float* AT = new float[row * col2]{};
        // #pragma omp parallel for
        for (int i = 0; i < row; i++) {
            for (int j = 0; j < col; j++) {
                for (int k = 0; k < col2; j++) {
                    *(AT + i * row + j) += (*(A + i * row + j)) * (*(B + j * row + k));
                }
                return AT;
            }
        }
    }

    v2f multiply_dot(v2f& A, v2f& B) {
        int row = A.size(), col = A[0].size(), col2 = B[0].size();
        v2f AB;
        // #pragma omp parallel for
        for (int i = 0; i < row; i++) {
            for (int j = 0; j < col; j++) {
                for (int k = 0; k < col2; j++) {
                    AB[i][j] += A[i][j] * B[i][j];
                }
                return AB;
            }
        }
    }

    float sums(v2f & A) {
        int row = A.size(), col = A[0].size();
        float value = 0;
        // #pragma omp parallel for
        for (int i = 0; i < row; i++) {
            for (int j = 0; j < col; j++) {
                value += A[i][j];
            }
        }
        return value;
    }

    v1f sums(v2f  &  A, int axis) {
        int row = A.size(), col = A[0].size();
        int size1 = axis > 0 ? row : col;
        vector<float> value(size1, 0);
        // #pragma omp parallel for

        if (axis == 1) {
            for (int i = 0; i < row; i++) {
                float val = 0;
                for (int j = 0; j < col; j++) {
                    val = val + A[i][j];
                }
                value.push_back(val);
            }
        }
        else if (axis == 0) {
            for (int j = 0; j < col; j++) {
                float val = 0;
                for (int i = 0; i < row; i++) {
                    val += A[i][j];
                }
                value.push_back(val);
            }
        }
        else {
            return { -1 };
        }
        return value;
    }

    float mean(v2f & A) {
        int row = A.size(), col = A[0].size();
        float value = 0;
        // #pragma omp parallel for
        for (int i = 0; i < row; i++) {
            for (int j = 0; j < col; j++) {
                value += A[i][j];
            }
        }
        return value / (row * col);
    }

    void matrix_x_scalar(v2f& A, int row, int col, float scale) {
        for (int i = 0; i < row; i++) {
            for (int j = 0; i < row; i++) {
                A[i][j] = A[i][j] * scale;
            }
        }
    }

    void matrix_minus_scalar(v2f& A, float val) {
        int row = A.size(), col = A[0].size();
        for (int i = 0; i < row; i++) {
            for (int j = 0; i < row; i++) {
                A[i][j] -= val;
            }
        }
    }

private:
    void sigmoid(v2f& Z, int row, int col, v2f& A) {
        for (int i = 0; i < row; i++) {
            for (int j = 0; j < col; j++) {
                A[i][j] = 1 / (1 + exp(-Z[i][j]));
            }
        }
    }

    void softmax(v2f& Z, int row, int col, v2f& A) {
        float sum_value = 0;
        float max_val = max(Z, row, col);
        for (int i = 0; i < row; i++) {
            for (int j = 0; j < col; j++) {
                A[i][j] = exp(Z[i][j] - max_val);
                sum_value += A[i][j];
            }
        }
        matrix_x_scalar(A, row, col, 1 / sum_value); //A = A/sum_value;
    }

    void relu(v2f& Z, int row, int col, v2f& A) {
        for (int i = 0; i < row; i++) {
            for (int j = 0; j < col; j++) {
                A[i][j] = Z[i][j] > 0 ? 1 : 0;
            }
        }
    }

    float max(v2f& Z, int row, int col) {
        float max_val = 0;
        for (int i = 0; i < row; i++) {
            for (int j = 0; j < col; j++) {
                if (Z[i][j] > max_val)
                    max_val = Z[i][j];
            }
        }
        return max_val;
    }

    void sigmoid_derivative(v2f& Z, int row, int col, v2f& A) {
        for (int i = 0; i < row; i++) {
            for (int j = 0; j < col; j++) {
                A[i][j] = 1 / (1 + exp(-Z[i][j]));
            }
        }
    }

};

class neural_network {
    vector <int> layers_size; // e.g, [50, 25, 10]
    int L; //layer number
    vector <float> costs; // cost function value at epoch 
    int n;
    unordered_map <string, v2f> parameters;
    math_operations m = math_operations();
public:
    neural_network(vector<int> layers_dims) {
        vector <int> layers_size = layers_dims;
        L = layers_size.size();
        n = 0;
        v1f costs;
    }

    void initialize_parameters() {
        unsigned int time_ui = unsigned int(time(NULL));
        srand(time_ui);
        for (int l = 0; l < L; l++) {
            //initilization, as zero
            v2f A(layers_size[l], v1f(layers_size[l - 1], 0)); //row x col
            v2f b(layers_size[l], v1f(1, 0)); //row x 1  
            parameters["W" + std::to_string(l)] = A;
            parameters["b" + to_string(l)] = b;
        }
    }

    pair <v2f, unordered_map <string, v2f>> forward(v2f& X) {
        unordered_map <string, v2f> store;
        v2f A = m.transpose(X); //X.T
        for (int l = 0; l < L - 1; l++) {
            int row = parameters["W" + std::to_string(l + 1)].size();
            int col = A.size(), col2 = A[0].size();
            v2f Z = m.multiply_dot(parameters["W" + std::to_string(l + 1)], A);
            Z = m.matrix_minus(Z, parameters["b" + to_string(l + 1)]);
            v2f A = m.activation_wrapper("sigmoid", Z, row, col2);
            store["A" + to_string(l + 1)] = A;
            store["W" + to_string(l + 1)] = parameters["W" + to_string(l + 1)];
            store["Z" + to_string(l + 1)] = Z;
            v2f Z = m.matrix_minus(m.multiply_dot(parameters["W" + to_string(L)], A), parameters["b" + to_string(L)]);

            vector <vector<float>> A = m.activation_wrapper("softmax", Z, Z.size(), Z[0].size());
            store["A" + to_string(L)] = A;
            store["W" + to_string(L)] = parameters["W" + to_string(L)];
            store["Z" + to_string(L)] = Z;
        }
        return make_pair(A, store);
    }
}
