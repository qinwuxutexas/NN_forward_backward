// An (aritificial, or deep) neural network with forward and backward computations, in C++
// Q. Xu,  Date: December 18, 2022

// Notes:
// 1) this is a quick prototype version 0.0.
// 2) low level calculation is used witout using library, and thus high performance computing methods are warrnted per users' facility, e.g., for cpus with Open MP (examples provided) or message passing.

#include <iostream>
#include <vector>
#include <string>
#include <unordered_map>
#include <fstream>
#include <sstream>
#include<string.h>
#include <random>

using namespace std;
typedef vector<vector<float>> v2f;
typedef vector<float> v1f;

class math_operations {
public:
    v2f activation_wrapper(string mode, v2f& Z) { //compute A = activateion_function (Z))
        int row = Z.size(), col = Z[0].size();
        v2f A(row, v1f(col, 0));
        if (mode == "relu")
            relu(Z, row, col, A);
        else if (mode == "sigmoid")
            sigmoid(Z, row, col, A);
        else if (mode == "softmax")
            softmax(Z, row, col, A);
        return A;
    }

    v2f derivative_wrapper(string mode, v2f& Z) { //compute derivative of activate functions
        int row = Z.size(), col = Z[0].size();
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
        for (int i = 0; i < row * col; i++) {
            int x = i / row, y = i % col;
            AT[x][y] = A[y][x];
        }
        return AT;
    }

    void log_scale_matrix(v2f& A) {
        int row = A.size(), col = A[0].size();
        for (int i = 0; i < row * col; i++) {
            int x = i / row, y = i % row;
            A[x][y] = log(A[x][y]);
        }
    }

    vector<int> argmax_matrix_cross_row(v2f A) {
        int row = A.size(), col = A[0].size();
        vector<int> index_row_max(row, 0);
        for (int i = 0; i < row; i++) {
            float max_val = -1e-8;
            int index = 0;
            for (int j = 0; j < col; j++) {
                if (A[i][j] > max_val) {
                    max_val = A[i][j];
                    index = j;
                }
            }
            index_row_max.push_back(index);
        }
        return index_row_max;
    }


    v2f matrix_minus(v2f& A, v2f& B, string sign) {
        float operation = (sign == "+" ? 1 : -1);
        int row = A.size(), col = A[0].size();
        v2f AT(row, v1f(col, 0));

        // #pragma omp parallel for
        for (int i = 0; i < row * col; i++) {
            int x = i / row, y = i % col;
            AT[x][y] = A[x][y] + operation * B[x][y];
        }
        return AT;
    }

    pair <float*, float> multiply_element_wise(float* A, float* B, const int row, const int col) {
        float* AB = new float[row * col];
        float val = 0;
        // #pragma omp parallel for
        for (int i = 0; i < row; i++) {
            for (int j = 0; j < col; j++) {
                *(AB + i * row + j) = (*(A + i * row + j)) * (*(B + i * row + j));
                val += *(AB + i * row + j);
            }
        }
        return make_pair(AB, val);
    }

    float* multiply_dot(float* A, float* B, const int row, const int col, const int col2) {
        float* AT = new float[row * col2]{};
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
        for (int i = 0; i < row; i++) {
            for (int j = 0; j < col; j++) {
                for (int k = 0; k < col2; j++) {
                    AB[i][j] += A[i][j] * B[i][j];
                }
                return AB;
            }
        }
    }

    float sums(v2f& A) {
        int row = A.size(), col = A[0].size();
        float value = 0;
        // #pragma omp parallel for
        for (int i = 0; i < row * col; i++) {
            int x = i / row, y = i % col;
            value += A[x][y];
        }
        return value;
    }

    v2f sums(v2f& A, int axis) {
        int row = A.size(), col = A[0].size();
        int size1 = axis > 0 ? row : col;
        v2f AT(size1, v1f(1, 0));

        if (axis == 1) { //operate across column
            for (int i = 0; i < row; i++) {
                float val = 0;
                v1f vec(col, 0);
                for (int j = 0; j < col; j++) {
                    val = val + A[i][j];
                }
                vec.push_back(val);
                AT.push_back(vec);
            }
        }

        else if (axis == 0) {//operate across row
            for (int j = 0; j < col; j++) {
                float val = 0;
                v1f vec(row, 0);
                for (int i = 0; i < row; i++) {
                    val += A[i][j];
                }
                vec.push_back(val);
                AT.push_back(vec);
            }
        }
        return AT;
    }

    float mean(v2f& A) {
        int row = A.size(), col = A[0].size();
        float value = 0;
        // #pragma omp parallel for
        for (int i = 0; i < row * col; i++) {
            int x = i / row, y = i % row;
            value += A[x][y];
        }
        return value / (row * col);
    }

    v2f matrix_x_scalar(v2f& A, float scale) {
        int row = A.size(), col = A[0].size();
        v2f A_scaled (row, v1f (col,0));
        // #pragma omp parallel for
        for (int i = 0; i < row * col; i++) {
            int x = i / row, y = i % row;
            A_scaled [x][y] *= A[x][y]*scale;
        }
        return A_scaled;
    }

    void matrix_minus_scalar(v2f& A, float val, string sign) {
        int row = A.size(), col = A[0].size();
        // #pragma omp parallel for
        for (int i = 0; i < row * col; i++) {
            int x = i / row, y = i % row;
            if (sign == "+")
                A[x][y] += val;
            else if (sign == "-")
                A[x][y] -= val;
        }
    }

private:
    void sigmoid(v2f& Z, int row, int col, v2f& A) {
        // #pragma omp parallel for
        for (int i = 0; i < row * col; i++) {
            int x = i / row, y = i % row;
            A[x][y] = 1 / (1 + exp(-Z[x][y]));
        }
    }

    void softmax(v2f& Z, int row, int col, v2f& A) {
        float sum_value = 0;
        float max_val = max(Z, row, col);
        // #pragma omp parallel for
        for (int i = 0; i < row; i++) {
            int x = i / row, y = i % row;
            A[x][y] = exp(Z[x][y] - max_val);
            sum_value += A[x][y];
        }
        A = matrix_x_scalar(A, 1.0 / sum_value);
    }

    void relu(v2f& Z, int row, int col, v2f& A) {
        // #pragma omp parallel for
        for (int i = 0; i < row * col; i++) {
            int x = i / row, y = i % row;
            A[x][y] = Z[x][y] > 0 ? Z[x][y] : 0;
        }
    }

    float max(v2f& Z, int row, int col) {
        float max_val = 0;
        // #pragma omp parallel for
        for (int i = 0; i < row * col; i++) {
            int x = i / row, y = i % row;
            if (Z[x][y] > max_val)
                max_val = Z[x][y];
        }
        return max_val;
    }

    void sigmoid_derivative(v2f& Z, int row, int col, v2f& A) {
        // #pragma omp parallel for
        for (int i = 0; i < row * col; i++) {
            int x = i / row, y = i % row;
            A[x][y] = 1 / (1 + exp(-Z[x][y]));
        }
    }
};

class neural_network {
    vector <int> layers_size; // e.g, [3, 25, 10]
    int L; //layer number
    vector <float> costs;
    int n;
    unordered_map <string, v2f> parameters;
    math_operations m = math_operations();

public:
    neural_network(vector<int> layers_dims, int n_) {
        vector <int> layers_size = layers_dims;
        L = layers_size.size();
        n = n_;
        v1f costs;
    }

    void initialize_parameters() {
        unsigned int time_ui = unsigned int(time(NULL));
        srand(time_ui);
        for (int l = 0; l < L; l++) {
            //initilization, as zero
            v2f A(layers_size[l], v1f(layers_size[l - 1], 0));
            v2f b(layers_size[l], v1f(1, 0));
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
            Z = m.matrix_minus(Z, parameters["b" + to_string(l + 1)], "+");
            v2f A = m.activation_wrapper("sigmoid", Z);
            store["A" + to_string(l + 1)] = A;
            store["W" + to_string(l + 1)] = parameters["W" + to_string(l + 1)];
            store["Z" + to_string(l + 1)] = Z;
            v2f WL_dot_A = m.multiply_dot(parameters["W" + to_string(L)], A);
            v2f Z = m.matrix_minus(WL_dot_A, parameters["b" + to_string(L)], "+");

            vector <vector<float>> A = m.activation_wrapper("softmax", Z);
            store["A" + to_string(L)] = A;
            store["W" + to_string(L)] = parameters["W" + to_string(L)];
            store["Z" + to_string(L)] = Z;
        }
        return make_pair(A, store);
    }

    unordered_map <string, v2f> backward(v2f X, v2f Y, unordered_map <string, v2f> store) {
        unordered_map <string, v2f> derivatives;
        store["A0"] = m.transpose(X);
        v2f A = store["A" + to_string(L)];
        v2f B = m.transpose(Y);
        v2f dZ = m.matrix_minus(A, B, "-");
        v2f AL1T = m.transpose (store["A" + to_string(L - 1)]);
        v2f dW = m.multiply_dot (dZ, AL1T);
        dW = m.matrix_x_scalar (dW, 1.0 / n);
        v2f db = m.sums (dZ, 1);
        v2f WLT = m.transpose (store["W" + to_string(L)]);
        v2f dAPrev = m.multiply_dot(WLT, dZ);
        derivatives ["dW" + to_string(L)] = dW;
        derivatives ["db" + to_string(L)] = db;

        for (int l = L - 1; l >= 0; l--) {
            v2f sig_z = m.activation_wrapper("sigmoid", store["Z" + to_string(l)]);
            v2f dZ = m.multiply_dot(dAPrev, sig_z);
            v2f AL_1T = m.transpose(store["A" + to_string(l - 1)]);
            v2f dZ_dot_AL_1T = m.multiply_dot(dZ, AL_1T);
            dZ = m.matrix_x_scalar(dZ, 1.0 / n);
            v2f db = m.sums(dZ, 1);
            db = m.matrix_x_scalar(db, 1. / n);

            if (l > 1) {
                v2f WLT = m.transpose(store["W" + to_string(l)]);
                dAPrev = m.multiply_dot(WLT, dZ);
                derivatives["dW" + to_string(l)] = dW;
                derivatives["db" + to_string(l)] = db;
            }
            return derivatives;
        }
    }

    v2f hot_vector_encoder(vector <string> y, vector<string> labels) {
        int n = y.size();
        int n_class = labels.size();
        v2f y_hot = v2f(n, v1f(n_class, 0));

        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n_class; ++j) {
                if (y[i] == labels[j]) {
                    y_hot[i][j] = 1;
                }
            }
        }
        return y_hot;
    }

    void normalize(v2f X, float scale) {
        m.matrix_x_scalar(X, scale);
    }


    v1f fit(v2f X, v2f Y, float learning_rate, int n_iterations) {
        std::random_device r;
        default_random_engine eng{ r() };
        uniform_real_distribution<double> urd(0, 1);

        int n = X.size();
        initialize_parameters();

        for (int loop = 0; loop < n_iterations; loop++) {
            tuple <v2f, unordered_map <string, v2f>> A_store = forward(X);
            v2f A = get<0>(A_store);
            unordered_map <string, v2f> store = get<1>(A_store);
            v2f AT = m.transpose(A);
            m.matrix_x_scalar(AT, 1e-8);
            m.log_scale_matrix(AT);
            v2f Y_dot_logAT = m.multiply_dot(Y, AT);
            float cost = -m.mean(Y_dot_logAT);

            unordered_map <string, v2f> derivatives = backward(X, Y, store);

            v1f costs;
            for (int l = 0; l < L + 1; l++) {
                v2f dW = derivatives["dW" + to_string(l)];
                v2f lr_dW = m.matrix_x_scalar (dW, learning_rate);
                parameters["W" + to_string(l)] = m.matrix_minus(parameters["W" + to_string(l)], lr_dW, "-");

                v2f lr_times_dbl = m.matrix_x_scalar (derivatives["db" + to_string(l)], learning_rate); 
                parameters["b" + to_string(l)] = m.matrix_minus (parameters["b" + to_string(l)], lr_times_dbl, "-");
                if (loop % 100 == 0) {
                    cout << "Cost: " << cost << endl;
                    cout << "Train Accuracy:" << accuracy(X, Y) << endl;
                }
                if (loop % 10 == 0)
                    costs.push_back(cost);
            }
        }
        return costs;
    }

    float accuracy(v2f X, v2f Y_hot) {
        tuple <v2f, unordered_map <string, v2f>> A_cache = forward(X);
        v2f A = get<0>(A_cache);
        unordered_map <string, v2f> cache = get<1>(A_cache);
        vector<int> y_predict = m.argmax_matrix_cross_row(A); //        y_hat = np.argmax (A, axis = 0);
        vector<int> y_truth = m.argmax_matrix_cross_row(Y_hot);
        int n = y_predict.size();
        float sum_val = 0;
        for (int i = 0; i < n; i++) {
            if (y_predict[i] = y_truth[i]) {
                sum_val += 1;
            }
        }
        return sum_val / n;
    }
};

int main() {
    // inputs:
    v2f tr_x;
    vector<string> tr_y;
    v2f te_x;
    vector<string> te_y;
    float scale = 1.0 / 255;
    vector<int> layers_dims = { 1, 25, 3 }; //the 1st value is the input data channel
    vector<string> labels = { "dog", "cat", "snake" };
    float learning_rate = 0.01;
    int n_iterations = 1000;
    // end of inputs

    int n_layer = layers_dims.size();
    neural_network* NN = &neural_network(layers_dims, n_layer);

    // normalize input data
    NN->normalize(tr_x, scale);
    NN->normalize(te_x, scale);

    //encoder string vector as one-hot numerical vector
    v2f tr_y_hot = NN->hot_vector_encoder(tr_y, labels);
    v2f te_y_hot = NN->hot_vector_encoder(tr_y, labels);

    v1f costs = NN->fit(tr_x, tr_y_hot, learning_rate, n_iterations);
    float tr_accuracy = NN->accuracy(tr_x, tr_y_hot);
    float te_accuracy = NN->accuracy(te_x, te_y_hot);
    cout << "Train accuracy = " << tr_accuracy;
    cout << "Ten accuracy = " << tr_accuracy;
}
