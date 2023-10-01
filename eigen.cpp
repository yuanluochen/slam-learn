#include <iostream>

#include <ctime>

//eigen 核心部分
#include <eigen3/Eigen/Core>
//稠密矩阵代数运算
#include <eigen3/Eigen/Dense>

#define MATRIX_SIZE 50

int main(int argc, char **argv){

    //声明一个2 * 3的float的矩阵
    Eigen::Matrix<float, 2, 3> matrix_23;
    //三维向量
    Eigen::Vector3d v_3d;//实际上是Eigen::Matrix<double, 3, 1>
    Eigen::Matrix<float, 3, 1> vd_3d;
    //Eigen::Matrix3d实际上是Eigen::Matrix<double, 3, 3>
    Eigen::Matrix3d matrix_33 = Eigen::Matrix3d::Zero();//初始化为零
    //动态大小的矩阵
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> matrix_dynamic;
    //更简单的写法
    Eigen::MatrixXd matrix_x;

    //输入数据
    matrix_23 << 1, 2, 3, 4, 5, 6;
    //输出
    std::cout << "matrix 2 * 3 from 1 to 6: \n" << matrix_23 << std::endl;

    //矩阵操作
    //用（） 访问矩阵中的元素
    std::cout << "print matrix 2 * 3: " << std::endl;
    for (int i = 0; i < 2; i++){
        for (int j = 0; j < 3; j++){
            std::cout << matrix_23(i, j) << "\t";
        }
        std::cout << std::endl;
    }

    //矩阵和向量相乘
    v_3d << 3, 2, 1;
    vd_3d << 4, 5, 6;

    //eigen不能混合两种不同类型的矩阵
    Eigen::Matrix<double, 2, 1> result = matrix_23.cast<double>() * v_3d;
    std::cout << "[1, 2, 3; 4, 5, 6] * [3, 2, 1]=" << result.transpose() << std::endl;

    //不能搞错矩阵的维度
    // Eigen::Matrix<double, 2, 3> result_wrong_dimension = matrix_23.cast<double>() * v_3d;

    matrix_33 = Eigen::Matrix3d::Random(); //随机数矩阵
    std::cout << "random matrix: \n" << matrix_33 << std::endl;
    std::cout << "transpose: \n" << matrix_33.transpose() << std::endl;// 转置
    std::cout << "sum: " << matrix_33.sum() << std::endl; //各元素和
    std::cout << "trace: " << matrix_33.trace() << std::endl; //trace
    std::cout << "times 10: \n" << 10 * matrix_33 << std::endl; //times 数乘
    std::cout << "inverse: \n" << matrix_33.inverse() << std::endl; //逆
    std::cout << "det: " << matrix_33.determinant() << std::endl;//行列式

    //特征值
    //实对称矩阵可以保证对角化成功
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eigen_solver(matrix_33.transpose() * matrix_33);
    std::cout << "Eigen values = \n" << eigen_solver.eigenvalues() << std::endl;
    std::cout << "Eigen Vectors = \n" << eigen_solver.eigenvectors() << std::endl;

    //解方程
    //求解 matrix_NN * x = v_Nd 方程
    //Nd的大小在前面宏里定义，他由随机数生成
    
    Eigen::Matrix<double, MATRIX_SIZE, MATRIX_SIZE> matrix_NN = Eigen::MatrixXd::Random(MATRIX_SIZE, MATRIX_SIZE);
    matrix_NN = matrix_NN * matrix_NN.transpose(); //保证半正定
    Eigen::Matrix<double, MATRIX_SIZE, 1> v_Nd = Eigen::MatrixXd::Random(MATRIX_SIZE, 1);

    clock_t time_stt = clock();//计时
    //直接求逆
    Eigen::Matrix<double, MATRIX_SIZE, 1> x = matrix_NN.inverse() * v_Nd;

    std::cout << "time of normal inverse is " << 1000 * (clock() - time_stt) / (double) CLOCKS_PER_SEC << "ms" << std::endl;

    std::cout << "x = " << x.transpose() << std::endl;

    //对于正定矩阵，还可以用cholesky 分解求解矩阵
    time_stt = clock();
    x = matrix_NN.ldlt().solve(v_Nd);
    std::cout << "time of ldlt decomposition is" << 1000 * (clock() - time_stt) / (double) CLOCKS_PER_SEC << "ms" << std::endl;
    std::cout << "x = " << x.transpose() << std::endl;
    return 0;
}
