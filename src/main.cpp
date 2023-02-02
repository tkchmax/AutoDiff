#include <iostream>
#include <random>
#include "Model.h"
#include "MNIST.h"
using namespace std;

namespace {
    Mat Flatten(const Mat2& mat) {
        Mat res;
        for (int i = 0; i < mat.size(); ++i) {
            for (int j = 0; j < mat[0].size(); ++j) {
                res.push_back(mat[i][j]);
            }

        }
        return res;
    }

    void ShowImg(const Mat2& img) {
        for (int i = 0; i < img.size(); ++i) {
            for (int j = 0; j < img.size(); ++j) {
                char c = (img[i][j] != 0) ? '@' : ' ';
                std::cout << c;
            }
            std::cout << std::endl;
        }
    }

    void Test(Model& model, const MNIST::LabeledSamples& testImg) {
        std::vector<Tensor> test, labels;
        for (int i = 0; i < testImg.size(); ++i) {
            test.push_back(Tensor(Flatten(testImg[i].second)).T());
            labels.push_back(Tensor(testImg[i].first).T());
        }

        int r = rand() % test.size();
        AccuracyMetric acc;
        for (int i = 0; i < test.size(); ++i) {
            Tensor output = model.predict(test[i]);
            acc.update_state(output, labels[i]);

            if (i == r) {
                ShowImg(testImg[i].second);
                std::cout << "#" << r << " predicted: " << output.T();
            }
        }
        std::cout << "\nTest size: " << test.size() << std::endl;
        std::cout << "Test Accuracy: " << acc.result() << std::endl;
    }
}

int main()
{
    srand(time(0));

    auto labeled_2d = MNIST::Get().GetLabeledImages();
    std::pair<MNIST::LabeledSamples, MNIST::LabeledSamples> train_test = MNIST::Get().GetTrainTestSamples();

    MNIST::LabeledSamples trainImg = train_test.first;
    MNIST::LabeledSamples testImg = train_test.second;

    std::vector<Tensor> train, labels;
    for (int i = 0; i < trainImg.size(); ++i) {
        train.push_back(Tensor(Flatten(trainImg[i].second)).T());
        labels.push_back(Tensor(trainImg[i].first).T());
    }

    Sequential model(
        Input(Tensor::Shape(28 * 28, 1)),
        DenseLayer(10, EActivation::softmax)
    );

    model.compile(ELoss::CROSS_ENTROPY, EMetrics::ACCURACY);
    model.fit(train, labels, 2);

    Test(model, testImg);
}

