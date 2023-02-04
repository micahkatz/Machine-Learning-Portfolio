#include <math.h>

#include <algorithm>
#include <fstream>
#include <iostream>
#include <tuple>
#include <vector>

using namespace std;

double sumVector(vector<double> inputVector) {
    double acc = 0.0;
    // loop through the vector fields
    for (int i = 0; i < inputVector.size(); i++) {
        acc += inputVector.at(i);
    }
    return acc;
}
double getMeanForVector(vector<double> inputVector) {
    const double sum = sumVector(inputVector);
    const double mean = sum / inputVector.size();
    return mean;
}
double getMedianForVector(vector<double> inputVector) {
    sort(inputVector.begin(), inputVector.end());
    const int n = inputVector.size();
    const bool isOdd = n % 2 != 0;
    double median;
    if (isOdd) {
        median = inputVector.at(n / 2);
    } else {  // is even
        median = (inputVector.at((n / 2) - 1) + inputVector.at((n / 2))) / 2.0;
    }
    return median;
}

double covar(vector<double> vector1, vector<double> vector2) {
    const double meanOfVector1 = getMeanForVector(vector1);
    const double meanOfVector2 = getMeanForVector(vector2);

    double x, y;
    double n = vector1.size();

    double acc = 0;

    for (int i = 0; i < n; i++) {
        x = vector1.at(i);
        y = vector2.at(i);

        acc += (((x - meanOfVector1) * (y - meanOfVector2)) / (n - 1));
    }

    return acc;
}

double variance(vector<double> inputVector) {
    const double sumOfVector = sumVector(inputVector);
    const double meanOfVector = getMeanForVector(inputVector);
    const int n = inputVector.size();

    double squareDifference = 0.0;
    double currVal;
    for (int i = 0; i < n; i++) {
        currVal = inputVector.at(i);
        squareDifference += pow((currVal - meanOfVector), 2);
    }
    double variance = squareDifference / n;
    return variance;
}

double cor(vector<double> vector1, vector<double> vector2) {
    const double covariance = covar(vector1, vector2);

    double n = vector1.size();

    double varianceOfVector1 = variance(vector1);
    double varianceOfVector2 = variance(vector2);

    const double standardDeviationOfVector1 = sqrt(varianceOfVector1);
    const double standardDeviationOfVector2 = sqrt(varianceOfVector2);

    const double correlation =
        covariance / (standardDeviationOfVector1 * standardDeviationOfVector2);

    return correlation;
}

vector<double> getRangeForVector(vector<double> inputVector) {
    double minItem = inputVector.at(0);
    double maxItem = inputVector.at(0);
    double currItem = inputVector.at(0);
    for (int i = 1; i < inputVector.size(); i++) {
        currItem = inputVector.at(i);
        if (currItem < minItem) {
            minItem = currItem;
        } else if (currItem > maxItem) {
            maxItem = currItem;
        }
    }
    vector<double> range(2);

    range.at(0) = minItem;
    range.at(1) = maxItem;

    return range;
}
void print_stats(vector<double> inputVector) {
    cout << "Sum = " << sumVector(inputVector) << endl;
    cout << "Mean = " << getMeanForVector(inputVector) << endl;
    cout << "Median = " << getMedianForVector(inputVector) << endl;
    vector<double> range = getRangeForVector(inputVector);
    cout << "Range = " << range.at(0) << ", " << range.at(1) << endl;
}

int main(int argc, char **argv) {
    ifstream inputCSV;
    inputCSV.open("Boston.csv");

    string currLine;
    string rm_in, medv_in;
    const int MAX_LEN = 1000;
    vector<double> rm(MAX_LEN);
    vector<double> medv(MAX_LEN);
    if (!inputCSV.is_open()) {
        cout << "Could not open file Boston.csv" << endl;
        return 1;  // return an error
    }

    // read csv data into vectors

    // read heading
    getline(inputCSV, currLine);
    int numObservations = 0;

    while (inputCSV.good()) {
        getline(inputCSV, rm_in,
                ',');  // will get the first item and stop at the comma
        getline(inputCSV, medv_in,
                '\n');  // will get the second item and stop at the newline

        rm.at(numObservations) =
            stof(rm_in);  // converts string to float and inserts into vector
        medv.at(numObservations) =
            stof(medv_in);  // converts string to float and inserts into vector

        numObservations++;
    }

    rm.resize(numObservations);
    medv.resize(numObservations);

    cout << "Closing file Boston.csv." << endl;
    inputCSV.close();  // Done with file, so close it

    cout << "Number of records: " << numObservations << endl;
    cout << endl << "Stats for rm" << endl;
    print_stats(rm);
    cout << endl << "Stats for medv" << endl;
    print_stats(medv);
    cout << endl;
    cout << "Covariance = " << covar(rm, medv) << endl;
    cout << "Correlation = " << cor(rm, medv) << endl;
    cout << "Program terminated." << endl;
    return 0;
}
