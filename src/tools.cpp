#include <iostream>
#include "tools.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
  VectorXd rmse = VectorXd::Zero(4);
  VectorXd residual = VectorXd::Zero(4);
  // check the validity of the following inputs:
  //  * the estimation vector size should not be zero
  if (estimations.size() == 0) {
      return rmse;
    }
  if (estimations.size() != ground_truth.size())  {
      return rmse;
    }
  for(int i=0; i < estimations.size(); ++i){
      residual = estimations[i] - ground_truth[i];
      rmse += residual.cwiseProduct(residual);
    }

  //calculate the mean
  rmse *= 1.0 / estimations.size();
  //calculate the squared root
  rmse = rmse.cwiseSqrt();
  //return the result
  return rmse;
}
