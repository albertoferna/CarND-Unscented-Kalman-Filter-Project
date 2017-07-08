#include "ukf.h"
#include "Eigen/Dense"
#include <iostream>

#define EPS 0.0000001 // A small number

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {
  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  // initial state vector
  x_ = VectorXd::Zero(5);

  // initial covariance matrix
  P_ = MatrixXd::Zero(5, 5);

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 5;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 0.1;

  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;

  /**
  TODO:

  Complete the initialization. See ukf.h for other member properties.

  Hint: one or more values initialized above might be wildly off...
  */
  is_initialized_ = false;
  n_x_ = 5;
  n_aug_ = 7;
  lambda_ = 3 - n_aug_;
  Xsig_pred_ = MatrixXd::Zero(n_x_, 2 * n_aug_ + 1);
  Xsig_aug_ = MatrixXd::Zero(n_aug_, 2 * n_aug_ + 1);
  weights_ = VectorXd(2*n_aug_+1);
  weights_.fill(1/(2*(lambda_+n_aug_)));
  weights_(0) *= 2 * lambda_;
}

UKF::~UKF() {}

/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
  /**
  TODO:

  Complete this function! Make sure you switch between lidar and radar
  measurements.
  */
  /*****************************************************************************
   *  Initialization
   ****************************************************************************/
  if (!is_initialized_) {
    // first measurement
    cout << "UKF: " << endl;
    x_ = VectorXd::Zero(5);

    if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
      // auxiliary variables
      float rho_measured = meas_package.raw_measurements_[0];
      float phi_measured = meas_package.raw_measurements_[1];
      float rhodot_measured = meas_package.raw_measurements_[2];
      // convert to cartesian coord
      float pos_x = rho_measured * cos(phi_measured);
      float pos_y = rho_measured * sin(phi_measured);
      // we don't know the full velocity, just rhodot. Our best guess is rhodot = vel_abs
      float vel_abs = rhodot_measured;
      float yaw_angle = phi_measured;
      // we lack info on way_rate at this point. Let's assume object is going straight
      float yaw_rate = 0;
      x_ << pos_x, pos_y, vel_abs, yaw_angle, yaw_rate;
    }
    else if (meas_package.sensor_type_ == MeasurementPackage::LASER) {
      // we only know position
      x_ << meas_package.raw_measurements_(0,0), meas_package.raw_measurements_(1,0), 0.0, 0.0, 0.0;
    }
    time_us_ = meas_package.timestamp_;
    // done initializing, no need to predict or update
    is_initialized_ = true;
    return;
  }
  // get time and predict
  double delta_t = meas_package.timestamp_ - time_us_;
  Prediction(delta_t);
  if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
    UpdateRadar(meas_package);
  } else {
    UpdateLidar(meas_package);
  }
}

/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Prediction(double delta_t) {
  //create augmented mean vector
  VectorXd x_aug = VectorXd(n_aug_);
  //create augmented state covariance
  MatrixXd P_aug = MatrixXd::Zero(n_aug_, n_aug_);

  //create augmented mean state
  x_aug.head(n_x_) = x_;
  x_aug.tail(n_aug_ - n_x_).fill(0.0);
  //create augmented covariance matrix
  MatrixXd Q(n_aug_ - n_x_, n_aug_ - n_x_);
  Q << std_a_ * std_a_ , 0.0,
       0.0, std_yawdd_ * std_a_;
  P_aug.topLeftCorner(n_x_, n_x_) = P_;
  P_aug.bottomRightCorner(n_aug_ - n_x_, n_aug_ - n_x_) = Q;

  //create square root matrix
  MatrixXd A = P_aug.llt().matrixL();

  //create augmented sigma points
  Xsig_aug_.col(0) = x_aug;
  for (int i = 0; i < n_aug_; i++)
  {
    Xsig_aug_.col(i+1)     = x_aug + sqrt(lambda_ + n_aug_) * A.col(i);
    Xsig_aug_.col(i+1+n_aug_) = x_aug - sqrt(lambda_ + n_aug_) * A.col(i);
  }

  //predict sigma points
  VectorXd aug_state_k = VectorXd(n_aug_);
  Xsig_pred_.fill(0.0);
  VectorXd noise_k(n_x_);
  for (int i=0; i< 2 * n_aug_ + 1; i++) {
      aug_state_k = Xsig_aug_.col(i);
      noise_k.fill(0.0);
      noise_k(0) = 0.5 * delta_t * delta_t * cos(aug_state_k(3)) * aug_state_k(5);
      noise_k(1) = 0.5 * delta_t * delta_t * sin(aug_state_k(3)) * aug_state_k(5);
      noise_k(2) = delta_t * aug_state_k(5);
      noise_k(3) = 0.5 * delta_t * delta_t * aug_state_k(6);
      noise_k(4) = delta_t * aug_state_k(6);
      if (aug_state_k(4) < 0.00001) {
          Xsig_pred_.col(i) = aug_state_k.head(5) + noise_k;
          Xsig_pred_(0, i) += aug_state_k(2) * cos(aug_state_k(3)) * delta_t;
          Xsig_pred_(1, i) += aug_state_k(2) * sin(aug_state_k(3)) * delta_t;
          Xsig_pred_(3, i) += aug_state_k(4) * delta_t;
      } else {
          Xsig_pred_.col(i) = aug_state_k.head(5) + noise_k;
          Xsig_pred_(0, i) += aug_state_k(2)/aug_state_k(4) * (sin(aug_state_k(3) + aug_state_k(4) * delta_t) - sin(aug_state_k(3)));
          Xsig_pred_(1, i) += aug_state_k(2)/aug_state_k(4) * (-cos(aug_state_k(3) + aug_state_k(4) * delta_t) + cos(aug_state_k(3)));
          Xsig_pred_(3, i) += aug_state_k(4) * delta_t;
      }
  }
  // Predict mean and Covariance
  x_ = Xsig_pred_ * weights_;
  P_.fill(0.0);
  for (int i=0; i<2*n_aug_+1; i++) {
      P_ += ((Xsig_pred_.col(i) - x_) * (Xsig_pred_.col(i) - x_).transpose()) * weights_(i);
  }
}

/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(MeasurementPackage meas_package) {
  /**
  TODO:

  Complete this function! Use lidar data to update the belief about the object's
  position. Modify the state vector, x_, and covariance, P_.

  You'll also need to calculate the lidar NIS.
  */
}

/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(MeasurementPackage meas_package) {
  /**
  TODO:

  Complete this function! Use radar data to update the belief about the object's
  position. Modify the state vector, x_, and covariance, P_.

  You'll also need to calculate the radar NIS.
  */

}

