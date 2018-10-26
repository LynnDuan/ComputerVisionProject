/**
 * @file safe_teleop.cpp
 * @brief Safe teleoperation library implementation
 * Created by rakesh on 28/09/18.
 */
#include <limits>
#include <safe_teleop/safe_teleop.h>

namespace safe_teleop
{

SafeTeleop::SafeTeleop() :
  is_shutdown_(false),
  max_cmd_vel_age_(1.0),
  max_linear_vel_(1.0),
  max_angular_vel_(1.0),
  linear_vel_increment_(0.05),
  angular_vel_increment_(0.05),
  laser_safety_check_angle_(0.25),
  min_safety_impact_time_(0.5),
  min_safety_distance_(0.5),
  linear_vel_(0.0),
  angular_vel_(0.0),
  last_command_timestamp_(0.0)
{
  ros::NodeHandle global_nh;
  cmd_vel_pub_ = global_nh.advertise<geometry_msgs::Twist>("cmd_vel", 5);
  // The subscriber callback is set to the laserScanCallback method of the instantiated object of this class
  laser_scan_sub_ = global_nh.subscribe("scan", 5, &SafeTeleop::laserScanCallback, this);

  run_thread_ = boost::thread(&SafeTeleop::run, this);
  displayCurrentSpeeds();
}

SafeTeleop::~SafeTeleop()
{
  shutdown();
  // wait for the run thread to terminate
  run_thread_.join();

  geometry_msgs::Twist zero_cmd_vel;
  zero_cmd_vel.linear.x = 0;
  zero_cmd_vel.angular.z = 0;
  cmd_vel_pub_.publish(zero_cmd_vel);
}

void SafeTeleop::run()
{
  ros::Rate r(10);
  while (ros::ok() && !is_shutdown_)
  {
    auto current_timestamp = ros::Time::now().toSec();

    auto last_cmd_vel_age = current_timestamp - last_command_timestamp_;

    if (last_cmd_vel_age > max_cmd_vel_age_)
    {
      ROS_WARN_THROTTLE(1.0, "Timeout not implemented\r");
    }
    else
    {
      auto is_safe = checkSafety(static_cast<double>(linear_vel_));

      geometry_msgs::Twist temp;
      temp.linear = linear_vel_;
      ROS_INFO("%s", boost::lexical_cast<std::string>(linear_vel_));
      cmd_vel_pub_.publish(temp);
      ROS_WARN_THROTTLE(1.0, "command velocity publishing not implemented\r");
    }

    r.sleep();
  }
}

void SafeTeleop::moveForward()
{
  linear_vel_.x = 1.0;
  ROS_WARN("Method not implemented\r");
}

void SafeTeleop::moveBackward()
{
  ROS_WARN("Method not implemented\r");
}

void SafeTeleop::rotateClockwise()
{
  ROS_WARN("Method not implemented\r");
}

void SafeTeleop::rotateCounterClockwise()
{
  ROS_WARN("Method not implemented\r");
}

void SafeTeleop::stop()
{
  ROS_WARN("Method not implemented\r");
}


void SafeTeleop::increaseLinearSpeed()
{
  ROS_WARN("Method not implemented\r");
  displayCurrentSpeeds();
}

void SafeTeleop::decreaseLinearSpeed()
{
  ROS_WARN("Method not implemented\r");
  displayCurrentSpeeds();
}

void SafeTeleop::increaseAngularSpeed()
{
  ROS_WARN("Method not implemented\r");
  displayCurrentSpeeds();
}

void SafeTeleop::decreaseAngularSpeed()
{
  ROS_WARN("Method not implemented\r");
  displayCurrentSpeeds();
}

bool SafeTeleop::checkSafety(double linear_vel)
{
  ROS_WARN("Method not implemented\r");
}

} // namespace safe_teleop_node


