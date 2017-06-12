#include "ros/ros.h"
#include <std_msgs/Int16.h>
#include <std_msgs/Float32.h>
#include <nav_msgs/Odometry.h>

class auto_control
{
  private:
    ros::NodeHandle nh_;
    ros::NodeHandle priv_nh_;
    ros::Subscriber sub_curvature_;
    ros::Subscriber sub_odom_;
    ros::Subscriber sub_speed_;
    ros::Publisher pub_steering_;
    ros::Publisher pub_speed_;
    float steering_Kp_;
    float steering_Kd_;
    float steering_Ki_;
    float speed_Kp_;
    float maximum_rpm_;
    float minimum_rpm_;
    float maximum_steering_;
    float minimum_steering_;
    float angle;
    float last_angle;
    float head;
    int speed_val;
    bool mFristTime;

    std_msgs::Int16 desired_steering;
    std_msgs::Int16 desired_speed;

  public:
    auto_control(ros::NodeHandle nh) : nh_(nh), priv_nh_("~")
    {
      priv_nh_.param<float>("steering_Kp", steering_Kp_, 2.7);
      priv_nh_.param<float>("steering_Kd", steering_Kd_, 0.0);
      priv_nh_.param<float>("steering_Ki", steering_Ki_, 0.0);
      priv_nh_.param<float>("speed_Kp", speed_Kp_, 0.05);
      priv_nh_.param<float>("maximum_rpm", maximum_rpm_, 1000);
      priv_nh_.param<float>("minimum_rpm", minimum_rpm_, 20);
      priv_nh_.param<float>("maximum_steering", maximum_steering_, 90);
      priv_nh_.param<float>("minimum_steering", minimum_steering_, -90);
      sub_curvature_ = nh_.subscribe( "/model_car/yaw", 1,  &auto_control::headAngleCallback,this);
      sub_odom_ = nh_.subscribe( "/odom", 1,  &auto_control::odomCallback,this);
      sub_speed_ = nh_.subscribe( "/manual_control/speed", 1,  &auto_control::speedCallback,this);


      pub_steering_= nh.advertise<std_msgs::Int16>(nh.resolveName("/manual_control/steering"), 1);
      pub_speed_= nh.advertise<std_msgs::Int16>(nh.resolveName("/manual_control/speed"), 1);
      ROS_INFO("Started control node.");
    }
    ~auto_control(){}
    void headAngleCallback(const std_msgs::Float32 head_angle);
    void odomCallback(const nav_msgs::Odometry odom);
    void speedCallback(const std_msgs::Int16 speed);
};

void auto_control::speedCallback(const std_msgs::Int16 speed)
{
  speed_val = speed.data;
  ROS_INFO("Speed Monitor = %d \n",speed.data);
}

void auto_control::odomCallback(const nav_msgs::Odometry odom)
{
  ROS_INFO("Current Odom %f\n",odom.pose.pose.position.x);
	if (odom.pose.pose.position.x>10.0)
	{
      desired_speed.data=0;
      //ROS_INFO("Speed Monitor = %d, DS: %d \n",speed_val, desired_speed.data);
    	pub_speed_.publish(desired_speed);
    	ROS_INFO("Stop 1m reached");
	}
}
void auto_control::headAngleCallback(const std_msgs::Float32 head_angle)
{
  //int DesiredSpeed=1000;
  //angle based on curvature
  if (mFristTime==false)
  {
    mFristTime=true;
    head=head_angle.data;
    desired_speed.data=-1000;
    pub_speed_.publish(desired_speed);
    ROS_INFO("desired_speed = %d",desired_speed.data);
  }
  angle= head_angle.data - head;
  //ROS_ERROR_STREAM("Control: received angle "<<angle);
  int DesiredSteering;
  DesiredSteering=steering_Kp_*angle+steering_Kd_*((angle-last_angle));///sampleTime);
  last_angle = angle;
  desired_steering.data=DesiredSteering+90;
  pub_steering_.publish(desired_steering);
}

int main(int argc, char **argv)
{
  ros::init(argc, argv, "auto_control_node");
  ros::NodeHandle nh;
  auto_control control1(nh);
   while(ros::ok())
  {
    ros::spinOnce();
  }
  return 0;
}
