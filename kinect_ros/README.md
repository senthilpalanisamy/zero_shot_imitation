## Description
This directory contains launch files for launching kinect camera node for calibrating
the relative postion between baxter and the kinect camera.

## Library requirements
Rethink workspace for baxter, freenect_stack for kinect 

## Launch file explanation
To make use of this, AR tag must be placed on the table such that the AR tag
is visible in both baxter's camera and kinect camera
1. ar_track_baxter_cam_left.launch - This launch file will launch a ar tag detection
   on the images of baxter's left camera.  The relative pose between the tag and
   baxter's base frame can be found after launching this launch file
2. baxter_table_calibration.launch -  This launch will launch ar tag detection 
  on the images of kinect camera. The relative transform between ar tag and kinect camera
  can be found after launching this node
3. conversion.py - This file can be used for calculating the relative transformmation between 
   baxter and kinect by entering the relative transformation between baxter & ar tag and kinect &
   AR tag.

## Why not a single launch file
An ideal solution would have been to write a publish camera to tag transformation in one case (for example:
baxter left camere to ar tag) and publish tag to camera transformation in the other case (for example: tag to
kinect camera) so the relative transformation between baxter and kinect is automatically calculated by RoS. While
this is ideal, AR tags cannot be parents in a TF tree as per their package construction and a child cannot have two
parents in RoS frame, a fully automatic solution is not possible and hence, a manual approach was followed

