To run code:
1. `rosrun baxter_learner setup_baxter` (sets up baxter with all initialisations)
2. `rosrun baxter_interface joint_trajectory_action_server.py` (starts joint trajectory server)
3. `roslaunch baxter_moveit_config baxter_grippers.launch` (Launch move it for baxter)
4. `rosrun baxter_learner poking_with_arm` (Command baxter to poke a object with a specified action)
