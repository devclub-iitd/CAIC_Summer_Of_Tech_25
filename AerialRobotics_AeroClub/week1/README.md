# 🚁 Week 1: Introduction to Webots and Multirotor Simulation

---

## 1. Objectives

- Understand the components and basic flight dynamics of a quadrotor  
- Install and configure Webots and Python controller support  
- Execute a simple Webots controller to perform take-off, hover, and land

---

## 2. Setup & Prerequisites

### ✅ System Requirements

- **OS:** Windows 10/11, macOS 10.15+, or Ubuntu 18.04+  
- **RAM:** 8 GB minimum (16 GB recommended)  
- **Disk:** ≥ 5 GB free  
- **CPU/GPU:** Any modern multi-core CPU (GPU optional)

---

### 🧭 Install Webots

- Download **Webots 2023a** from:  
  👉 https://cyberbotics.com/#download  
- Follow the installer instructions for your OS

---

### 🐍 Install Python and Dependencies

- Install Python 3.7–3.9 from: https://python.org

#### Create and activate a virtual environment

```bash
python -m venv webots-env

# On Linux/macOS
source webots-env/bin/activate

# On Windows
webots-env\Scripts\activate
```
Install required packages:
 - pip install controller==0.1.0
 - (The controller package provides the Python API for Webots.)


*Obtain Starter World and Controller*
  - Clone or download the AeroClub starter repository (if provided).
  - Alternatively, use Webots’ built-in mavic_2_pro example:
  - In Webots, navigate to File → Open Sample World.
  - Select projects/samples/robots/mavic_2_pro.wbt


*Directory Structure*


 Your project folder should contain:

 week1/

 
├── mavic2pro.wbt  
├── controllers/  
│   └── takeoff_hover_land.py  
└── README.md  

---

## 3. 🚀 Concepts Covered

### ✈️ Lift and Thrust
Rotor speed generates **upward force**. Increasing the RPM (rotations per minute) of the propellers increases lift, allowing the drone to rise.

### 🔁 Orientation Control

- **Roll**: Tilting about the **front-to-rear** axis (longitudinal).
- **Pitch**: Tilting about the **left-to-right** axis (lateral).
- **Yaw**: Rotating about the **vertical** axis (directional heading).

### 🧭 Coordinate Frames

- **World Frame**: A **fixed** coordinate system representing the simulation environment.
- **Body Frame**: A coordinate system **attached to the drone**, which moves with it.

### 🔄 API Control Loop

The standard control loop followed in simulation:



Sensor read → compute commands → send actuation → repeat

---

## 4. 📋 Tasks

### ✅ Load the Starter World

1. Open Webots.
2. Navigate to `worlds/` folder.
3. Load the file `quadcopter.wbt`.
4. Ensure the environment loads correctly and the quadrotor model is visible.

### 🧠 Implement the Python Controller

Create a new controller script `controllers/takeoff_hover_land/takeoff_hover_land.py` with the following objectives:

- **Takeoff**: Increase rotor speeds to lift off.
- **Hover**: Maintain a stable altitude.
- **Land**: Decrease rotor speeds gradually to descend.

### 🔧 Code Structure (Skeleton Logic to Complete)

```
 from controller import Robot

#Create Robot instance
robot = Robot()
timestep = int(robot.getBasicTimeStep())

#Motor device names — adjust if needed
motor_names = ["front left propeller", "front right propeller", "rear left propeller", "rear right propeller"]

#Initialize motors
motors = []

for name in motor_names:
    motor = robot.getDevice(name)
    if motor is None:
        print(f"Error: Motor device '{name}' not found. Please check device names.")
        exit(1)
    motor.setPosition(float('inf'))  # Velocity control mode
    motor.setVelocity(0.0)
    motors.append(motor)

def set_all_motors_velocity(velocity):
  for motor in motors:
       motor.setVelocity(velocity)

#Set motor velocities with opposite directions for proper torque balance
def set_all_motors_velocity(base_velocity):

    #Direction multipliers for motors: +1 or -1 depending on spin direction
    directions = [ - , - , -  , - ]  # Edit and adjust if your drone motor layout differs
    for i, motor in enumerate(motors):
        motor.setVelocity(base_velocity * directions[i])

#Set the Parameters
takeoff_duration = 0.0   #seconds to reach hover velocity
hover_duration = 0.0    #seconds to hover
land_duration = 0.0     #seconds to land
max_velocity =  0.0    #max motor velocity for hover (tune as needed)



#########################################################

     WRITE THE CODE FOR TAKEOFF, HOVER AND LAND HERE

#########################################################


set_all_motors_velocity(0.0)
print("Flight sequence complete.")
```

 ## 5. 🚀 How to Run

### 1. Assign the Controller

- Open your Webots project.
- In the Scene Tree, select the `mavic_2_pro` node.
- Set the controller field to: `takeoff_hover_land`.

### 2. Run the Simulation

- Click the green **Run** button (▶️) in Webots.
- The drone should:
  - Take off to ~1 meter
  - Hover for a few seconds
  - Land smoothly

---

## 6. 🎯 Bonus Challenge (Optional)

### Square Flight Pattern Before Landing

Enhance the controller to:

1. Take off to 1 meter
2. Perform a square pattern:
   - Forward 1 m
   - Right 1 m
   - Backward 1 m
   - Left 1 m
3. Return to starting point
4. Land

Use GPS or internal odometry to track relative motion.

---
## 7. 📚 Resources

- [Webots Installation Guide](https://cyberbotics.com/doc/guide/installing-webots)
- [Python API Reference](https://cyberbotics.com/doc/reference/python-api)
- [Mavic 2 Pro Tutorial](https://cyberbotics.com/doc/guide/mavic-2-pro?version=R2023a)

---

## 8. SUBMISSION
- Submit your work through the google form attached below
- https://forms.office.com/Pages/ResponsePage.aspx?id=S1xNYsVFIkGM0ETw-E6UXcue0fDziu5KmhlObRm5LTlUM01KRjNWVUxIOEpTUUpVQzgyQ1FXQjhTNy4u


