# 🛩️ CAIC Summer of Tech (CSoT) – Aerial Robotics
Organized by AeroClub, IIT Delhi

The Aerial Robotics problem statement is part of the CAIC Summer of Tech (CSoT) initiative aimed at introducing first-year students and beginners to the foundational concepts of drone technology and autonomous aerial systems. Emphasizing a simulation-first approach using Webots, an open-source robotics simulator, this five-week journey blends theory with hands-on programming in Python to simulate quadcopter behavior, control, and visual navigation.

Participants will gain core skills in:

- Multirotor physics and dynamics
- PID-based control systems
- Path planning and waypoint navigation
- Real-time computer vision for visual tracking
- Full mission simulation integrating flight and perception

This initiative builds a strong base for future involvement in AeroClub projects and competitions such as Inter-IIT Tech Meet 2025.

## 📅 Weekly Breakdown & Objectives
### Week 1: Introduction to Drone Dynamics and Takeoff
- Understand multirotor basics: thrust, torque, degrees of freedom.

- Learn how quadcopters are modeled in Webots.

- Implement a Python controller for motor initialization and basic takeoff.

Deliverables: takeoff_controller.py, short demo video, README.

### Week 2: Autonomous Waypoint Navigation
- Build on Week 1 to implement waypoint-based navigation (e.g., square pattern).

- Use simple velocity commands (no PID yet) to move the drone in 2D.

- Tasks include defining waypoints, heading control, and safe landing.

Deliverables: waypoint_nav.py, simulation video, brief README.

### Week 3: PID-Based Altitude Control
- Introduce feedback control using a PID controller to maintain altitude.

- Tune PID gains and integrate sensor data (altimeter or position field).

- Understand concepts like oscillation, steady-state error, and overshoot.

Deliverables: pid_altitude.py, tuning explanation, plot of altitude over time.

### Week 4: Visual Navigation and Object Tracking
- Use Webots’ simulated camera to detect a colored marker using OpenCV.

- Apply image processing (HSV thresholding, contour detection).

- Convert marker location into motion commands, enabling autonomous tracking.

- Maintain altitude using the PID controller from Week 3.

Deliverables:

vision_tracking.py with image processing pipeline, 
Demo video showing the drone tracking and centering on a marker, 
Updated README with algorithm description

### Week 5: Integrated Navigate & Inspect Mission
- Combine altitude hold, waypoint navigation, and marker inspection.

- Structure code into reusable modules (takeoff, nav, vision, landing).

- Use a mission_params.json to define altitude, waypoints, and HSV values.

Log performance in mission_log.csv (e.g., waypoint times, marker detections).

Deliverables:

inspect_mission.py, mission_params.json, and mission_log.csv, 
Demo video of complete mission (≤3 mins), 
README describing setup, logic, and sample results

## 🎯 Outcome
By the end of the 5 weeks, participants will:

- Be comfortable with drone control theory and Webots simulation
- Understand real-time feedback systems like PID
- Implement and integrate OpenCV-based visual tracking
- Gain project experience simulating a mission-like autonomous flight

Whether you’re preparing for tech competitions or planning to join AeroClub’s core teams, this program sets a solid foundation in aerial robotics with an emphasis on both practical skills and structured coding.
