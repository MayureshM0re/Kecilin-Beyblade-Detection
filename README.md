Hello, please kindly read all the instructions. Thank you!

I have trained the dataset using my laptop GPU and YOLOv8n NANO model is being used
GPU specification: NVIDIA GeForce RTX 3050 Ti 4gb DDR6 .


Pre-requisites to run the code :

      1) VS Code 
      
      2) Python version 3.10 ( minimum)
      
      3) pip install ultralytics
      
      4) pip install opencv-python
      
      5) pip install numpy
      
Optional Pre-requsites (if you wish to train the datasets on your GPU):

      1) NVIDIA CUDA 12.4 Toolkit ( only if you plan to train the model, i have already provided "best.pt" model which is trained on my GPU ) 
      
      2) pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124 (only if you plan to train the model by yourself )



Google Drive Links to dataset for training, video for script and video output (demo) :

      1) Dataset = 
      
https://drive.google.com/drive/folders/1g14tHUODm5rmkSqffTSxPQVYf0OJ5lf7?usp=drive_link
      
      2) Original Video = 
      
https://drive.google.com/file/d/1s-WQNkZxArbUQHtiqw-n0FsetnbQ3Wj1/view?usp=drive_link
      
      3) Output Video =
      
https://drive.google.com/file/d/1Ztcj2DGx0dk14RJ7GCmnFF1Gs9pe2ITp/view?usp=drive_link

---

Instructions to run my code :

      1) Train.py = it is the code to train the YOLOv8n model
      
      2) mainscript.py = it is the main code to detect Beyblades in a video frame and to dtermine the winner and loser between two beyblades 1 v 1 fight and winning beyblade spinning time after winning the game.

Instructions for the outputs :

      1) battle_analysis.mp4 = it is the output video that demonstrates that Beyblades are getting detected with the help of best.pt model, duration of batlle between , winner of Beybalde battle and the reason 
         for winnign the battle.
      
      2) battle_results.csv = It is a CSV file that contains Battle duration ( ins seconds ), Winner name, Battle end reason
      
      3) best.pt = the best.pt is the trained model 

---

1) Collect Data :

  1.  To acquire the data I cut a specific time stamp video from a YouTube video https://www.youtube.com/watch?v=HlG29zJmodM, time stamp 06.40 to 07.15, after that  I wrote a Python script with its file name 
   image_extract.py, in this script, I wrote code in such a way that it extracted 120 frames from the video file which I extracted from the downloaded YouTube video. Then I took these 120 images and manually labelled them from Makesens.ai website and covnerted them in YOLO format that is the .txt format and then I arranged the dataset in train and val folders

---

2) Train data :

   1.  To train the data we need to create a data.yaml file in it we need to give 'NC' number of classes over here there are 3 classes Beyblade 1 Beyblade 2 Beyblade 3 and thier names

![Screenshot (38)](https://github.com/user-attachments/assets/6300b423-73fa-45b9-9663-1c512fd03fb8)


Below I have uploaded a screenshot after training the dataset,I trained in 60 epochs with 16 batch size and it got trained using my Laptops GPU.



![Screenshot (37)](https://github.com/user-attachments/assets/15dc734b-670f-46d9-a43e-3fed0d34f087)

After training my model I got mAP50: Mean Average Precision at an IoU threshold of 0.5 (a common metric for object detection accuracy) The mAP50 is 100% (or 1.000) for the "all" class, which indicates perfect accuracy in object detection at the IoU 0.5 threshold.

mAP50-95 values for these classes are 0.755, 0.761, and 0.712 respectively, indicating that Beyblade 2 has the best detection performance, while Beyblade 3 is slightly less accurate but still performs well.

The mAP50-95 is 0.756 overall, meaning the model is reasonably accurate across a range of IoU thresholds. This is a very good score for object detection models.


---

3) ###Code Logic walkthrough### :

---

1) :

![1](https://github.com/user-attachments/assets/1cf3f0e7-18cf-4318-9ad8-31827dd3459b)


This initialization block sets up the core components of the tracker:

Loads the YOLO model for Beyblade detection
Initializes battle state variables for tracking start/end conditions
Sets up winner tracking variables
Configures movement tracking with thresholds for stop detection
Establishes dictionaries for position and stop frame tracking

---

2)  :


![2](https://github.com/user-attachments/assets/39936b49-33cf-4d03-9681-117605f27485)


The movement calculation function:

Computes the Euclidean distance between current and previous positions
Returns infinity for first detection to prevent false stops
Uses the Pythagorean theorem to calculate actual pixel distance moved

The second function handles battle result storage:

Creates/appends to a CSV file
Adds headers if file is new
Formats durations in MM:SS format
Stores battle duration, winner, end reason, and winner's spin duration

---

3) 
   
![3](https://github.com/user-attachments/assets/ece7c803-bf9e-43fa-a01e-af12172b81cb)


YOLO detection processing:

Extracts bounding box coordinates for each detected Beyblade
Calculates center points for movement tracking
Records detection confidence and class information
Draws colored bounding boxes around detected Beyblades
Maintains list of currently detected Beyblades

For each frame:

YOLO model detects Beyblades
Bounding box coordinates are extracted
Center point is calculated as the average of box corners
Position is stored in current_positions dictionary

Over here I have opted for an algorithm, the algorithm uses the center point (centroid) of each detected object’s bounding box in each frame to estimate and track movement. Tracking is accomplished by:

Calculating each detected beyblade’s center in the frame.
Storing these centers over consecutive frames to observe motion or lack thereof.

---


4) 

![4](https://github.com/user-attachments/assets/eb465fa8-d21f-4bfe-ac9d-232e27d17747)

The tracking process follows these steps:

Position Comparison

Gets current position from YOLO detection
Compares with previous position
Calculates Euclidean distance between points


Movement Detection

If movement < MOVEMENT_THRESHOLD (5 pixels):

Increment stop_frames counter


If movement ≥ MOVEMENT_THRESHOLD:

Reset stop_frames counter to 0


Update previous position for next frame


Stop Detection

If stop_frames reaches STOP_THRESHOLD (15 frames):

Beyblade is considered stopped
Battle end is triggered

The Euclidean distance formula is used:  distance = √[(x₂-x₁)² + (y₂-y₁)²]          
Where:
(x₁,y₁) is the previous position
(x₂,y₂) is the current position
Thresholds Explained

MOVEMENT_THRESHOLD = 5

Minimum pixel distance to consider real movement
Helps filter out:

Camera noise
Detection jitter
Small vibrations


Values less than 5 pixels are considered stationary


STOP_THRESHOLD = 15

Number of consecutive low-movement frames
Must be reached to declare a Beyblade stopped
At 30 FPS video:

15 frames = 0.5 seconds
Prevents false stops from momentary pauses

---

5) 

![5](https://github.com/user-attachments/assets/948d3dd1-a666-495c-83ab-22d6cc2d64b2)

Check for Battle End
This block detects when the battle has ended based on beyblade inactivity:

Loop through Beyblades: For each beyblade ('Beyblade 1' and 'Beyblade 2'), the code checks if self.stop_frames[beyblade] has reached or exceeded self.STOP_THRESHOLD, meaning the beyblade has stopped spinning for a specific number of frames.
End Battle Trigger: If any beyblade meets this stop condition:
self.battle_ended is set to True, and self.battle_end_frame records the current frame count, marking the end.
self.stopped_beyblade stores which beyblade stopped, and self.winner is set to the other beyblade (the one still spinning).
The reason for the battle’s end (self.battle_end_reason) is set to indicate that the stopped beyblade "stopped spinning."
End Notification: A message is printed to confirm the battle end and display the timestamp.

Track Winner’s Remaining Spin After Battle Ends
This block measures how long the winning beyblade continues to spin after the other has stopped.

Check for Battle End: If self.battle_ended is True, the winner exists (self.winner is not None), and the winner's final duration (self.winner_final_duration) hasn’t been recorded, the code performs additional checks.
Track Remaining Spin:
If the winning beyblade is still detected in detected_beyblades, it calculates winner_duration, which is the time in seconds since the other beyblade stopped.
If the winner is no longer detected (likely removed from the frame), it records the winner_final_duration as the last calculated remaining spin duration and calculates the battle_duration.
Save Results: Finally, the save_results function is called to store the battle’s details, including:
Total battle_duration
The winner beyblade
The battle_end_reason (e.g., one beyblade stopping)
The winner_final_duration (remaining spin time of the winner after the battle ended)

   ---


6) 


![6](https://github.com/user-attachments/assets/9fda383a-fb93-4cc9-8041-0c85c883ef1d)


Initialize Status: The status variable is initially set to "Battle Not Started" to represent the default state before any detection occurs.

Check Battle State:

If self.battle_started is True: This means the battle has started.
If self.battle_ended is also True: The battle has concluded, so status is updated to indicate that the battle has ended, along with self.battle_end_reason (e.g., which beyblade stopped spinning).
Else: If the battle has started but not ended, status is updated to "Battle In Progress".

Display Status on Frame:
cv2.putText is used to draw the status text onto output_frame.
This overlay provides real-time feedback on the battle’s state for easy visualization in the video output.
This ensures that each frame of the video contains an updated message about the battle's current status.

---
7)


![7](https://github.com/user-attachments/assets/21412e36-5aa3-466f-8b75-937bc53a4863)

Display Winner and Spin Duration:

Checks if self.battle_ended and self.winner are both defined, indicating that the battle has concluded and there’s a winner.
Winner Text: Creates a string winner_text to display the winner's name.
Spin Duration Text:
If self.winner_final_duration is not None, it means the spin has completely stopped, and this final spin duration is displayed in spin_text.
Otherwise, calculates the current spin time since the battle ended and stores it in current_spin, updating spin_text to show this ongoing spin time.

out.write(output_frame): Writes the updated frame to the output video file.
display_frame = self.resize_frame(output_frame): Resizes the frame for better display.
cv2.imshow('Beyblade Battle Analysis', display_frame): Shows the frame with the battle analysis in a window.

Release Resources:

When the loop ends, releases cap (video capture), out (video writer), and all OpenCV windows with cv2.destroyAllWindows().



   


   



