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

3) ###Code Logic walkthrough :

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


3) Helper functions, optical flow calculation, estimate the remaining spin duration :
   
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

---


4) 



Video Capture: Captures each frame and resizes it to 640x360 for processing.
YOLO Model: Runs inference to detect Beyblades, returning bounding boxes (bbox) and their classes (Beyblade 1 and Beyblade 2).
Not considering Beyblade 3 here cause it is stationary in this video and it is not present in battle fight.

---


5) Track motion using optical flow , Detect when a Beyblade stops

   ![5](https://github.com/user-attachments/assets/bb0f7d2e-60b2-4ebe-ac5e-d0473852691e)


   For each detected Beyblade, it calculates the displacement between frames and tracks the motion over a window of 30 frames.
   If the average motion in the last 30 frames falls below the threshold, the Beyblade is considered stopped.

   ---


6) Declaring the winner displaying.

   


![6](https://github.com/user-attachments/assets/38214656-8827-43e3-9f38-a54d8253891b)

When one Beyblade stops, the other is declared the winner, and the results are saved (screenshots and CSV entries).
The annotated frame is displayed in a window, and pressing 'q' allows the user to exit the video early.



   


   



