Hello, please kindly read all the instructions. Thank you!

I have trained the dataset using my laptop GPU and YOLOv8n NANO model is being used
GPU specification: NVIDIA GeForce RTX 3050 Ti 4gb DDR6 .


Pre-requisites to run the code :

      1) VS Code 
      
      2) Python version 3.10 ( minimum)
      
      3) pip install ultralytics
      
      4) pip install opencv-python
      
      5) pip install pandas
      
Optional Pre-requsites (if you wish to train the datasets on your GPU):

      1) NVIDIA CUDA 12.4 Toolkit ( only if you plan to train the model, i have already provided "best.pt" model which is trained on my GPU ) 
      
      2) pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124 (only if you plan to train the model by yourself )



Google Drive Links to dataset for training, video for script and video output (demo) :

      1) Dataset = 
      
https://drive.google.com/drive/folders/1g14tHUODm5rmkSqffTSxPQVYf0OJ5lf7?usp=drive_link
      
      2) Original Video = 
      
https://drive.google.com/file/d/1s-WQNkZxArbUQHtiqw-n0FsetnbQ3Wj1/view?usp=drive_link
      
      3) Video demo/output =
       
https://drive.google.com/file/d/1lz71sdbi5xgjmyM4bNwqUhebbytOrelU/view?usp=drive_link

---

Instructions to run my code :

      1) Train.py = it is the code to train the YOLOv8n model
      
      2) mainscript.py = it is the main code to detect Beyblades in a video frame and to dtermine the winner and loser between two beyblades 1 v 1 fight and winning beyblade spinning time after winning the game.

Instructions for the outputs :

      1) video output.mp4 = it is the output video which demonstrates that persons are getting detected with the help of best.pt model and user creating an region of interest (ROI) and getting alerts .
      
      2) battle_results.csv = it is a CSV file it contains Total spin duration ( ins seoncds ) of the battle, Winner name, Remainign spin duration in seconds 
      
      3) best.pt = the best.pt is the trained model 

---
