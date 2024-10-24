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

1) Collect Data :

  1.  To acquire the data I cut a specific time stamp video from a YouTube video https://www.youtube.com/watch?v=HlG29zJmodM, time stamp 06.40 to 07.15, after that  I wrote a Python script with its file name 
   image_extract.py, in this script, I wrote code in such a way that it extracted 120 frames from the video file which I extracted from the downloaded YouTube video. Then I took these 120 images and manually labelled them from Makesens.ai website and covnerted them in YOLO format that is the .txt format

---

2) Train data :

   1.  To train the data we need to create a data.yaml file in it we need to give 'NC' number of classes over here there are 3 classes Beyblade 1 Beyblade 2 Beyblade 3 and thier names

![Screenshot (38)](https://github.com/user-attachments/assets/6300b423-73fa-45b9-9663-1c512fd03fb8)


Below I have uploaded a screenshot after training the dataset,I trained in 60 epochs with 16 batch size and it got trained using my Laptops GPU.



![Screenshot (37)](https://github.com/user-attachments/assets/15dc734b-670f-46d9-a43e-3fed0d34f087)

After training my model on the dataset I got a mAP50 OF 0.998 and Precision and Recall for all three Beyblades very high Recall is 100 percent for all three beyblades. precision is also 0.999 for all three Beyblades.


