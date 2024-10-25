from ultralytics import YOLO
import cv2
import numpy as np
import csv

class BeybladeBattleTracker:
    def __init__(self, model_path='best.pt', video_path='Beybladefight.mp4'):
        self.model = YOLO(model_path)
        self.video_path = video_path
        
        # States for battle tracking
        self.battle_started = False
        self.battle_start_frame = None
        self.battle_ended = False
        self.battle_end_frame = None
        self.battle_end_reason = None
        self.stopped_beyblade = None
        
        # Winner tracking
        self.winner = None
        self.winner_stop_frame = None
        self.winner_final_duration = None
        
        # Movement tracking
        self.prev_positions = {'Beyblade 1': None, 'Beyblade 2': None}
        self.stop_frames = {'Beyblade 1': 0, 'Beyblade 2': 0}
        self.STOP_THRESHOLD = 15
        self.MOVEMENT_THRESHOLD = 5
        
        # Colors for visualization
        self.COLORS = {
            'text': (139, 0, 0),
            'box_active': (255, 0, 0),
            'box_static': (0, 0, 255),
            'winner_text': (0, 0, 255)
        }
        
        # Results storage
        self.csv_file = 'battle_results.csv'

    def seconds_to_minsec(self, seconds):
        """Convert decimal seconds to 'MM:SS.xx' format"""
        minutes = int(seconds // 60)
        remaining_seconds = seconds % 60
        return f"{minutes:02d}:{remaining_seconds:05.2f}"
        
    def calculate_movement(self, current_pos, previous_pos):
        if previous_pos is None:
            return float('inf')
        return np.sqrt((current_pos[0] - previous_pos[0])**2 + 
                      (current_pos[1] - previous_pos[1])**2)
    
    def save_results(self, battle_duration, winner, end_reason, winner_spin_duration=None):
        """Save battle results to CSV with formatted time"""
        with open(self.csv_file, 'a', newline='') as file:
            writer = csv.writer(file)
            if file.tell() == 0:
                writer.writerow(['Battle Duration (MM:SS)', 'Winner', 'End Reason', 'Winner Spin Duration (MM:SS)'])
            
            # Format durations as MM:SS
            battle_duration_fmt = self.seconds_to_minsec(battle_duration)
            winner_spin_fmt = self.seconds_to_minsec(winner_spin_duration) if winner_spin_duration is not None else 'N/A'
            
            writer.writerow([battle_duration_fmt, winner, end_reason, winner_spin_fmt])

    def resize_frame(self, frame, target_width=1280):
        height, width = frame.shape[:2]
        aspect = width / height
        target_height = int(target_width / aspect)
        return cv2.resize(frame, (target_width, target_height))
    
    def analyze_battle(self):
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            print(f"Error: Couldn't open video: {self.video_path}")
            return
        
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_count = 0
        
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter(
            'battle_analysis.mp4',
            cv2.VideoWriter_fourcc(*'mp4v'),
            fps,
            (width, height)
        )
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                if self.battle_ended and self.winner and self.winner_final_duration is None:
                    self.winner_final_duration = (frame_count - self.battle_end_frame) / fps
                    self.save_results(
                        (self.battle_end_frame - self.battle_start_frame) / fps,
                        self.winner,
                        self.battle_end_reason,
                        self.winner_final_duration
                    )
                break
            
            frame_count += 1
            current_time = frame_count / fps
            
            output_frame = frame.copy()
            results = self.model(frame, conf=0.3)
            
            detected_beyblades = set()
            current_positions = {}
            
            # Process detections
            for result in results[0]:
                boxes = result.boxes
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    confidence = float(box.conf)
                    class_id = int(box.cls)
                    class_name = result.names[class_id]
                    
                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2
                    current_positions[class_name] = (center_x, center_y)
                    detected_beyblades.add(class_name)
                    
                    color = self.COLORS['box_active'] if class_name != 'Beyblade 3' else self.COLORS['box_static']
                    cv2.rectangle(output_frame, (x1, y1), (x2, y2), color, 2)
                    
                    label = f'{class_name} ({confidence:.2f})'
                    cv2.putText(output_frame, label,
                              (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                              1.0, color, 3)
            
            # Battle start detection
            if not self.battle_started and len(detected_beyblades) >= 2 and \
               'Beyblade 1' in detected_beyblades and 'Beyblade 2' in detected_beyblades:
                self.battle_started = True
                self.battle_start_frame = frame_count
                print(f"Battle Started at {self.seconds_to_minsec(current_time)}!")
            
            # Battle tracking
            if self.battle_started and not self.battle_ended:
                for beyblade in ['Beyblade 1', 'Beyblade 2']:
                    if beyblade in current_positions:
                        movement = self.calculate_movement(
                            current_positions[beyblade],
                            self.prev_positions[beyblade]
                        )
                        
                        if movement < self.MOVEMENT_THRESHOLD:
                            self.stop_frames[beyblade] += 1
                        else:
                            self.stop_frames[beyblade] = 0
                        
                        self.prev_positions[beyblade] = current_positions[beyblade]
                
                # Check for battle end
                for beyblade in ['Beyblade 1', 'Beyblade 2']:
                    if self.stop_frames[beyblade] >= self.STOP_THRESHOLD:
                        self.battle_ended = True
                        self.battle_end_frame = frame_count
                        self.stopped_beyblade = beyblade
                        self.winner = 'Beyblade 2' if beyblade == 'Beyblade 1' else 'Beyblade 1'
                        self.battle_end_reason = f"{self.stopped_beyblade} stopped spinning"
                        print(f"Battle ended at {self.seconds_to_minsec(current_time)}!")
                        
            # Track winner's remaining spin after battle ends
            if self.battle_ended and self.winner and self.winner_final_duration is None:
                if self.winner in detected_beyblades:
                    winner_duration = (frame_count - self.battle_end_frame) / fps
                else:
                    # Winner is no longer detected (picked up or out of frame)
                    self.winner_final_duration = (frame_count - self.battle_end_frame) / fps
                    battle_duration = (self.battle_end_frame - self.battle_start_frame) / fps
                    self.save_results(
                        battle_duration,
                        self.winner,
                        self.battle_end_reason,
                        self.winner_final_duration
                    )
            
            # Display information on frame
            if self.battle_start_frame:
                if not self.battle_ended:
                    current_duration = (frame_count - self.battle_start_frame) / fps
                    duration_text = f'Battle Duration: {self.seconds_to_minsec(current_duration)}'
                else:
                    battle_duration = (self.battle_end_frame - self.battle_start_frame) / fps
                    duration_text = f'Battle Duration: {self.seconds_to_minsec(battle_duration)}'
                cv2.putText(output_frame, duration_text,
                          (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, self.COLORS['text'], 2)
            
            # Add battle status
            status = "Battle Not Started"
            if self.battle_started:
                if self.battle_ended:
                    status = f"Battle Ended - {self.battle_end_reason}"
                else:
                    status = "Battle In Progress"
            cv2.putText(output_frame, f'Status: {status}',
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, self.COLORS['text'], 2)
            
            # Display winner and remaining spin duration
            if self.battle_ended and self.winner:
                winner_text = f'Winner: {self.winner}'
                
                if self.winner_final_duration is not None:
                    spin_duration = self.winner_final_duration
                    spin_text = f'Winner Spin After Battle: {self.seconds_to_minsec(spin_duration)}'
                else:
                    current_spin = (frame_count - self.battle_end_frame) / fps
                    spin_text = f'Winner Current Spin After Battle: {self.seconds_to_minsec(current_spin)}'
                
                text_size = cv2.getTextSize(winner_text, cv2.FONT_HERSHEY_SIMPLEX, 2, 3)[0]
                text_x = width//2 - text_size[0]//2
                text_y = height//2
                
                cv2.rectangle(output_frame, 
                            (text_x - 10, text_y - text_size[1] - 10),
                            (text_x + text_size[0] + 10, text_y + 10),
                            (255, 255, 255), -1)
                
                cv2.putText(output_frame, winner_text,
                          (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 
                          2, self.COLORS['winner_text'], 3)
                
                cv2.putText(output_frame, spin_text,
                          (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 
                          1, self.COLORS['text'], 2)
            
            out.write(output_frame)
            
            display_frame = self.resize_frame(output_frame)
            cv2.imshow('Beyblade Battle Analysis', display_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        out.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    tracker = BeybladeBattleTracker(
        model_path='best.pt',
        video_path='Beybladefight.mp4'
    )
    tracker.analyze_battle()
