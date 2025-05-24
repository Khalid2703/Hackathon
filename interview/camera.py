import cv2
import mediapipe as mp
import numpy as np
from datetime import datetime
import time
import threading
from mic import record_audio, analyze_audio
import pandas as pd
import os
import json


EXCEL_DIR = r"C:\Users\hp\OneDrive\Desktop"
JSON_DIR = r"C:\Users\hp\OneDrive\Desktop"
LEADERBOARD_PATH = r"C:\Users\hp\OneDrive\Desktop\all_candidates.json"
responses = []
analyses = []

questions = [
    "Tell me about yourself.",
    "Why do you want to work here?",
    "Why should we hire you?",
]

question_idx = 0
current_question = questions[question_idx]
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

def log_event(event):
    with open(proctoring_log_path, "a") as f:
        f.write(f"{datetime.now().isoformat()}: {event}\n")

# --- Threading helpers ---
is_recording = False
is_processing = False
is_paused = False
recording_thread = None
recording_result = {}
candidate_name = input("Enter candidate's name (no spaces): ").strip().replace(" ", "_")
proctoring_log_path = os.path.join(EXCEL_DIR, f"{candidate_name}_proctoring_log.txt")

def record_and_analyze(audio_filename):
    global is_recording, recording_result, is_paused
    record_audio(audio_filename, duration=15)
    is_recording = False  # Recording done
    is_paused = True      # Enter pause state
    time.sleep(1.5)       # Show "Paused..." for 1.5 seconds (adjust as needed)
    is_paused = False
    analysis = analyze_audio(audio_filename)
    recording_result["analysis"] = analysis

def process_analysis():
    global is_processing
    analysis = recording_result.pop("analysis")
    responses.append(analysis["transcription"])
    analyses.append(analysis)
    print("Analysis:", analysis)
    is_processing = False  # Processing done

with mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5) as face_mesh:
    prev_gray = None
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21,21), 0)

        if prev_gray is not None:
            frame_delta = cv2.absdiff(prev_gray, gray)
            thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
            motion_score = np.sum(thresh) / 255
            if motion_score > 5000:
                print(" Motion Detected!")
                log_event(f"Motion Detected: {motion_score:.2f}")
        prev_gray = gray

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(image)

        if results.multi_face_landmarks:
            print("Face detected")
            log_event("Face detected")
            for face_landmarks in results.multi_face_landmarks:
                nose = face_landmarks.landmark[1]
                chin = face_landmarks.landmark[152]
                dx = nose.x - chin.x
                dy = nose.y - chin.y
                angle = np.degrees(np.arctan2(dy, dx))
                if abs(angle) < 10:
                    print("Face is Upright")
                    log_event("Face is Upright")
                else:
                    print("Face is tilted")
                    log_event("Face is tilted")
        else:
            print("No face Detected")
            log_event("No face Detected")

        cv2.putText(frame, current_question, (30,50), cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,255),2)
        if is_recording:
            cv2.putText(frame, "Recording...", (30,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        elif is_paused:
            cv2.putText(frame, "Paused...", (30,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,165,255), 2)
        elif is_processing:
            cv2.putText(frame, "Processing...", (30,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)

        cv2.imshow('Proctoring - Face Detection', frame)
        key = cv2.waitKey(1) & 0xFF

        # Start recording
        if key == ord('r') and not is_recording and not is_processing:
            is_recording = True
            is_processing = False
            recording_result.clear()
            audio_filename = f"responses_{question_idx+1}.wav"
            recording_thread = threading.Thread(target=record_and_analyze, args=(audio_filename,))
            recording_thread.start()

        # Start processing after recording is done
        if not is_recording and "analysis" in recording_result and not is_processing:
            is_processing = True
            threading.Thread(target=process_analysis).start()

        # Move to next question
        if key == ord('n') and not is_recording and not is_processing:
            question_idx += 1
            if question_idx < len(questions):
                current_question = questions[question_idx]
            else:
                break
        elif key == ord('q') and not is_recording and not is_processing:
            break

cap.release()
cv2.destroyAllWindows()

# Save results
# Save results (Excel & JSON)


data = []
for i in range(len(questions)):
    row = {
        "Question": questions[i],
        "Response": responses[i] if i < len(responses) else "",
    }
    if i < len(analyses):
        row.update(analyses[i])
    data.append(row)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
excel_path = fr"{EXCEL_DIR}\interview_{candidate_name}_{timestamp}.xlsx"
json_path = fr"{JSON_DIR}\interview_{candidate_name}_{timestamp}.json"
df = pd.DataFrame(data)
df.to_excel(excel_path, index=False)
with open(json_path, "w") as f:
    json.dump(data, f, indent=2)
print(f"Saved interview to {excel_path} and {json_path}")

# Update leaderboard
try:
    with open(LEADERBOARD_PATH, "r") as f:
        leaderboard = json.load(f)
except FileNotFoundError:
    leaderboard = []

scores = [a.get("final_score", 0) for a in analyses if "final_score" in a]
avg_score = sum(scores) / len(scores) if scores else 0
leaderboard.append({"name": candidate_name, "score": avg_score, "details": analyses})
leaderboard = sorted(leaderboard, key=lambda x: x["score"], reverse=True)
with open(LEADERBOARD_PATH, "w") as f:
    json.dump(leaderboard, f, indent=2)
print("Leaderboard updated!")

# Generate personal report
report_path = os.path.join(EXCEL_DIR, f"{candidate_name}_personal_report.xlsx")
df_report = pd.DataFrame(data)
df_report["Candidate Name"] = candidate_name
df_report["Average Score"] = avg_score
df_report.to_excel(report_path, index=False)
print(f"Personal report saved at {report_path}")

def summarize_proctoring_log(log_path):
    summary = {
        "Face Detected": 0,
        "No Face Detected": 0,
        "Face Upright": 0,
        "Face Tilted": 0,
        "Motion Detected": 0
    }
    try:
        with open(log_path, "r") as f:
            for line in f:
                if "Face detected" in line:
                    summary["Face Detected"] += 1
                elif "No face Detected" in line:
                    summary["No Face Detected"] += 1
                elif "Face is Upright" in line:
                    summary["Face Upright"] += 1
                elif "Face is tilted" in line:
                    summary["Face Tilted"] += 1
                elif "Motion Detected" in line:
                    summary["Motion Detected"] += 1
    except FileNotFoundError:
        pass
    return summary

# --- FIX: Call the function and assign to proctoring_summary ---
proctoring_summary = summarize_proctoring_log(proctoring_log_path)

# Add summary to the report DataFrame (same value for all rows, or just the first row)
for key, value in proctoring_summary.items():
    df_report[key] = value

# Optionally, add a simple behavioral remark
if proctoring_summary["No Face Detected"] == 0 and proctoring_summary["Motion Detected"] == 0 and proctoring_summary["Face Tilted"] < 3:
    behavior_remark = "Candidate maintained good behavior during the interview."
else:
    behavior_remark = "Some proctoring alerts detected. Please review the log."

df_report["Proctoring Remark"] = behavior_remark

# Now save the final report
df_report.to_excel(report_path, index=False)
print(f"Personal report saved at {report_path}")