import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
from datetime import datetime
import time
import threading
import pandas as pd
import os
import json
import tempfile
from PIL import Image
import queue

# Import your mic functions (assuming they exist)
try:
    from mic import record_audio, analyze_audio
    MIC_AVAILABLE = True
except ImportError:
    MIC_AVAILABLE = False
    st.warning("mic.py module not found. Audio recording will be simulated.")

# Constants
EXCEL_DIR = tempfile.gettempdir()  # Using temp directory for demo
JSON_DIR = tempfile.gettempdir()

LEADERBOARD_PATH = os.path.join(os.path.dirname(__file__), "all_candidates.json")

QUESTIONS = [
    "Tell me about yourself.",
    "Why do you want to work here?",
    "Why should we hire you?",
]

# Initialize session state
def initialize_session_state():
    if 'page' not in st.session_state:
        st.session_state.page = "Landing Page"
    if 'interview_started' not in st.session_state:
        st.session_state.interview_started = False
    if 'candidate_name' not in st.session_state:
        st.session_state.candidate_name = ""
    if 'current_question_idx' not in st.session_state:
        st.session_state.current_question_idx = 0
    if 'responses' not in st.session_state:
        st.session_state.responses = []
    if 'analyses' not in st.session_state:
        st.session_state.analyses = []
    if 'is_recording' not in st.session_state:
        st.session_state.is_recording = False
    if 'recording_complete' not in st.session_state:
        st.session_state.recording_complete = False

class InterviewProcessor:
    def __init__(self, candidate_name):
        self.candidate_name = candidate_name.strip().replace(" ", "_")
        self.proctoring_log_path = os.path.join(EXCEL_DIR, f"{self.candidate_name}_proctoring_log.txt")
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5
        )
        self.prev_gray = None
        
    def log_event(self, event):
        """Log proctoring events"""
        try:
            with open(self.proctoring_log_path, "a") as f:
                f.write(f"{datetime.now().isoformat()}: {event}\n")
        except:
            pass  # Handle file write errors gracefully
    
    def process_frame(self, frame):
        """Process frame for face detection and motion detection"""
        results = {
            'frame': frame,
            'face_detected': False,
            'face_upright': False,
            'motion_detected': False
        }
        
        try:
            # Convert to grayscale for motion detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (21, 21), 0)
            
            # Motion detection
            if self.prev_gray is not None:
                frame_delta = cv2.absdiff(self.prev_gray, gray)
                thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
                motion_score = np.sum(thresh) / 255
                if motion_score > 5000:
                    results['motion_detected'] = True
                    self.log_event(f"Motion Detected: {motion_score:.2f}")
            
            self.prev_gray = gray
            
            # Face detection and analysis
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_results = self.face_mesh.process(image)
            
            if face_results.multi_face_landmarks:
                results['face_detected'] = True
                self.log_event("Face detected")
                
                for face_landmarks in face_results.multi_face_landmarks:
                    nose = face_landmarks.landmark[1]
                    chin = face_landmarks.landmark[152]
                    dx = nose.x - chin.x
                    dy = nose.y - chin.y
                    angle = np.degrees(np.arctan2(dy, dx))
                    
                    if abs(angle) < 10:
                        results['face_upright'] = True
                        self.log_event("Face is Upright")
                    else:
                        self.log_event("Face is tilted")
            else:
                self.log_event("No face Detected")
        except Exception as e:
            st.error(f"Error processing frame: {e}")
        
        return results

def simulate_audio_analysis():
    """Simulate audio analysis when mic.py is not available"""
    return {
        "transcription": "This is a simulated response for demonstration purposes. I am excited about this opportunity.",
        "word_count": 15,
        "filler_count": 1,
        "keyword_score": 3,
        "confidence_score": 75,
        "sentiment": "POSITIVE",
        "sentiment_score": 0.8,
        "final_score": 7.5
    }

def record_and_analyze_audio(question_idx):
    """Record and analyze audio response"""
    if MIC_AVAILABLE:
        try:
            audio_filename = f"temp_response_{question_idx}.wav"
            record_audio(audio_filename, duration=15)
            analysis = analyze_audio(audio_filename)
            # Clean up temp file
            if os.path.exists(audio_filename):
                os.remove(audio_filename)
            return analysis
        except Exception as e:
            st.error(f"Audio recording failed: {e}")
            return simulate_audio_analysis()
    else:
        return simulate_audio_analysis()

def save_interview_results(candidate_name, responses, analyses):
    """Save interview results to Excel and JSON"""
    data = []
    for i in range(len(QUESTIONS)):
        row = {
            "Question": QUESTIONS[i],
            "Response": responses[i] if i < len(responses) else "",
        }
        if i < len(analyses):
            row.update(analyses[i])
        data.append(row)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    excel_path = os.path.join(EXCEL_DIR, f"interview_{candidate_name}_{timestamp}.xlsx")
    json_path = os.path.join(JSON_DIR, f"interview_{candidate_name}_{timestamp}.json")
    
    try:
        df = pd.DataFrame(data)
        df.to_excel(excel_path, index=False)
        
        with open(json_path, "w") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        st.error(f"Error saving results: {e}")
        return None, None
    
    return excel_path, json_path

def update_leaderboard(candidate_name, analyses):
    """Update the leaderboard with new candidate results"""
    try:
        with open(LEADERBOARD_PATH, "r") as f:
            leaderboard = json.load(f)
    except FileNotFoundError:
        leaderboard = []
    
    scores = [a.get("final_score", 0) for a in analyses if "final_score" in a]
    avg_score = sum(scores) / len(scores) if scores else 0
    
    leaderboard.append({
        "name": candidate_name,
        "score": avg_score,
        "details": analyses
    })
    
    leaderboard = sorted(leaderboard, key=lambda x: x["score"], reverse=True)
    
    try:
        with open(LEADERBOARD_PATH, "w") as f:
            json.dump(leaderboard, f, indent=2)
    except Exception as e:
        st.error(f"Error updating leaderboard: {e}")
        return 0
    
    return avg_score

def landing_page():
    """Landing page with candidate name input"""
    st.header("Welcome! Start your interview below.")
    
    with st.form("candidate_form"):
        candidate_name = st.text_input("Enter your name (no spaces):", key="candidate_input")
        submitted = st.form_submit_button("🎯 Start Interview", type="primary")
        
        if submitted and candidate_name:
            st.session_state.candidate_name = candidate_name
            st.session_state.interview_started = True
            st.session_state.page = "Interview"  # Automatically switch to interview page
            st.success(f"Interview started for {candidate_name}!")
            st.rerun()
        elif submitted and not candidate_name:
            st.error("Please enter your name to start the interview.")
    
    # Show leaderboard
    st.subheader("📊 Current Leaderboard")
    if os.path.exists(LEADERBOARD_PATH):
        try:
            with open(LEADERBOARD_PATH, "r") as f:
                leaderboard = json.load(f)
            if leaderboard:
                df = pd.DataFrame(leaderboard)[["name", "score"]]
                df.index = df.index + 1
                st.dataframe(df, use_container_width=True)
            else:
                st.info("No interviews completed yet.")
        except:
            st.info("No interviews completed yet.")
    else:
        st.info("Leaderboard will appear here after the first interview.")

def interview_page():
    """Main interview page with video feed and controls"""
    if not st.session_state.interview_started:
        st.warning("Please start an interview from the Landing Page first.")
        return

    # Initialize processor if not already done
    if 'processor' not in st.session_state or st.session_state.candidate_name != getattr(st.session_state.get('processor', None), 'candidate_name', None):
        st.session_state.processor = InterviewProcessor(st.session_state.candidate_name)

    st.header(f"🎤 Interview Session: {st.session_state.candidate_name}")
    
    # Create columns for video and controls
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📹 Video Feed")
        camera_image = st.camera_input("Camera", key="camera")
        if camera_image is not None:
            img = Image.open(camera_image)
            frame = np.array(img)
            # Process frame as before
            results = st.session_state.processor.process_frame(frame)
            
            # Add overlays
            current_question = QUESTIONS[st.session_state.current_question_idx]
            cv2.putText(frame, current_question, (30, 50), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            # Recording status
            if st.session_state.is_recording:
                cv2.putText(frame, "Recording...", (30, 90), 
                          cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # Face detection status
            if results['face_detected']:
                cv2.putText(frame, "Face Detected", (30, 130), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "No Face", (30, 130), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Convert BGR to RGB for Streamlit
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            st.image(frame_rgb, channels="RGB", use_container_width=True)
    
    with col2:
        st.subheader("🎙️ Interview Controls")
        
        # Current question display
        st.info(f"**Question {st.session_state.current_question_idx + 1}/{len(QUESTIONS)}:**\n\n{QUESTIONS[st.session_state.current_question_idx]}")
        
        # Recording controls
        if not st.session_state.is_recording and not st.session_state.recording_complete:
            if st.button("🎙️ Record Answer", key="record_btn", type="primary"):
                st.session_state.is_recording = True
                st.session_state.recording_start_time = time.time()
                st.rerun()

        elif st.session_state.is_recording:
            elapsed = time.time() - st.session_state.get("recording_start_time", time.time())
            st.info(f"Recording... {int(elapsed)}s / 15s")
            if elapsed >= 15:
                # Simulate or process audio here
                analysis = record_and_analyze_audio(st.session_state.current_question_idx)
                st.session_state.responses.append(analysis["transcription"])
                st.session_state.analyses.append(analysis)
                st.session_state.is_recording = False
                st.session_state.recording_complete = True
                st.success("✅ Recording completed!")
                st.rerun()
            else:
                st.rerun()
        
        elif st.session_state.recording_complete:
            st.success("✅ Answer recorded successfully!")
            
            # Show transcription preview
            if st.session_state.responses:
                with st.expander("View your response"):
                    st.write(st.session_state.responses[-1][:200] + "..." if len(st.session_state.responses[-1]) > 200 else st.session_state.responses[-1])
            
            # Next question or finish
            if st.session_state.current_question_idx < len(QUESTIONS) - 1:
                if st.button("➡️ Next Question", key="next_btn"):
                    st.session_state.current_question_idx += 1
                    st.session_state.recording_complete = False
                    st.rerun()
            
            if st.button("✅ Finish Interview", key="finish_btn", type="primary"):
                finish_interview()
        
        # Progress indicator
        progress = (st.session_state.current_question_idx + 1) / len(QUESTIONS)
        st.progress(progress)
        st.caption(f"Progress: {st.session_state.current_question_idx + 1}/{len(QUESTIONS)} questions")
        
        # Response summary
        if st.session_state.responses:
            st.subheader("📝 Responses Summary")
            st.write(f"Completed: {len(st.session_state.responses)} / {len(QUESTIONS)} questions")

def finish_interview():
    """Finish the interview and save results"""
    if len(st.session_state.responses) > 0:
        # Save results
        excel_path, json_path = save_interview_results(
            st.session_state.candidate_name,
            st.session_state.responses,
            st.session_state.analyses
        )
        
        # Update leaderboard
        avg_score = update_leaderboard(
            st.session_state.candidate_name,
            st.session_state.analyses
        )
        
        st.success(f"🎉 Interview completed! Average score: {avg_score:.2f}")
        if excel_path:
            st.info(f"📄 Results saved to: {excel_path}")
        
        # Reset session
        reset_interview_session()
        st.session_state.page = "Results"
        st.rerun()
    else:
        st.warning("Please record at least one answer before finishing.")

def reset_interview_session():
    """Reset interview session variables"""
    st.session_state.interview_started = False
    st.session_state.current_question_idx = 0
    st.session_state.responses = []
    st.session_state.analyses = []
    st.session_state.is_recording = False
    st.session_state.recording_complete = False
    
    # Release camera
    if 'cap' in st.session_state and st.session_state.cap:
        st.session_state.cap.release()
        del st.session_state.cap

def results_page():
    """Results and leaderboard page"""
    st.header("📈 Interview Results & Leaderboard")
    
    if os.path.exists(LEADERBOARD_PATH):
        try:
            with open(LEADERBOARD_PATH, "r") as f:
                leaderboard = json.load(f)
            
            if leaderboard:
                # Display leaderboard
                st.subheader("🏆 Leaderboard")
                df = pd.DataFrame(leaderboard)[["name", "score"]]
                df.index = df.index + 1
                st.dataframe(df, use_container_width=True)
                
                # Detailed view
                st.subheader("📊 Detailed Analysis")
                selected_candidate = st.selectbox("Select candidate for details:", 
                                                [entry["name"] for entry in leaderboard])
                
                candidate_data = next(entry for entry in leaderboard if entry["name"] == selected_candidate)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Overall Score", f"{candidate_data['score']:.2f}")
                
                with col2:
                    st.metric("Number of Responses", len(candidate_data['details']))
                
                # Show individual response scores
                if candidate_data['details']:
                    scores_df = pd.DataFrame([
                        {
                            "Response": i+1,
                            "Final Score": detail.get('final_score', 0),
                            "Confidence": detail.get('confidence_score', 0),
                            "Sentiment": detail.get('sentiment', 'N/A')
                        }
                        for i, detail in enumerate(candidate_data['details'])
                    ])
                    st.dataframe(scores_df, use_container_width=True)
            else:
                st.info("No interviews completed yet.")
        except Exception as e:
            st.error(f"Error loading results: {e}")
    else:
        st.info("No interviews completed yet. Complete an interview to see results here.")
    
    if st.button("🏠 Back to Home"):
        st.session_state.page = "Landing Page"
        st.rerun()

def main():
    st.set_page_config(
        page_title="AI Interviewer", 
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize session state
    initialize_session_state()
    
    # Main title
    st.title("🤖 AI Interviewer System")
    
    # Sidebar for navigation
    with st.sidebar:
        st.header("Navigation")
        
        # Manual page selection
        selected_page = st.radio(
            "Go to:", 
            ["Landing Page", "Interview", "Results"],
            index=["Landing Page", "Interview", "Results"].index(st.session_state.page)
        )
        
        # Update page if manually changed
        if selected_page != st.session_state.page:
            st.session_state.page = selected_page
            st.rerun()
        
        # Show current status
        if st.session_state.interview_started:
            st.success(f"👤 Candidate: {st.session_state.candidate_name}")
            st.info(f"📊 Progress: {len(st.session_state.responses)}/{len(QUESTIONS)} completed")
    
    # Route to appropriate page
    if st.session_state.page == "Landing Page":
        landing_page()
    elif st.session_state.page == "Interview":
        interview_page()
    elif st.session_state.page == "Results":
        results_page()

if __name__ == "__main__":
    main()