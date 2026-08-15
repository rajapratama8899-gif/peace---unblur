import cv2
import mediapipe as mp
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

BLUR_KSIZE = 35


def is_peace_sign(landmarks):
    tips = [8, 12, 16, 20]
    pips = [6, 10, 14, 18]
    up = [landmarks[t].y < landmarks[p].y for t, p in zip(tips, pips)]
    return up[0] and up[1] and not up[2] and not up[3]


class PeaceUnblurProcessor(VideoProcessorBase):
    def __init__(self):
        self.hands = mp_hands.Hands(
            max_num_hands=1, min_detection_confidence=0.7
        )

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = self.hands.process(rgb)

        peace_detected = False
        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                if is_peace_sign(hand_landmarks.landmark):
                    peace_detected = True
                mp_draw.draw_landmarks(
                    img, hand_landmarks, mp_hands.HAND_CONNECTIONS
                )

        if peace_detected:
            output = img.copy()
            label, color = "PEACE terdeteksi - Tajam!", (0, 255, 0)
        else:
            output = cv2.GaussianBlur(img, (BLUR_KSIZE, BLUR_KSIZE), 0)
            label, color = "Tunjukkan peace untuk fokus", (0, 0, 255)

        cv2.putText(
            output, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2
        )
        return av.VideoFrame.from_ndarray(output, format="bgr24")


st.set_page_config(page_title="Peace Unblur Trend", page_icon="✌️")
st.title("✌️ Peace Unblur Trend")
st.write("Tunjukkan gestur peace ke kamera supaya gambar berubah dari blur jadi tajam.")

webrtc_streamer(
    key="peace-unblur",
    video_processor_factory=PeaceUnblurProcessor,
    media_stream_constraints={"video": True, "audio": False},
)
