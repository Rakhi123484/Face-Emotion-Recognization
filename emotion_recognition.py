import cv2
from deepface import DeepFace

# Initialize the webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Press 'q' to exit the program.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture image.")
        break

    try:
        # Analyze the frame for emotions
        analysis = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
        
        # Handle the case where the result is a list of dictionaries
        if isinstance(analysis, list):
            dominant_emotion = analysis[0]['dominant_emotion']
        else:
            dominant_emotion = analysis['dominant_emotion']

        # Display the emotion on the frame
        cv2.putText(frame, f"Emotion: {dominant_emotion}", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    except Exception as e:
        print(f"Error analyzing frame: {e}")
        cv2.putText(frame, "Emotion: Unable to detect", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Show the frame with the emotion
    cv2.imshow('Emotion Recognition', frame)

    # Exit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close windows
cap.release()
cv2.destroyAllWindows()
