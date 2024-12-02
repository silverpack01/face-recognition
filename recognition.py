import cv2

# Load the classifiers
classifier_frontalface_alt_tree = cv2.CascadeClassifier('haarcascade_frontalface_alt_tree.xml')
classifier_profileface = cv2.CascadeClassifier('haarcascade_profileface.xml')

# Load the LBPH face recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('lbph_model.yml')

# Initialize video capture
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    if not ret:
        print('Failed to capture frame.')
        continue

    # Convert frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces using all classifiers
    faces_frontalface_alt_tree = classifier_frontalface_alt_tree.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    faces_profileface = classifier_profileface.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    # Concatenate detected faces
    all_faces = list(faces_frontalface_alt_tree) + list(faces_profileface)

    # Process each detected face
    for (x, y, w, h) in all_faces:
        roi_gray = gray[y:y + h, x:x + w]

        # Perform face recognition
        id_, conf = recognizer.predict(roi_gray)
        if conf > 50 or conf < 85:
            print(id_)

        # Save face image
        img_item = 'my_image.png'
        cv2.imwrite(img_item, roi_gray)

        # Draw rectangle around the face
        color = (0, 255, 0)
        stroke = 2
        end_cord_x = x + w
        end_cord_y = y + h
        cv2.rectangle(frame, (x, y), (end_cord_x, end_cord_y), color, stroke)

    # Display the frame
    cv2.imshow('frame', frame)

    # Break the loop if 'Enter' or 'Esc' key is pressed
    key = cv2.waitKey(20)
    if key == 13 or key == 27:
        break

# Release video capture and close all windows
cap.release()
cv2.destroyAllWindows()