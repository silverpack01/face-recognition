import cv2

# Load the pre-trained face cascade classifier
classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


# Function to extract and save faces from an image
def extract_and_save_faces(img, counts):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = classifier.detectMultiScale(gray, 1.1, 3, 5)

    for idx, (x, y, w, h) in enumerate(faces):
        # Adjust face region to include more area
        x -= 10
        y -= 10
        w += 50
        h += 50

        face = gray[y:y + h, x:x + w]

        # Resize the face image
        face_resized = cv2.resize(face, (200, 200))

        # Save the face image
        path = f'dataset/{counts + idx}.jpg'
        cv2.imwrite(path, face_resized)

        # Display the face image
        cv2.putText(face_resized, str(counts + idx), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('cropped', face_resized)

    return len(faces)


cap = cv2.VideoCapture(0)
count = 0

while count <= 100:
    # Capture frame from the webcam
    ret, frame = cap.read()

    if ret:
        # Extract and save faces from the frame
        count += extract_and_save_faces(frame, count)

        key = cv2.waitKey(10)
        if key == 13 or key == 27:  # Exit loop on Enter key press
            break
    else:
        print('Failed to capture frame')

cap.release()
cv2.destroyAllWindows()
print('Dataset collection completed')
