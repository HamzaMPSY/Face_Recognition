import cv2
import os
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
	print("Usage:")
	print(" - Press SPACE to capture and save an image.")
	print(" - Press ESC to exit.")

	name = input("Enter your name: ").strip()
	if not name:
		logging.error("Name cannot be empty.")
		return

	dataset_dir = "dataset"
	user_dir = os.path.join(dataset_dir, name)
	os.makedirs(user_dir, exist_ok=True)
	logging.info(f"Images will be saved to: {user_dir}")

	cap = cv2.VideoCapture(0)
	if not cap.isOpened():
		logging.error("Cannot open webcam.")
		return

	i = 1
	while True:
		ret, frame = cap.read()
		if not ret:
			logging.error("Failed to capture frame.")
			break

		cv2.imshow("frame", frame)
		k = cv2.waitKey(10)
		if k == 32:  # SPACE
			img_path = os.path.join(user_dir, f"{name}_{i}.jpg")
			cv2.imwrite(img_path, frame)
			logging.info(f"Saved image: {img_path}")
			i += 1
		elif k == 27:  # ESC
			logging.info("Exiting.")
			break

	cap.release()
	cv2.destroyAllWindows()

if __name__ == "__main__":
	main()
