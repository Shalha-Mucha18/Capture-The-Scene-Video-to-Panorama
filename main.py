import cv2
import os

def extract_frames(video_path, output_folder):
    
    #extract frames from a video file and save them as images.
    
    os.makedirs(output_folder, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_path = os.path.join(output_folder, f"frame_{count:04d}.jpg")
        cv2.imwrite(frame_path, frame)
        count += 1

    cap.release()
    print(f"extracted {count} frames to {output_folder}")


def stitch_frames(frame_folder, output_path, step=5):
    
    # stitch overlapping frames to create a panoramic image.step: Interval to sample frames 
    
    images = []
    files = sorted(os.listdir(frame_folder))
    for i, file_name in enumerate(files):
        if file_name.endswith(".jpg") and i % step == 0:
            img_path = os.path.join(frame_folder, file_name)
            img = cv2.imread(img_path)
            if img is not None:
                images.append(img)

    print(f"using {len(images)} frames for stitching.")
    stitcher = cv2.Stitcher_create() if cv2.__version__.startswith("4") else cv2.createStitcher()
    status, pano = stitcher.stitch(images)

    if status == cv2.Stitcher_OK:
        cv2.imwrite(output_path, pano)
        print(f"panoramic image saved to {output_path}")
    else:
        print(f"error in stitching, status code: {status}")



def main():
    video_path = "C:/Users/ASUS/Desktop/Capture The Scene/video.mp4" 
    frame_folder = "C:/Users/ASUS/Desktop/Capture The Scene/frame" 
    output_path = "C:/Users/ASUS/Desktop/Capture The Scene/output/panorama_output.png"

    print("Step 1: extracting frames from the video")
    extract_frames(video_path, frame_folder)

    print("Step 2: stitching frames into a panoramic image")
    stitch_frames(frame_folder, output_path)

if __name__ == "__main__":
    main()