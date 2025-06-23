import matplotlib.pyplot as plt
import cv2
import os

def visualize_gradcam(cam, pt_path, frame_folder, save_dir="GradCamFrames"):
    os.makedirs(save_dir, exist_ok=True)
    T, N = cam.shape

    # Retrieve frames
    segment_name = os.path.basename(pt_path).replace("_tracking.pt", "")
    for t in range(T):
        frame_path = os.path.join(frame_folder, segment_name, f"frame_{t:04d}.jpg")
        if not os.path.exists(frame_path):
            continue
        frame = cv2.imread(frame_path)
        frame = cv2.resize(frame, (512, 384))

        heatmap = cam[t]
        heatmap_img = np.zeros((384, 512))
        # Distribute 1000-point CAM on the image space
        for i in range(N):
            x = int(cam[t][i][0] * 512)
            y = int(cam[t][i][1] * 384)
            if 0 <= x < 512 and 0 <= y < 384:
                heatmap_img[y, x] += cam[t][i]

        heatmap_img = cv2.GaussianBlur(heatmap_img, (11, 11), 0)
        heatmap_img = (heatmap_img / np.max(heatmap_img) * 255).astype(np.uint8)
        heatmap_img = cv2.applyColorMap(heatmap_img, cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(frame, 0.6, heatmap_img, 0.4, 0)

        out_path = os.path.join(save_dir, f"{segment_name}_t{t:02d}.jpg")
        cv2.imwrite(out_path, overlay)