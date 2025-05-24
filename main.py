import cv2
import torch
import torchvision.transforms as transforms
from PIL import Image
from treinamento_modelo_fogo import FireDetectionCNN

lower_orange = (5, 150, 150)
upper_orange = (15, 255, 255)
lower_red_light = (0, 150, 150)
upper_red_light = (10, 255, 255)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = FireDetectionCNN().to(device)
model.load_state_dict(torch.load('modelo_fogo.pth', map_location=device))
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

def detect_fire(frame):
    frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    frame_tensor = transform(frame_pil).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(frame_tensor)
        prediction = torch.sigmoid(output)

    print(f'Confiança do modelo: {prediction.item():.4f}')  # DEBUG
    threshold = 0.7
    return prediction.item() > threshold

def check_fire_color(frame):
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask_orange = cv2.inRange(hsv_frame, lower_orange, upper_orange)
    mask_red_light = cv2.inRange(hsv_frame, lower_red_light, upper_red_light)
    combined_mask = cv2.bitwise_or(mask_orange, mask_red_light)
    area = cv2.countNonZero(combined_mask)
    return area > 300, combined_mask

def detect_fire_in_video(frame):
    has_color, mask = check_fire_color(frame)
    if has_color and detect_fire(frame):
        return True, mask
    return False, None

def play_video_sequentially(video_files):
    for video_file in video_files:
        cap = cv2.VideoCapture(video_file)
        if not cap.isOpened():
            print(f"Erro ao abrir o vídeo {video_file}!")
            continue

        print(f"Reproduzindo vídeo: {video_file}")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            fire_detected, mask = detect_fire_in_video(frame)
            if fire_detected and mask is not None:
                frame_green = frame.copy()
                frame_green[mask > 0] = (0, 255, 0)
                frame = cv2.addWeighted(frame, 0.7, frame_green, 0.3, 0)

                cv2.putText(frame, "Fogo Detectado", (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0, 0, 255), 2, cv2.LINE_AA)

            cv2.imshow('Q para Sair', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
    cv2.destroyAllWindows()

video_files = ['Queimadas.mp4', 'NATUREZA.mp4']
play_video_sequentially(video_files)
