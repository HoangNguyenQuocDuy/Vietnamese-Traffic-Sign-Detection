import torch
import pathlib
pathlib.PosixPath = pathlib.WindowsPath

model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt', force_reload=True)
img = 't2.png'  # Đổi thành ảnh test của bạn

results = model(img)
results.show()  # Hiển thị ảnh với bounding box
results.save()  # Lưu ảnh kết quả