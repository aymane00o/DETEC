import cv2
import numpy as np

# Chargement du modèle YOLO
net = cv2.dnn.readNet("C:/Users/ayman/Desktop/yolov32-tiny.weights", "C:/Users/ayman/Desktop/yolov3-tiny.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]

# Charger les noms de classes
with open("C:/Users/ayman/Desktop/coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Classe d'intérêt pour détecter les personnes
target_class = "person"

# Charger la vidéo
cap = cv2.VideoCapture(0)  # 0 pour la caméra par défaut
# cap = cv2.VideoCapture("C:/Users/ayman/Desktop/1.mp4")  # Utilise cette ligne si tu veux une vidéo enregistrée
if not cap.isOpened():
    print("Erreur : Impossible d'ouvrir la caméra")
    exit()

# Propriétés vidéo pour sauvegarde
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter("C:/Users/ayman/Desktop/output.mp4", fourcc, fps, (width, height))

# Variable pour compter les personnes détectées
person_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Prétraitement YOLO
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outputs = net.forward(output_layers)

    # Analyse des résultats
    class_ids = []
    confidences = []
    boxes = []
    person_count = 0  # Reset le compteur à chaque image

    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            class_name = classes[class_id]

            # Si l'objet est une personne et que la confiance est suffisante
            if confidence > 0.5 and class_name == target_class:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = center_x - w // 2
                y = center_y - h // 2
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

                # Incrémentation du compteur de personnes
                person_count += 1

    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Dessin des détections
    if len(indices) > 0:
        for i in indices.flatten():
            x, y, w, h = boxes[i]
            class_id = class_ids[i]
            class_name = classes[class_id]

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, class_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Affichage du nombre de personnes détectées
    cv2.putText(frame, f"Personnes detectees : {person_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    # Affichage et écriture dans le fichier de sortie
    out.write(frame)
    cv2.imshow("Person Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libération des ressources
cap.release()
out.release()
cv2.destroyAllWindows()
