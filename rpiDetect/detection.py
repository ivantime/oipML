import numpy as np
import pandas as pd
import mpl_toolkits
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import ensemble
import csv
import re
import cv2
from tflite_runtime.interpreter import Interpreter
import time

CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
dirtyCount = 0
cleanCount = 0
totalCount = 0

def append_csv(drytime):
    # Append to csv code
    data = [drytime, totalCount]

    with open('dry.csv', 'a') as f:
        writer = csv.writer(f)
        writer.writerow(data)


def train_model():
    # uses linear regression
    reg = LinearRegression()
    data = pd.read_csv("dry.csv")
    dryLabels = data['DryTime']

    train1 = data.drop(['DryTime'], axis=1)

    ##For Drying time
    # Train model with drylabel and no.of syringe
    x_train, x_test, y_train, y_test = train_test_split(train1, dryLabels, test_size=0.10, random_state=2)
    reg.fit(x_train, y_train)
    reg.score(x_test, y_test)
    # Boost LR
    clf = ensemble.GradientBoostingRegressor(n_estimators=400, max_depth=5, min_samples_split=2,
                                             learning_rate=0.1, loss='ls')
    clf.fit(x_train, y_train)
    clf.score(x_test, y_test)
    ##Save model
    filename = 'dryModel.sav'
    pickle.dump(clf, open(filename, 'wb'))


def predict():
    # load model from storage
    loaded_model = pickle.load(open('dryModel.sav', 'rb'))

    pred = loaded_model.predict(pd.DataFrame({"no_syringe": totalCount}, index=[1]))
    ##Recommended Timing
    return(int(pred))


def load_labels(path='labels.txt'):
    """Loads the labels file. Supports files with or without index numbers."""
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        labels = {}
        for row_number, content in enumerate(lines):
            pair = re.split(r'[:\s]+', content.strip(), maxsplit=1)
            if len(pair) == 2 and pair[0].strip().isdigit():
                labels[int(pair[0])] = pair[1].strip()
            else:
                labels[row_number] = pair[0].strip()
    return labels


def set_input_tensor(interpreter, image):
    """Sets the input tensor."""
    tensor_index = interpreter.get_input_details()[0]['index']
    input_tensor = interpreter.tensor(tensor_index)()[0]
    input_tensor[:, :] = np.expand_dims((image - 255) / 255, axis=0)


def get_output_tensor(interpreter, index):
    """Returns the output tensor at the given index."""
    output_details = interpreter.get_output_details()[index]
    tensor = np.squeeze(interpreter.get_tensor(output_details['index']))
    return tensor


def detect_objects(interpreter, image, threshold):
    """Returns a list of detection results, each a dictionary of object info."""
    set_input_tensor(interpreter, image)
    interpreter.invoke()
    # Counting
    global dirtyCount, cleanCount, totalCount

    # Get all output details
    scores = get_output_tensor(interpreter, 0)
    boxes = get_output_tensor(interpreter, 1)
    # classes = get_output_tensor(interpreter, 2)
    classes = get_output_tensor(interpreter, 3)
    count = int(get_output_tensor(interpreter, 3).size)
    results = []
    for i in range(count):
        if scores[i] >= threshold:
            result = {
                'bounding_box': boxes[i],
                'class_id': classes[i],
                'score': scores[i]
            }
            if (classes[i] == 0.0):
                cleanCount += 1
            else:
                dirtyCount += 1

            results.append(result)
            totalCount += 1
    return results


def videoFeed():
    labels = load_labels()
    interpreter = Interpreter('detect.tflite')
    interpreter.allocate_tensors()
    _, input_height, input_width, _ = interpreter.get_input_details()[0]['shape']
    #10 sec vid
    capture_duration = 10
    cap = cv2.VideoCapture(0)
    start_time = time.time()
    while(int(time.time() - start_time) < capture_duration):
        ret, frame = cap.read()
        img = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), (320, 320))
        res = detect_objects(interpreter, img, 0.2)
        for result in res:
            ymin, xmin, ymax, xmax = result['bounding_box']
            xmin = int(max(1, xmin * CAMERA_WIDTH))
            xmax = int(min(CAMERA_WIDTH, xmax * CAMERA_WIDTH))
            ymin = int(max(1, ymin * CAMERA_HEIGHT))
            ymax = int(min(CAMERA_HEIGHT, ymax * CAMERA_HEIGHT))

            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 3)
            cv2.putText(frame, labels[int(result['class_id'])], (xmin, min(ymax, CAMERA_HEIGHT - 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
        # print(dirtyCount, " dirty")
        # print(cleanCount, "clean")
        # print(totalCount, "total")
    cap.release()
    cv2.destroyAllWindows()

def run_timeml():
    #run vid to get no of syringe from vid
    print("Scanning no of syringe...")
    videoFeed()
    print("vidfeed:",totalCount)
    #add data into csv, buffer 3 sec
    print("Adding data...")
    append_csv(3600)
    time.sleep(3)
    #train model, buffer 10 sec
    print("Training Model...")
    train_model()
    time.sleep(10)
    #run predict
    print("Predicting...")
    recTime = predict()
    print("Predicted Value:",recTime)

def run_camml():
    #run vid to get clean or dirty
    videoFeed()
    print("cam clean:",cleanCount)
    print("cam dirty:",dirtyCount)

if __name__ == "__main__":
       run_timeml()