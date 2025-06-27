from flask import Flask, render_template, Response, request
import yolo_detect
import rcnn_detect

app = Flask(__name__)
model_type = "yolo"  # default

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/select_model', methods=['POST'])
def select_model():
    global model_type
    data = request.get_json()
    model_type = data.get('model', 'yolo')
    print(f"Model switched to: {model_type}")
    return '', 204

def generate_frames():
    while True:
        if model_type == "yolo":
            frame = yolo_detect.detect_frame()
        else:
            frame = rcnn_detect.detect_frame()
        
        if frame is None:
            continue
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video')
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
