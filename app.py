from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
from flask_socketio import SocketIO,send,emit

import os
import shutil
from framing import video_to_frame
from behaviors_detector import BehaviorDetection
from facial_detector import FacialDetection
from movement_detector import MovementDetection
import json

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'
socketio = SocketIO(app)

process = {

}

@app.route('/session/get')
@cross_origin()
def get_session():
    return jsonify(os.listdir("result/"))


@app.route('/session/create', methods=['POST'])
@cross_origin() 
def create_session():
    session_id = request.json['session-id']
    if os.path.exists("result/" + session_id):
        return "-1"
    os.mkdir("result/" + session_id)
    result_folder = ["frames", "behaviors", "facial", "movement", "logs"]
    for fol in result_folder:
        os.mkdir("result/" + session_id + "/" + fol)
    return "0"


@app.route('/session/delete', methods=['POST'])
@cross_origin()
def delete_session():
    session_id = request.json['session-id']
    if os.path.exists("result/" + session_id):
        shutil.rmtree("result/"+session_id)
        return "0"
    return "-1"

def get_status_of(sess_id, on_event=False):
    session_id = sess_id
    if os.path.exists('result/' + session_id):
        total_frame = len(os.listdir('result/' + session_id + "/frames"))
        behavior_result = len(os.listdir(
            'result/' + session_id + '/behaviors'))
        facial_result = len(os.listdir('result/' + session_id + '/facial'))
        movement_done = os.path.exists(
            'result/' + session_id + '/movement/result.json')
        data = {
            "total_frame": total_frame,
            "behavior_result": behavior_result,
            "facial_result": facial_result,
            "movement_done": movement_done
        }
        if on_event:
            send(data,json=True)
        return jsonify(data)
    return "-1"

@app.route('/session/status', methods=['POST'])
@cross_origin()
def get_status():
    session_id = request.json['session-id']
    return get_status_of(session_id)


@app.route('/session/<session_id>/upload/', methods=['POST'])
@cross_origin()
def upload_file(session_id):
    print(session_id)
    if os.path.exists('result/' + session_id):
        if request.method == 'POST':
            f = request.files['file']
            f.save('result/' + session_id + '/' + 'video.mp4')
            
            shutil.rmtree('result/' + session_id + '/frames')
            os.mkdir('result/' + session_id + '/frames')
            
            video_to_frame(session_id,'result/'+session_id+'/video.mp4',1)

            return "0"
    return "-1"

@app.route('/session/inference/<infer_type>', methods=['POST'])
@cross_origin()
def inference(infer_type):
    session_id = request.json['session-id']
    if os.path.exists('result/' + session_id):
        if infer_type == 'behaviors':
            shutil.rmtree('result/' + session_id + '/behaviors')
            os.mkdir('result/' + session_id + '/behaviors')
            if session_id not in process:
                process[session_id] = {}
            process[session_id]["behaviors"] = BehaviorDetection()
            process[session_id]["behaviors"].behaviors_detect(session_id=session_id)
            return "0"
        if infer_type == 'facial':
            shutil.rmtree('result/' + session_id + '/facial')
            os.mkdir('result/' + session_id + '/facial')
            if session_id not in process:
                process[session_id] = {}
            process[session_id]["facial"] = FacialDetection()
            process[session_id]["facial"].facial_detect(session_id=session_id)
            return "0"
        if infer_type == 'movement':
            shutil.rmtree('result/' + session_id + '/movement')
            os.mkdir('result/' + session_id + '/movement')
            if session_id not in process:
                process[session_id] = {}
            process[session_id]["movement"] = MovementDetection()
            process[session_id]["movement"].movement_detect(session_id=session_id)
            return "0"
    return "-1"

@app.route('/session/inference/abort/<infer_type>', methods=['POST'])
@cross_origin()
def abort_infer(infer_type):
    session_id = request.json['session-id']
    if os.path.exists('result/' + session_id):
        if infer_type == 'behaviors':
            if "behaviors" in process[session_id]:
                process[session_id]["behaviors"].is_aborted = True
                return "0"
        if infer_type == 'facial':
            if "facial" in process[session_id]:
                process[session_id]["facial"].is_aborted = True
                return "0"
        if infer_type == 'movement':
            if "movement" in process[session_id]:
                process[session_id]["movement"].is_aborted = True
                return "0"
        
    return "-1"

def check_status(infer_type, session_id):
    if infer_type in process[session_id]:
        if process[session_id][infer_type].is_done == True:
            return 2
        if process[session_id][infer_type].is_aborted == False:
            return 1  
    return 0


@app.route('/session/inference/status', methods=['POST'])
@cross_origin()
def inference_status():
    result = {
        "behaviors": 0,
        "facial": 0,
        "movement":0
    }
    session_id = request.json['session-id']
    infer_list = ["behaviors","facial","movement"]
    if session_id in process:
        for infer_type in infer_list:
            result[infer_type] = check_status(infer_type,session_id)
    return result



@app.route('/session/inference/fetch', methods=['POST'])
@cross_origin()
def fetch_infer_status():
    session_id = request.json['session-id']
    socketio.on_event(session_id + '->infer', get_status_of(session_id,True));

@app.route('/session/result', methods=['POST'])
@cross_origin()
def get_result():
    session_id = request.json['session-id']
    if os.path.exists('result/' + session_id):
        movement_file = open('result/' + session_id + '/movement/result.json', 'r')
        data = {"movement": json.loads(movement_file.read())}
        movement_file.close()

        behaviors_path = os.listdir('result/' + session_id + '/behaviors')
        behaviors_data = {}
        for bpath in behaviors_path:
            behaviors_file = open('result/' + session_id + '/behaviors/' + bpath)
            behaviors_data[bpath.split('.')[0].split('_')[1]] = json.loads(behaviors_file.read())
        data["behaviors"] = behaviors_data
        facial_path = os.listdir('result/' + session_id + '/facial')
        facial_data = {}
        for fpath in facial_path:
            facial_file = open('result/' + session_id + '/facial/' + fpath)
            facial_data[fpath.split('.')[0].split('_')[1]] = json.loads(facial_file.read())
        data["facial"] = facial_data
        return jsonify(data)
    return "-1"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80)
    socketio.run(app)
