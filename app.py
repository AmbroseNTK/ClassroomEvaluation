from flask import Flask, request, jsonify
import os
import shutil
from framing import video_to_frame
from behaviors_detector import BehaviorDetection

app = Flask(__name__)


process = {

}

@app.route('/session/get')
def get_session():
    return jsonify(os.listdir("result/"))


@app.route('/session/create', methods=['POST'])
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
def delete_session():
    session_id = request.json['session-id']
    if os.path.exists("result/" + session_id):
        shutil.rmtree("result/"+session_id)
        return "0"
    return "-1"


@app.route('/session/status', methods=['POST'])
def get_status():
    session_id = request.json['session-id']
    if os.path.exists('result/' + session_id):
        total_frame = len(os.listdir('result/' + session_id + "/frames"))
        behavior_result = len(os.listdir(
            'result/' + session_id + '/behaviors'))
        facial_result = len(os.listdir('result/' + session_id + '/facial'))
        movement_done = os.path.exists(
            'result/' + session_id + '/movement/result.json')
        return jsonify({
            "total_frame": total_frame,
            "behavior_result": behavior_result,
            "facial_result": facial_result,
            "movement_done": movement_done
        })
    return "-1"


@app.route('/session/<session_id>/upload/', methods=['POST'])
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
    return "-1"

@app.route('/session/inference/abort/<infer_type>', methods=['POST'])
def abort_infer(infer_type):
    session_id = request.json['session-id']
    if os.path.exists('result/' + session_id):
        if infer_type == 'behaviors':
            if "behaviors" in process[session_id]:
                del process[session_id]["behaviors"]
        
    return "-1"

if __name__ == '__main__':
    app.run(host='0.0.0.0',port=80)
