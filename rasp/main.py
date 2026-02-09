from flask import Flask, Response, request, jsonify
from video_stream import gen_frames
import time

# ----- Estado de Visión -----
vision_state = {"colors": [], "centroids": [], "areas": [], "time": 0.0}


# ----- Flask -----
app = Flask(__name__)


@app.route("/video_feed")
def video_feed():
    return Response(gen_frames(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


# @app.route("/command", methods=["POST"])
# def command():
#     global control_state

#     data = request.get_json(force=True) or {}

#     return jsonify({"status": "ok", "control_state": control_state})


@app.route("/vision_data", methods=["POST"])
def vision_data():
    global vision_state

    data = request.get_json(force=True) or {}
    vision_state["colors"] = data.get("colors", [])
    vision_state["centroids"] = data.get("centroids", [])
    vision_state["areas"] = data.get("areas", [])
    vision_state["time"] = data.get("time", time.time())

    return jsonify({"status": "ok"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, threaded=True)