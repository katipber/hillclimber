import base64
from pathlib import Path

from IPython import display as ipythondisplay
from gym.wrappers import Monitor
from pyvirtualdisplay import Display

display = Display(visible=0, size=(1400, 900))
display.start()


def get_recorder(env, path="videos"):
    return Monitor(env, path, force=True, video_callable=lambda episode: True)


def show_videos(path="videos"):
    html = []
    for mp4 in Path(path).glob("*.mp4"):
        video_b64 = base64.b64encode(mp4.read_bytes())
        html.append('''<video alt="{}" autoplay
                      loop controls style="height: 400px;">
                      <source src="data:video/mp4;base64,{}" type="video/mp4" />
                 </video>'''.format(mp4, video_b64.decode('ascii')))
    ipythondisplay.display(ipythondisplay.HTML(data="<br>".join(html)))


def start_recording(env):
    env.unwrapped.automatic_rendering_callback = env.video_recorder.capture_frame
