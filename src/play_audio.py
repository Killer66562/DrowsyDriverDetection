from pydub import AudioSegment
from pydub.playback import play

song = AudioSegment.from_mp3("audio/never_gonna_give_you_up.mp3")
play(song)