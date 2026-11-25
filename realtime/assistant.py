"""Emotion tagging and spoken replies via the Gemini API.
Falls back to keyword emotion if no API key or no reachable model."""
import json
import os
import queue
import re
import threading
import time
import urllib.error
import urllib.request


API = "https://generativelanguage.googleapis.com/v1beta/"
MODEL = "gemini-3.5-flash-lite"            # empty = auto-discover a flash model at startup
FALLBACKS = ("gemini-3.1-flash-lite",)     # tried in order if MODEL fails


SYSTEM = (
    "You are the voice assistant of a live speech-enhancement and speaker-diarization "
    "demo at a college project viva. You receive one transcribed utterance at a time, "
    "with the speaker label. The transcript comes from a small streaming recognizer, so "
    "expect wrong words; infer intent and never comment on transcription errors.\n"
    "Reply with ONLY a JSON object, no other text:\n"
    '{"emotion": "<happy|neutral|sad|angry|surprised|confused>", "reply": "<one or two '
    'short spoken sentences, under 30 words>"}\n'
    'Set "reply" to an empty string if nothing useful can be said.'
)
POS = ("good", "great", "thanks", "nice", "happy", "awesome", "love")
NEG = ("bad", "angry", "annoyed", "hate", "terrible", "wrong", "stupid")




def guess_emotion(text):
    t = (text or "").lower()
    if any(w in t for w in POS):
        return "happy"
    if any(w in t for w in NEG):
        return "angry"
    return "neutral"




def list_models(key, timeout=15):
    """Model names that support generateContent, newest-looking first."""
    req = urllib.request.Request(API + "models", headers={"x-goog-api-key": key})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read())
    out = []
    for m in data.get("models", []):
        if "generateContent" in m.get("supportedGenerationMethods", []):
            out.append(m["name"].replace("models/", ""))
    return out




def pick_model(names):
    """Prefer a plain flash model, newest version number first."""
    def ver(n):
        m = re.search(r"(\d+)\.(\d+)", n)
        return (int(m.group(1)), int(m.group(2))) if m else (0, 0)
    flash = [n for n in names if "flash" in n and "preview" not in n
             and "exp" not in n and "thinking" not in n and "8b" not in n]
    if not flash:
        flash = [n for n in names if "flash" in n]
    if not flash:
        flash = names
    flash.sort(key=ver, reverse=True)
    return flash[0] if flash else None




class Assistant(threading.Thread):


    def __init__(self, convo, wake="system", reply_all=False, speak=True, model=MODEL):
        super().__init__(daemon=True)
        self.convo = convo
        self.wake = (wake or "").lower()
        self.reply_all = reply_all
        self.speak_on = speak
        self.model = model
        self.q = queue.Queue()
        self.history = []
        self.tts = None
        self.key = os.environ.get("GEMINI_API_KEY")
        if not self.key:
            print("[assistant] no GEMINI_API_KEY: emotion by keywords, no replies")
            return
        if not self.model:
            try:
                names = list_models(self.key)
                self.model = pick_model(names)
                print("[assistant] using model: " + str(self.model)
                      + "  (" + str(len(names)) + " available)")
            except Exception as ex:
                print("[assistant] could not list models: " + str(ex))
                self.key = None


    def submit(self, entry):
        self.q.put(entry)


    def _wants_reply(self, text):
        if self.reply_all:
            return True
        return bool(self.wake) and self.wake in text.lower()


    def run(self):
        try:
            import pyttsx3
            self.tts = pyttsx3.init()
            self.tts.setProperty("rate", 175)
        except Exception as ex:
            print("[assistant] TTS unavailable: " + str(ex))
        while True:
            e = self.q.get()
            try:
                self._process(e)
            except Exception as ex:
                e["status"] = "error"
                print("[assistant] " + str(ex))


    def _process(self, e):
        want = self._wants_reply(e["text"]) or e.get("label") == "typed"
        e["status"] = "thinking" if want else "resolved"
        emotion, reply = self._ask(e, want)
        e["emotion"] = emotion
        e["reply"] = reply if want else None
        e["status"] = "done"
        line = "      emotion: " + str(emotion)
        if e["reply"]:
            line += " | reply: " + e["reply"]
        print(line)
        if e["reply"]:
            self._say(e["reply"])


    def _call(self, msgs, model, timeout=8):
        """One generateContent request to `model`. Returns the raw text of the reply."""
        contents = [{"role": "model" if m["role"] == "assistant" else "user",
                     "parts": [{"text": m["content"]}]} for m in msgs]
        body = json.dumps({
            "systemInstruction": {"parts": [{"text": SYSTEM}]},
            "contents": contents,
            "generationConfig": {"maxOutputTokens": 800,
                                 "responseMimeType": "application/json"},
        }).encode()
        req = urllib.request.Request(
            API + "models/" + model + ":generateContent",
            data=body, method="POST",
            headers={"Content-Type": "application/json",
                     "x-goog-api-key": self.key})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            r = json.loads(resp.read())
        parts = r["candidates"][0]["content"]["parts"]
        return "".join(p.get("text", "") for p in parts).strip()


    def _ask(self, e, want):
        emotion = guess_emotion(e["text"])
        if not self.key or not self.model:
            return emotion, None


        user = "Speaker " + str(e.get("speaker")) + ' said: "' + e["text"] + '"'
        if not want:
            user += "\n(Not addressed to you. Set reply to an empty string.)"
        msgs = self.history[-6:] + [{"role": "user", "content": user}]


        t0 = time.time()
        text = None
        for name in [self.model] + [m for m in FALLBACKS if m != self.model]:
            try:
                text = self._call(msgs, name, timeout=8)
                break
            except urllib.error.HTTPError as ex:
                detail = ""
                try:
                    detail = ex.read().decode()[:120]
                except Exception:
                    pass
                print("[assistant] " + name + " HTTP " + str(ex.code) + " " + detail)
            except Exception as ex:
                print("[assistant] " + name + " failed: " + str(ex))
        if text is None:
            return emotion, None


        print("      (api " + format(time.time() - t0, ".1f") + "s)")
        reply = None
        try:
            m = re.search(r"\{.*\}", text, re.S)
            d = json.loads(m.group(0) if m else text)
            emotion = (d.get("emotion") or emotion).lower()
            reply = (d.get("reply") or "").strip() or None
        except Exception:
            reply = text or None
        self.history = msgs + [{"role": "assistant", "content": text}]
        return emotion, reply


    def _say(self, text):
        if self.tts is None or not self.speak_on:
            return
        self.convo.speaking.set()          # gate the mic so we do not hear ourselves
        try:
            self.tts.say(text)
            self.tts.runAndWait()
        except Exception as ex:
            print("[assistant] TTS failed: " + str(ex))
        finally:
            time.sleep(0.3)
            self.convo.speaking.clear()