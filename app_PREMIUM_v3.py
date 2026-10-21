import asyncio
import base64
import io
import math
import json
import os
import re
import shutil
import subprocess
import queue as queue_module
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
import secrets
import hashlib
from pathlib import Path

import edge_tts
from flask import Flask, jsonify, render_template_string, request, send_from_directory, make_response, Response
from PIL import Image, ImageDraw, ImageFont, ImageOps, ImageEnhance

ROOT = Path(__file__).resolve().parent

def _ensure_dir(path):
    # Some ZIP tools turn empty folders into 0-byte FILES. The old code then
    # crashed at startup with FileExistsError before the page even opened.
    if path.exists() and not path.is_dir():
        path.unlink()
    path.mkdir(parents=True, exist_ok=True)

OUT = ROOT / 'generated'
_ensure_dir(OUT)
UPLOADS = ROOT / "uploads"
_ensure_dir(UPLOADS)

# --- Gemini image model (fixed Oct 2026) -----------------------------------
# The old default 'gemini-2.5-flash-image' is now a LEGACY model on Google's
# docs, and Google's pricing page lists EVERY Gemini image model as
# "Free Tier: Not available" -- image generation needs a billing-enabled /
# prepaid-credit project. Cheapest current replacement:
DEFAULT_IMAGE_MODEL = 'gemini-3.1-flash-lite-image'
LEGACY_IMAGE_MODELS = {
    'gemini-2.5-flash-image': DEFAULT_IMAGE_MODEL,
    'gemini-2.5-flash-image-preview': DEFAULT_IMAGE_MODEL,
    'gemini-2.0-flash-preview-image-generation': DEFAULT_IMAGE_MODEL,
    'gemini-2.0-flash-exp-image-generation': DEFAULT_IMAGE_MODEL,
}

def _normalize_image_model(model):
    m = str(model or '').strip()
    return LEGACY_IMAGE_MODELS.get(m, m or DEFAULT_IMAGE_MODEL)

def _friendly_gemini_error(exc):
    raw = str(exc)
    low = raw.lower()
    if 'resource_exhausted' in low or 'quota' in low or '429' in raw:
        return ('Gemini IMAGE QUOTA / BILLING error: Under current Google rules, all Gemini image '
                'models are NOT available on the FREE tier (Free Tier = Not available). In the Google project that owns '
                'this API key, enable billing / prepaid credits; otherwise, images cannot be generated. '
                'Original Google error: ' + raw[:350])
    if 'api_key_invalid' in low or 'api key not valid' in low:
        return 'Gemini API KEY is incorrect. Create a correct key in Google AI Studio and save it again. Original error: ' + raw[:350]
    if 'not_found' in low or '404' in raw:
        return (f'Gemini MODEL not found (it may be an old or discontinued model). Set the model to {DEFAULT_IMAGE_MODEL} and '
                'save/test it again. Original error: ' + raw[:350])
    if '402' in raw or 'billing' in low or 'prepay' in low or 'credits' in low:
        return 'There is a Gemini BILLING / CREDITS problem (402). Add credits in Google Cloud Billing. Original error: ' + raw[:350]
    if 'permission' in low or '403' in raw:
        return ('Gemini PERMISSION error (403): this key/project does not have permission to use the image model - often because billing '
                'is not enabled or because of a region restriction. Original error: ' + raw[:350])
    return raw
app = Flask(__name__)
LAN_MODE = os.environ.get('STUDIO_LAN') == '1'
LAN_PASSWORD = os.environ.get('STUDIO_ACCESS_PASSWORD', '')

@app.before_request
def lan_access_guard():
    if not LAN_MODE:
        return None
    credentials = request.authorization
    if not credentials or not secrets.compare_digest(credentials.password or '', LAN_PASSWORD) or not secrets.compare_digest(credentials.username or '', 'studio'):
        return Response('Authentication required', 401, {'WWW-Authenticate': 'Basic realm="AI Video Studio Pro"'})
    return None
JOBS = {}
LOCK = threading.Lock()
WORK_Q = queue_module.Queue()
PROJECTS_FILE = ROOT / 'projects.json'
PROJECTS_LOCK = threading.Lock()
FLOW_PROFILES_FILE = ROOT / 'flow_profiles.json'
VOICES_CACHE_FILE = ROOT / 'voices_cache.json'
MAX_SCENES = 300
MAX_SCRIPT_CHARS = 400000


def _resolve_voice(value):
    if value in VOICES:
        return VOICES[value]
    if value and re.fullmatch(r'[a-z]{2,3}-[A-Z]{2}-[A-Za-z0-9]+Neural', str(value)):
        return str(value)
    return VOICES['English Female']
def _same_origin():
    origin = request.headers.get('Origin')
    if not origin: return True
    return origin.rstrip('/') == request.host_url.rstrip('/')

VOICES = {
    'English Female': 'en-US-JennyNeural',
    'English Male': 'en-US-GuyNeural',
    'English Female Aria': 'en-US-AriaNeural',
    'English Male Christopher': 'en-US-ChristopherNeural',
    'English UK Female': 'en-GB-SoniaNeural',
    'English UK Male': 'en-GB-RyanNeural',
    'English India Female': 'en-IN-NeerjaNeural',
    'English India Male': 'en-IN-PrabhatNeural',
    'Hindi Female': 'hi-IN-SwaraNeural',
    'Hindi Male': 'hi-IN-MadhurNeural',
    'Urdu Female': 'ur-PK-UzmaNeural',
    'Urdu Male': 'ur-PK-AsadNeural',
    'Arabic Female': 'ar-SA-ZariyahNeural',
    'Arabic Male': 'ar-SA-HamedNeural',
    'Arabic Egypt Female': 'ar-EG-SalmaNeural',
    'Bengali Male': 'bn-IN-BashkarNeural',
    'Malay Female': 'ms-MY-YasminNeural',
    'Malay Male': 'ms-MY-OsmanNeural',
    'Tamil Female': 'ta-IN-PallaviNeural',
    'Telugu Female': 'te-IN-ShrutiNeural',
    'Spanish Female': 'es-ES-ElviraNeural',
    'French Female': 'fr-FR-DeniseNeural',
    'German Female': 'de-DE-KatjaNeural',
    'Turkish Male': 'tr-TR-AhmetNeural',
    'Indonesian Female': 'id-ID-GadisNeural',
}

HTML = r'''<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>AI Video Studio Pro — Advanced</title>
<style>
:root{color-scheme:dark}*{box-sizing:border-box}body{margin:0;background:#070b16;color:#edf3ff;font:15px Arial,sans-serif}
a{color:#92b8ff}button{border:0;border-radius:10px;padding:12px 16px;background:#4d83ff;color:#fff;font-weight:bold;cursor:pointer;font-size:14px;min-height:44px}button:hover{filter:brightness(1.12)}button:disabled{opacity:.55;cursor:wait}button.ghost{background:#243b61}button.danger{background:#93414c}button.gold{background:#8a6d1f}
input,select,textarea{width:100%;background:#091426;border:1px solid #3b4e6a;color:#f1f6ff;padding:11px;border-radius:9px;font:inherit;min-height:44px}textarea{min-height:150px;resize:vertical}
label{display:block;color:#b4c6e4;margin:12px 0 6px}.small{font-size:12px;color:#9bb0d1;line-height:1.55}
.layout{display:flex;min-height:100vh}.side{width:205px;flex:0 0 205px;background:#0b1322;border-right:1px solid #22314e;padding:18px 12px;position:sticky;top:0;height:100vh}.side h1{font-size:19px;margin:2px 6px 4px}.side .ver{color:#7f95b8;font-size:11px;margin:0 6px 16px}.navbtn{display:block;width:100%;text-align:left;background:transparent;color:#c9d9f2;padding:12px 12px;margin:3px 0;border-radius:9px;font-weight:normal}.navbtn:hover{background:#16263f}.navbtn.active{background:#1d3a63;color:#fff;font-weight:bold}.badge{background:#c0392b;color:#fff;border-radius:20px;padding:1px 8px;font-size:11px;margin-left:6px}
.main{flex:1;min-width:0;padding:20px 22px;max-width:1380px}.view{display:none}.view.active{display:block}
.grid{display:grid;grid-template-columns:1.25fr 1fr;gap:18px}.card{background:#111d33;border:1px solid #2b3c59;border-radius:15px;padding:19px;margin-bottom:16px;box-shadow:0 8px 28px #0002}h2{font-size:17px;margin:0 0 12px;color:#d7e9ff}h3{margin:4px 0 8px}
.choices{display:grid;grid-template-columns:repeat(3,1fr);gap:8px}.choices label{background:#1a2944;border:1px solid #344869;border-radius:9px;padding:11px 6px;text-align:center;cursor:pointer;margin:0;color:#fff}.choices input{width:auto;min-height:0;margin-right:4px}
.two{display:grid;grid-template-columns:1fr 1fr;gap:12px}.three{display:grid;grid-template-columns:1fr 1fr 1fr;gap:10px}.rowflex{display:flex;gap:9px;flex-wrap:wrap;align-items:center}
.check{display:flex;gap:8px;align-items:center}.check input{width:auto;min-height:0}
.primary{width:100%;background:linear-gradient(90deg,#557cf8,#8757e7);font-size:18px;padding:18px}
.status{background:#0a1528;border:1px solid #34496a;padding:14px;border-radius:10px;min-height:70px;white-space:pre-wrap;overflow-wrap:anywhere;line-height:1.6}progress{width:100%;height:18px;accent-color:#6d9bff;margin:12px 0}
.notice{padding:13px;border:1px solid #66532c;background:#2a2415;color:#f9dda0;border-radius:9px;margin-bottom:16px}
video{width:100%;margin-top:12px;border-radius:10px}.item{background:#0c172b;border:1px solid #344869;border-radius:10px;padding:12px;margin:9px 0}.item b{color:#fff}.pill{display:inline-block;padding:2px 10px;border-radius:20px;font-size:11px;background:#265a8a}.pill.done{background:#1e7a46}.pill.error{background:#93414c}.pill.queued{background:#8a6d1f}.pill.running{background:#265a8a}
.assets{display:grid;grid-template-columns:repeat(auto-fill,minmax(130px,1fr));gap:10px}.assets img{width:100%;border-radius:9px;border:1px solid #344869}
.presets{display:grid;grid-template-columns:repeat(auto-fill,minmax(118px,1fr));gap:9px}.preset{background:#1a2944;border:1px solid #36578e;border-radius:11px;padding:12px 8px;text-align:center;cursor:pointer}.preset .em{font-size:26px;display:block;margin-bottom:5px}.preset span{font-size:12px;color:#dbe8ff}
.vtabs{display:flex;gap:7px;flex-wrap:wrap;margin:8px 0}.vtab{background:#1a2944;padding:8px 13px;min-height:38px;font-weight:normal}.vtab.active{background:#4d83ff}
.logbox{background:#050a14;border:1px solid #2b3c59;border-radius:10px;padding:13px;font:12px monospace;white-space:pre-wrap;max-height:420px;overflow:auto;line-height:1.7}
@media(max-width:900px){.layout{flex-direction:column}.side{width:100%;flex:none;height:auto;position:static;display:flex;flex-wrap:wrap;gap:4px;align-items:center}.side h1{width:100%}.navbtn{width:auto}.grid,.two,.three{grid-template-columns:1fr}.main{padding:14px 10px}}
</style></head>
<body><div class="layout">
<aside class="side">
<h1>🎬 AI Video Studio Pro</h1><p class="ver">ADVANCED v3 · FREE images edition</p>
<button class="navbtn active" data-nav="studio" onclick="switchView('studio')">🎥 Studio</button>
<button class="navbtn" data-nav="queue" onclick="switchView('queue')">📋 Queue <span class="badge" id="queueCount" style="display:none">0</span></button>
<button class="navbtn" data-nav="assets" onclick="switchView('assets')">🖼 Asset Wall</button>
<button class="navbtn" data-nav="projects" onclick="switchView('projects')">📁 Projects / Drafts <span class="badge" id="draftCount" style="display:none">0</span></button>
<button class="navbtn" data-nav="deliverables" onclick="switchView('deliverables')">✅ Deliverables</button>
<button class="navbtn" data-nav="flow" onclick="switchView('flow')">🌊 Google Flow / Veo</button>
<button class="navbtn" data-nav="connect" onclick="switchView('connect')">🔑 Connect Gemini</button>
<button class="navbtn" data-nav="about" onclick="switchView('about')">ℹ️ About / Help</button>
<button class="navbtn" data-nav="console" onclick="switchView('console')">🖥 Console</button>
<p class="small" style="margin:16px 6px">FREE images: Pollinations (no key needed), Cloudflare FLUX (free account), or free stock photos — Openverse (no key), Pixabay and Pexels (free key). Optional: connect your own Gemini account on the Connect Gemini page (saved once, no password).</p>
</aside>
<div class="main">

<section class="view active" id="view-studio">
<div class="notice"><b>Advanced Studio:</b> Save projects, queue multiple videos, get scripts written by the FREE AI Writer, 25 voices + your own voiceover + background music. <b>Note:</b> Veo video clips and ElevenLabs cloned voices are not included in this free version — they are paid services. Here, videos are created from pictures + camera motion.</div>
<div class="grid"><div>
<div class="card"><h2>📁 Project</h2>
<div class="two"><div><label>Project name</label><input id="projectName" placeholder="e.g. Want a phone story"></div>
<div><label>Load a saved project</label><select id="scriptSelect" onchange="applySelectedProject()"><option value="">— Choose a project —</option></select></div></div>
<div class="rowflex" style="margin-top:12px"><button type="button" onclick="saveProject()">💾 Save Project / Draft</button><button type="button" class="danger" onclick="deleteSelectedProject()">Delete Project</button></div>
<p class="small" id="projectMsg">Projects are saved on this PC — when you reopen the studio, your script and settings are restored.</p></div>

<div class="card"><h2>1 · Script</h2>
<label>Enter a topic — the FREE AI Writer will write the script</label>
<div class="two"><input id="writerTopic" placeholder="e.g. A boy who wants a phone but has no money"><select id="writerLang"><option>English</option><option>Urdu</option><option>Hindi</option><option>Roman Urdu</option></select></div>
<div class="rowflex" style="margin-top:10px"><button type="button" class="gold" onclick="writeScript()">✍ FREE AI Writer — Create Script</button></div>
<label>Or paste your own script</label><textarea id="script" style="min-height:210px" placeholder="Paste your ready script here. One paragraph per scene works best."></textarea>
<p class="small">The video length is determined automatically by the narration. Each paragraph = one scene = one image.</p></div>

<div class="card"><h2>2 · Scene Timeline & Character</h2>
<label>Character Bible / Consistent Character</label><textarea id="characterBible" style="min-height:88px" placeholder="Example: Ahmed, 10 years old, short black hair, green shalwar kameez, kind smile. Keep same look in every scene."></textarea>
<label>Ready-made character presets (click to use)</label><div class="presets" id="presets"></div>
<div class="rowflex" style="margin:12px 0"><button type="button" onclick="previewScenes()">🎞 Preview / Refresh Scenes</button><button type="button" class="ghost" onclick="resetSceneEdits()">Reset Scene Edits</button><button type="button" class="gold" onclick="downloadFlowPrompts()">📋 Flow Prompt Sheet (Download)</button></div>
<p class="small">Flow Prompt Sheet: a ready-made prompt file for every scene — paste the prompts into Google Flow one by one, create the clips, download them, and add them above under "Veo / Flow scene clips".</p>
<p id="sceneInfo" class="small">The scene timeline will appear here. You can edit the narration and image prompt of each scene separately.</p><div id="sceneTimeline"></div></div>

<div class="card"><h2>3 · Visual Style & Format</h2><div class="choices" id="styles">
<label><input type="radio" name="style" value="2D Cartoon" checked>2D Cartoon</label><label><input type="radio" name="style" value="3D Animated Pixar style">3D Pixar</label><label><input type="radio" name="style" value="3D Cartoon">3D Cartoon</label><label><input type="radio" name="style" value="Realistic">Realistic</label><label><input type="radio" name="style" value="Anime">Anime</label><label><input type="radio" name="style" value="Stickman">Stickman</label><label><input type="radio" name="style" value="Cinematic">Cinematic</label><label><input type="radio" name="style" value="Comic Book">Comic Book</label><label><input type="radio" name="style" value="Documentary">Documentary</label>
</div>
<div class="two"><div><label>Aspect Ratio</label><select id="ratio"><option>9:16</option><option>16:9</option><option>1:1</option></select></div>
<div><label>Image Source</label><select id="imageMode"><option value="pollinations" selected>🆓 FREE — Pollinations (no key needed)</option><option value="cloudflare">🆓 FREE — Cloudflare FLUX (account + token)</option><option value="gemini">Gemini (connected account — billing required)</option><option value="openverse">🆓 FREE — Stock Photos: Openverse (no key)</option><option value="pixabay">🆓 FREE — Stock Photos: Pixabay (free key)</option><option value="pexels">🆓 FREE — Stock Photos: Pexels (free key)</option><option value="upload">My uploaded pictures</option></select></div></div>
<p class="small">Pollinations FREE and Openverse stock photos: nothing is required. Cloudflare: enter the Account ID + Token in the fields below (the token is NOT saved). Gemini: select a connected account on the Connect Gemini page. Pixabay / Pexels: enter your free API key below (keys are used only for this job and are NOT saved). Stock photos are real photographs matched to each scene's prompt.</p>
<div class="two"><div><label>Cloudflare Account ID</label><input id="cfAccount" autocomplete="off" placeholder="Only for Cloudflare mode"></div><div><label>Cloudflare API Token</label><input id="cfToken" type="password" autocomplete="off" placeholder="With Workers AI permission"></div></div>
<div class="two"><div><label>Pixabay API Key (free)</label><input id="pixabayKey" type="password" autocomplete="off" placeholder="Only for Pixabay mode — not saved"></div><div><label>Pexels API Key (free)</label><input id="pexelsKey" type="password" autocomplete="off" placeholder="Only for Pexels mode — not saved"></div></div>
<label>Upload images (for Upload mode; scene order = file order)</label><input id="uploads" type="file" accept="image/png,image/jpeg,image/webp" multiple>
<label>🎬 Veo / Flow scene clips (optional MP4 — the first clips will take the place of the first scenes; the remaining scenes will be created from images)</label><input id="sceneClips" type="file" accept="video/mp4,video/quicktime,video/webm,.mp4,.mov,.webm,.mkv" multiple>
<p class="small">Create the clips in Google Flow and download them (see the Flow page below), then add them here — captions and voiceover will be applied to them as well.</p></div>

<div class="card"><h2>4 · Voice & Music</h2>
<div class="vtabs" id="voiceTabs">
<button type="button" class="vtab active" onclick="renderVoices('All',this)">All</button><button type="button" class="vtab" onclick="renderVoices('Fav',this)">⭐ Favourites</button><button type="button" class="vtab" onclick="renderVoices('English',this)">English</button><button type="button" class="vtab" onclick="renderVoices('Hindi',this)">Hindi</button><button type="button" class="vtab" onclick="renderVoices('Urdu',this)">Urdu</button><button type="button" class="vtab" onclick="renderVoices('Malay',this)">Malay</button>
</div>
<label>Narrator Voice — FREE (the full catalog loads — hundreds of voices)</label><input id="voiceSearch" placeholder="🔎 Search voices: urdu, hindi, arabic, female, en-US..." oninput="renderVoices(currentVoiceFilter)"><select id="voice" style="margin-top:8px"></select>
<div class="rowflex" style="margin-top:10px"><button type="button" onclick="previewVoice()">▶ Voice Preview</button><button type="button" class="ghost" onclick="toggleFav()">⭐ Add/Remove Favourite</button></div>
<p class="small" id="voiceMsg">An internet connection is required for the preview (Edge TTS is free).</p>
<div class="two"><div><label>Voice speed</label><select id="speed"><option value="+0%">Normal</option><option value="-15%">Slow</option><option value="+15%">Fast</option></select></div>
<div><label>Upload your own voiceover (optional — if provided, TTS will not be used)</label><input id="ownVoice" type="file" accept="audio/*,.mp3,.wav,.m4a,.ogg"></div></div>
<label>Upload background music (optional; it will be mixed in automatically at a low volume)</label><input id="musicFile" type="file" accept="audio/*,.mp3,.wav,.m4a,.ogg"></div>

<div class="card"><h2>5 · Motion, Transitions & Captions</h2>
<div class="two"><div><label>Camera motion</label><select id="motion"><option value="pan">Pan (left → right)</option><option value="zoomin">Zoom In</option><option value="zoomout">Zoom Out</option><option value="off">Static (fastest)</option></select></div>
<div><label>Transitions</label><select id="transitions"><option value="mixed">Mixed (fade + cuts)</option><option value="fade">Fade</option><option value="cut">Hard cut (fastest)</option></select></div></div>
<label class="check"><input id="captions" type="checkbox" checked> Add captions</label>
<label>Caption style</label><select id="captionStyle"><option value="bold">Bold Shorts</option><option value="highlight">Word Highlight (yellow)</option><option value="comic">Bold Comic</option><option value="green">Simple Green</option><option value="minimal">Minimal</option><option value="box">Boxed</option></select></div>

<button class="primary" id="launch" onclick="launch(false)">🚀 LAUNCH FULL VIDEO</button>
<div class="rowflex" style="margin-top:10px"><button type="button" class="ghost" onclick="launch(true)" style="flex:1">➕ Add to Queue</button><button type="button" class="ghost" onclick="verifySystem()" style="flex:1">✔ Verify System</button></div>
<p class="small" id="verifyResult"></p>
</div>
<div>
<div class="card"><h2>Production Queue</h2><div class="status" id="status">Ready. Paste your script and press Launch.</div><progress id="progress" max="100" value="0"></progress><div id="result"></div></div>
<div class="card"><h2>What happens automatically?</h2><p>✓ Script → scenes</p><p>✓ FREE images (Pollinations / Cloudflare / stock photos), Gemini (connected account) or your uploads</p><p>✓ Voiceover (25 free voices or your own)</p><p>✓ Camera motion + transitions + captions</p><p>✓ Background music mix</p><p>✓ MP4 render + download</p><p class="small">FFmpeg must be installed on this PC. Long videos take time. Videos in the queue are created one after another.</p></div>
</div></div>
</section>

<section class="view" id="view-queue"><div class="card"><h2>📋 Queue — All Jobs</h2><div id="queueList"><p class="small">No jobs yet. Launch a video from the Studio.</p></div></div></section>

<section class="view" id="view-assets"><div class="card"><h2>🖼 Asset Wall — Scene Images</h2><label>Select a job</label><select id="assetsJobSel" onchange="loadAssets()"><option value="">— run a job first —</option></select><div class="assets" id="assetsGrid" style="margin-top:14px"></div></div></section>

<section class="view" id="view-projects"><div class="card"><h2>📁 Projects / Drafts</h2><div id="projectsList"><p class="small">No saved projects.</p></div></div></section>

<section class="view" id="view-deliverables"><div class="card"><h2>✅ Deliverables — Finished Videos</h2><div id="videosList"><p class="small">No finished video yet.</p></div></div></section>

<section class="view" id="view-flow"><div class="card" style="border-color:#5274ac"><h2>🌊 Google Flow / Veo — Profiles</h2>
<p class="small"><b>The honest truth:</b> Google Flow has no free official API. On the Flow website (labs.google), only an account with a <b>Google AI Plus / Pro / Ultra plan</b> will work — Veo clips are created using the credits of that plan (Lite ~10, Fast ~20, Quality ~100 credits per clip). That is why Flow is connected in this software as follows: save a profile → <b>Open & Login</b> opens Flow on this PC → sign in yourself and create clips for your scenes (the Prompt Sheet can be downloaded from the Studio page) → download the clips and add them in the Studio under "Veo / Flow scene clips". The software will apply voiceover and captions to those clips and produce the final video.</p>
<p class="small" style="color:#f9dda0">⚠ Some tools run Flow automatically across many accounts — that can violate Google's terms and carries a risk of account bans, so that method has NOT been used here. Your own account, operated by you — the safe method.</p>
<div class="two"><div><label>Profile name (e.g.: acc1 — Primary)</label><input id="flowName" placeholder="acc1 — Primary"></div><div><label>Note (e.g.: which Gmail / which plan it is on)</label><input id="flowNote" placeholder="Google AI Pro — my main account"></div></div>
<div class="rowflex" style="margin-top:12px"><button type="button" onclick="saveFlowProfile()">💾 Save Profile</button><button type="button" class="gold" onclick="openFlow()">🌐 Open &amp; Login — Sign in to Google Flow</button></div>
<p class="small" id="flowMsg"></p><div id="flowList" style="margin-top:10px"></div></div></section>

<section class="view" id="view-connect"><div class="card" style="border-color:#5274ac"><h2>🔑 Connect Gemini (optional)</h2>
<p class="small">Gemini is optional — the FREE sources (Pollinations, Cloudflare, stock photos) need nothing. To use Gemini, enter your own Google AI Studio API key <b>once</b>: it is saved directly on this server (no password, no unlock step) and then used automatically whenever you pick Gemini as the image source. Gemini image generation requires billing / prepaid credits in your Google project. Never share your key with anyone.</p>
<div class="two"><div><label>Saved Gemini account</label><select id="accountSelect" onchange="chooseAccount()"><option value="">— No account saved yet —</option></select></div><div><label>Account name</label><input id="accountName" placeholder="My Gemini" autocomplete="off"></div></div>
<div class="two"><div><label>Google AI Studio API Key</label><input id="accountKey" type="password" autocomplete="off" placeholder="Paste API key; never share"></div><div><label>Image model</label><input id="imageModel" value="gemini-3.1-flash-lite-image"></div></div>
<div class="rowflex" style="margin-top:12px"><button type="button" onclick="saveAccount()">💾 Save Account</button><button type="button" class="danger" onclick="deleteAccount()">Delete</button><button type="button" onclick="testAccount()" style="background:#286a67">Test Gemini Key</button></div>
<p class="small" id="accountMessage">No Gemini account connected yet. Add one above, or just use a FREE image source.</p></div></section>

<section class="view" id="view-about"><div class="card"><h2>ℹ️ About / Help</h2>
<p class="small">This website runs on FREE image sources by default — no sign-in and no paid account is required. You can also optionally connect your own Gemini API key on the Connect Gemini page (saved once, then fully automatic).</p>
<p class="small"><b>Image sources:</b> Pollinations (free AI images, no key), Cloudflare FLUX (free account + token, entered per job and never saved), Gemini (your connected account), Openverse stock photos (no key), Pixabay and Pexels stock photos (free API key, entered per job and never saved), or your own uploaded pictures.</p>
<p class="small"><b>How it works:</b> paste a script in the Studio, pick an image source and a free voice, then press Launch. Videos are rendered one at a time in the Queue; finished videos appear under Deliverables.</p>
<p class="small"><b>Google Flow / Veo:</b> you can download a ready-made Prompt Sheet from the Studio, create clips yourself in Google Flow, and add them under "Veo / Flow scene clips" — the studio will add voiceover and captions and finish the video.</p>
</div></section>

<section class="view" id="view-console"><div class="card"><h2>🖥 Console — Job Log</h2><label>Select a job</label><select id="consoleJobSel" onchange="showConsole()"><option value="">— select a job —</option></select><div class="logbox" id="consoleLog" style="margin-top:12px">The log will appear here.</div></div></section>

</div></div>
<script>
async function api(path,data){const opts={method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(data||{})};const r=await fetch(path,opts);const x=await r.json();if(!r.ok)throw Error(x.error||'Request failed');return x;}
function switchView(name){document.querySelectorAll('.view').forEach(v=>v.classList.remove('active'));document.getElementById('view-'+name).classList.add('active');document.querySelectorAll('.navbtn').forEach(b=>b.classList.toggle('active',b.dataset.nav===name));if(name==='queue')loadJobs();if(name==='deliverables')loadVideos();if(name==='projects')loadProjects();if(name==='assets')fillJobSelects();if(name==='console')fillJobSelects();if(name==='flow')loadFlowProfiles();}
function msg(t){document.getElementById('accountMessage').textContent=t;}
async function refreshAccounts(selected){try{const r=await fetch('/api/accounts');const x=await r.json();const sel=document.getElementById('accountSelect');sel.innerHTML='<option value="">-- Select Gemini account --</option>';for(const a of x.accounts){const opt=document.createElement('option');opt.value=a.name;opt.textContent=a.name;sel.appendChild(opt);}if(selected&&x.accounts.some(a=>a.name===selected))sel.value=selected;chooseAccount();}catch(e){msg(e.message)}}
function chooseAccount(){const sel=document.getElementById('accountSelect');document.getElementById('accountName').value=sel.value;document.getElementById('accountKey').value='';}
async function saveAccount(){const name=document.getElementById('accountName').value.trim(),key=document.getElementById('accountKey').value.trim(),model=document.getElementById('imageModel').value.trim();if(!name){msg('Account name is required.');return;}if(!key&&document.getElementById('accountSelect').value!==name){msg('An API key is required for a new account.');return;}try{await api('/api/accounts/save',{name,key,model});document.getElementById('accountKey').value='';await refreshAccounts(name);msg('Gemini account saved: '+name+' — it will now be used automatically in Gemini mode.');}catch(e){msg('Save error: '+e.message)}}
async function deleteAccount(){const name=document.getElementById('accountSelect').value;if(!name)return msg('Select an account.');if(!confirm('Delete Gemini account '+name+'?'))return;try{await api('/api/accounts/delete',{name});await refreshAccounts();msg('Account deleted.');}catch(e){msg(e.message)}}
async function testAccount(){const name=document.getElementById('accountSelect').value;if(!name)return msg('Select an account.');msg('Checking Gemini API key...');try{const x=await api('/api/accounts/test',{name});msg(x.message)}catch(e){msg('Gemini test failed: '+e.message)}}
let sceneEdits=[];
function sceneSplit(script){const paras=script.trim().split(/\n\s*\n/).map(s=>s.trim()).filter(Boolean);if(paras.length>1)return paras;const sentences=script.trim().split(/(?<=[.!?।؟])\s+/);let result=[];for(let i=0;i<sentences.length;i+=3)result.push(sentences.slice(i,i+3).join(' '));return result.filter(Boolean);}
function previewScenes(){const script=document.getElementById('script').value.trim();if(!script){alert('Paste a script first.');return;}const scenes=sceneSplit(script);if(scenes.length>300){alert('Maximum 300 scenes per video.');return;}sceneEdits=scenes.map(n=>({narration:n,prompt:n}));drawTimeline();}
function resetSceneEdits(){sceneEdits=[];document.getElementById('sceneTimeline').replaceChildren();document.getElementById('sceneInfo').textContent='Scene edits reset.';}
function drawTimeline(){const root=document.getElementById('sceneTimeline');root.replaceChildren();document.getElementById('sceneInfo').textContent=sceneEdits.length+' scenes · Edit the narration and visual prompt separately.';sceneEdits.forEach((scene,i)=>{const card=document.createElement('div');card.style.cssText='border:1px solid #344869;border-radius:10px;padding:12px;margin:10px 0;background:#0c172b';const title=document.createElement('strong');title.textContent='Scene '+(i+1);card.appendChild(title);for(const [key,label] of [['narration','Narration (spoken)'],['prompt','AI image / Flow prompt (visual only)']]){const l=document.createElement('label');l.textContent=label;const t=document.createElement('textarea');t.style.minHeight='80px';t.value=scene[key];t.addEventListener('input',()=>{sceneEdits[i][key]=t.value;});card.appendChild(l);card.appendChild(t);}root.appendChild(card);});}
function downloadFlowPrompts(){const script=document.getElementById('script').value.trim();if(!script){alert('Paste a script first.');return;}const style=document.querySelector('input[name=style]:checked').value;const bible=document.getElementById('characterBible').value.trim();const ratio=document.getElementById('ratio').value;const scenes=sceneEdits.length?sceneEdits.map(s=>s.prompt):sceneSplit(script);let txt='GOOGLE FLOW / VEO PROMPT SHEET\nStyle: '+style+' | Ratio: '+ratio+'\nCharacter: '+bible+'\n\nPaste the prompt for each scene into Flow, create the clip, download it, and add it in the Studio under Veo clips at the same scene number.\n\n';scenes.forEach((s,i)=>{txt+='--- SCENE '+(i+1)+' ---\n'+style+' style, aspect ratio '+ratio+'. '+(bible?('Consistent character: '+bible+'. '):'')+s+'\n\n';});const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([txt],{type:'text/plain'}));a.download='flow_prompts.txt';a.click();}
const PRESETS=[
['🧍','Stickman Hero','Stickman','Simple stickman hero, thin black stick figure with a round white head and a red scarf. Keep the exact same simple look in every scene, plain colorful backgrounds.'],
['🏛','Ancient Roman','3D Animated Pixar style','Marcus, an ancient Roman boy, short brown hair, white tunic with a red sash, leather sandals and a bronze bracelet. Keep his face, hair and clothes the same in every scene.'],
['👦','Pixar Boy','3D Animated Pixar style','A cute 8-year-old Pixar-style boy with big brown eyes, curly black hair, a yellow t-shirt and blue shorts. Keep his face, hair and outfit identical in every scene.'],
['🌸','Anime Girl','Anime','Yuki, an anime girl with long pink hair, big sparkling eyes and a blue school uniform with a red ribbon. Keep her design identical in every scene.'],
['🕌','Village Boy','2D Cartoon','Ahmed, a 10-year-old Pakistani village boy, short black hair, green shalwar kameez, kind smile and a small brown bag. Keep the same look in every scene.'],
['🤖','Robot Friend','3D Cartoon','Robo, a small friendly round robot, white and orange body, glowing blue eyes and a tiny antenna. Keep the same design in every scene.']];
function buildPresets(){const root=document.getElementById('presets');PRESETS.forEach(p=>{const d=document.createElement('div');d.className='preset';d.innerHTML='<span class="em">'+p[0]+'</span><span>'+p[1]+'</span>';d.onclick=()=>{document.getElementById('characterBible').value=p[3];const r=document.querySelector('input[name=style][value="'+p[2]+'"]');if(r)r.checked=true;};root.appendChild(d);});}
const EXTRA_STYLES=['Claymation','LEGO Brick','Pixel Art','Studio Ghibli','90s Retro Anime','Cyberpunk','Oil Painting','Pencil Sketch','Low Poly Game','Isometric','Paper Cutout','Miniature Diorama','Neon Noir','Vintage Film','Bollywood Poster','Chibi','Dark Fantasy','Watercolor','Flat Vector','Photorealistic Portrait'];
function buildExtraStyles(){const root=document.getElementById('styles');EXTRA_STYLES.forEach(s=>{const l=document.createElement('label');l.innerHTML='<input type="radio" name="style" value="'+s+'">'+s;root.appendChild(l);});}
let ALL_VOICES=[],currentVoiceFilter='All';
function favs(){try{return JSON.parse(localStorage.getItem('favVoices')||'[]');}catch(e){return[];}}
function renderVoices(filter,btn){if(typeof filter==='string')currentVoiceFilter=filter;if(btn){document.querySelectorAll('#voiceTabs .vtab').forEach(b=>b.classList.remove('active'));btn.classList.add('active');}
const q=(document.getElementById('voiceSearch').value||'').toLowerCase();const sel=document.getElementById('voice');const cur=sel.value;sel.innerHTML='';
let list=ALL_VOICES.filter(v=>{if(currentVoiceFilter==='Fav'&&!favs().includes(v.short))return false;if(currentVoiceFilter!=='All'&&currentVoiceFilter!=='Fav'&&!(v.name+' '+v.locale).toLowerCase().includes(currentVoiceFilter.toLowerCase()))return false;if(q&&!(v.name+' '+v.locale+' '+v.gender+' '+v.short).toLowerCase().includes(q))return false;return true;});
list.slice(0,600).forEach(v=>{const o=document.createElement('option');o.value=v.short;o.textContent=(favs().includes(v.short)?'⭐ ':'')+v.locale+' · '+v.name.replace(/^Microsoft\s*/,'')+' ('+v.gender+')';sel.appendChild(o);});
if(cur&&[...sel.options].some(o=>o.value===cur))sel.value=cur;
document.getElementById('voiceMsg').textContent=list.length+' voices found (total '+ALL_VOICES.length+'). Press the ⭐ Favourite button to add a voice to Favourites.';}
async function loadVoices(){try{const r=await fetch('/api/voices');const x=await r.json();ALL_VOICES=x.voices||[];}catch(e){ALL_VOICES=[];}
if(!ALL_VOICES.length){ALL_VOICES=[{short:'en-US-JennyNeural',locale:'en-US',gender:'Female',name:'Jenny'}];}
renderVoices('All');}
function toggleFav(){const v=document.getElementById('voice').value;if(!v)return;let f=favs();if(f.includes(v))f=f.filter(x=>x!==v);else f.push(v);localStorage.setItem('favVoices',JSON.stringify(f));renderVoices(currentVoiceFilter);}
let previewAudio=null;
async function previewVoice(){const v=document.getElementById('voice').value;if(!v)return;const m=document.getElementById('voiceMsg');m.textContent='Generating preview...';try{const x=await api('/api/voice-preview',{voice:v});if(previewAudio)previewAudio.pause();previewAudio=new Audio(x.url);previewAudio.play();m.textContent='Playing preview.';}catch(e){m.textContent='Preview error: '+e.message;}}
function collectSettings(){return{script:document.getElementById('script').value,characterBible:document.getElementById('characterBible').value,style:document.querySelector('input[name=style]:checked').value,voice:document.getElementById('voice').value,ratio:document.getElementById('ratio').value,speed:document.getElementById('speed').value,imageMode:document.getElementById('imageMode').value,transitions:document.getElementById('transitions').value,motion:document.getElementById('motion').value,captionStyle:document.getElementById('captionStyle').value,captions:document.getElementById('captions').checked};}
function applySettings(s){if(!s)return;document.getElementById('script').value=s.script||'';document.getElementById('characterBible').value=s.characterBible||'';const r=document.querySelector('input[name=style][value="'+s.style+'"]');if(r)r.checked=true;if(s.voice){const sel=document.getElementById('voice');const o=document.createElement('option');o.value=s.voice;o.textContent=s.voice;sel.appendChild(o);sel.value=s.voice;}
document.getElementById('ratio').value=s.ratio||'9:16';document.getElementById('speed').value=s.speed||'+0%';const imSel=document.getElementById('imageMode');imSel.value=s.imageMode||'pollinations';if(!imSel.value)imSel.value='pollinations';document.getElementById('transitions').value=s.transitions||'mixed';document.getElementById('motion').value=s.motion||'pan';document.getElementById('captionStyle').value=s.captionStyle||'bold';document.getElementById('captions').checked=s.captions!==false;sceneEdits=[];document.getElementById('sceneTimeline').replaceChildren();}
async function saveProject(){const name=document.getElementById('projectName').value.trim();if(!name){alert('Enter a project name.');return;}const sel=document.getElementById('scriptSelect');try{const x=await api('/api/projects/save',{id:sel.value||'',name,data:collectSettings()});document.getElementById('projectMsg').textContent='Project saved: '+name;await loadProjects();document.getElementById('scriptSelect').value=x.id;}catch(e){document.getElementById('projectMsg').textContent='Save error: '+e.message;}}
async function loadProjects(){try{const r=await fetch('/api/projects');const x=await r.json();const sel=document.getElementById('scriptSelect');const cur=sel.value;sel.innerHTML='<option value="">— Choose a project —</option>';x.projects.forEach(p=>{const o=document.createElement('option');o.value=p.id;o.textContent=p.name;sel.appendChild(o);});if(cur)sel.value=cur;
const badge=document.getElementById('draftCount');badge.style.display=x.projects.length?'inline-block':'none';badge.textContent=x.projects.length;
const root=document.getElementById('projectsList');root.innerHTML=x.projects.length?'':'<p class="small">No saved projects.</p>';
x.projects.forEach(p=>{const d=document.createElement('div');d.className='item';d.innerHTML='<b></b> <span class="small"></span>';d.querySelector('b').textContent=p.name;d.querySelector('span').textContent='· script '+p.chars+' characters · '+new Date(p.updated*1000).toLocaleString();const b=document.createElement('button');b.textContent='Open in Studio';b.style.marginLeft='10px';b.onclick=async()=>{await applyProject(p.id);switchView('studio');};d.appendChild(b);root.appendChild(d);});}catch(e){}}
async function applyProject(id){const r=await fetch('/api/projects/'+id);const x=await r.json();if(x.project){document.getElementById('projectName').value=x.project.name;applySettings(x.project.data);document.getElementById('projectMsg').textContent='Project loaded: '+x.project.name;}}
async function applySelectedProject(){const id=document.getElementById('scriptSelect').value;if(id)applyProject(id);}
async function deleteSelectedProject(){const id=document.getElementById('scriptSelect').value;if(!id){alert('Select a project first.');return;}if(!confirm('Delete this project?'))return;await api('/api/projects/delete',{id});document.getElementById('scriptSelect').value='';await loadProjects();document.getElementById('projectMsg').textContent='Project deleted.';}
async function writeScript(){const topic=document.getElementById('writerTopic').value.trim();if(!topic){alert('Enter a topic.');return;}const btn=event.target;btn.disabled=true;btn.textContent='✍ Writing...';try{const x=await api('/api/writer',{topic,language:document.getElementById('writerLang').value});document.getElementById('script').value=x.script;sceneEdits=[];document.getElementById('projectMsg').textContent='The AI Writer created the script — use Preview Scenes to check it.';}catch(e){alert(e.message);}btn.disabled=false;btn.textContent='✍ FREE AI Writer — Create Script';}
async function loadFlowProfiles(){try{const r=await fetch('/api/flow/profiles');const x=await r.json();const root=document.getElementById('flowList');root.innerHTML=x.profiles.length?'':'<p class="small">No saved Flow profile.</p>';x.profiles.forEach(p=>{const d=document.createElement('div');d.className='item';const b=document.createElement('b');b.textContent=p.name;d.appendChild(b);const s=document.createElement('span');s.className='small';s.textContent=' '+(p.note||'');d.appendChild(s);const open=document.createElement('button');open.textContent='🌐 Open & Login';open.style.margin='8px 6px 0 0';open.onclick=openFlow;const del=document.createElement('button');del.textContent='Delete';del.className='danger';del.onclick=async()=>{await api('/api/flow/profiles/delete',{id:p.id});loadFlowProfiles();};d.appendChild(document.createElement('br'));d.appendChild(open);d.appendChild(del);root.appendChild(d);});}catch(e){}}
async function saveFlowProfile(){const name=document.getElementById('flowName').value.trim();if(!name){alert('Enter a profile name.');return;}try{await api('/api/flow/profiles/save',{name,note:document.getElementById('flowNote').value.trim()});document.getElementById('flowMsg').textContent='Profile saved: '+name;document.getElementById('flowName').value='';document.getElementById('flowNote').value='';loadFlowProfiles();}catch(e){alert(e.message);}}
async function openFlow(){try{await api('/api/flow/open',{});document.getElementById('flowMsg').textContent='Google Flow is opening in the browser on this PC — sign in there with your Google account. (If you opened this from mobile, the Flow link will open on the PC, not on the phone.)';}catch(e){alert(e.message+' — Open labs.google/fx/tools/flow in a browser yourself.');}}
async function loadJobs(){try{const r=await fetch('/api/jobs');const x=await r.json();const active=x.jobs.filter(j=>j.state==='queued'||j.state==='running').length;const badge=document.getElementById('queueCount');badge.style.display=active?'inline-block':'none';badge.textContent=active;
const root=document.getElementById('queueList');if(!root)return;root.innerHTML=x.jobs.length?'':'<p class="small">No jobs yet.</p>';
x.jobs.forEach(j=>{const d=document.createElement('div');d.className='item';d.innerHTML='<b></b> <span class="pill '+j.state+'">'+j.state+'</span> <span class="small"></span>';d.querySelector('b').textContent=j.name||'Video job';d.querySelector('.small').textContent='· '+j.progress+'% · '+(j.message||'');if(j.url){const a=document.createElement('a');a.href=j.url;a.textContent='⬇ Download MP4';a.style.marginLeft='10px';d.appendChild(a);}const c=document.createElement('button');c.textContent='Console';c.className='ghost';c.style.marginLeft='8px';c.onclick=()=>{switchView('console');document.getElementById('consoleJobSel').value=j.job;showConsole();};d.appendChild(c);root.appendChild(d);});}catch(e){}}
async function fillJobSelects(){try{const r=await fetch('/api/jobs');const x=await r.json();for(const id of ['assetsJobSel','consoleJobSel']){const sel=document.getElementById(id);const cur=sel.value;sel.innerHTML='<option value="">— select a job —</option>';x.jobs.forEach(j=>{const o=document.createElement('option');o.value=j.job;o.textContent=(j.name||j.job.slice(0,8))+' · '+j.state;sel.appendChild(o);});if(cur)sel.value=cur;}}catch(e){}}
async function loadAssets(){const job=document.getElementById('assetsJobSel').value;const grid=document.getElementById('assetsGrid');grid.innerHTML='';if(!job)return;try{const r=await fetch('/api/assets/'+job);const x=await r.json();const frames=x.files.filter(f=>f.name.startsWith('frame'));const show=frames.length?frames:x.files;show.forEach(f=>{const img=document.createElement('img');img.src=f.url;img.title=f.name;grid.appendChild(img);});if(!show.length)grid.innerHTML='<p class="small">No images have been created for this job yet.</p>';}catch(e){}}
async function loadVideos(){try{const r=await fetch('/api/videos');const x=await r.json();const root=document.getElementById('videosList');root.innerHTML=x.videos.length?'':'<p class="small">No finished video yet.</p>';x.videos.forEach(v=>{const d=document.createElement('div');d.className='item';d.innerHTML='<b></b> <span class="small"></span> ';d.querySelector('b').textContent=v.name;d.querySelector('.small').textContent='· '+v.size_mb+' MB · '+new Date(v.mtime*1000).toLocaleString();const a=document.createElement('a');a.href=v.url;a.textContent='⬇ Download / Play';d.appendChild(a);root.appendChild(d);});}catch(e){}}
async function showConsole(){const job=document.getElementById('consoleJobSel').value;const box=document.getElementById('consoleLog');if(!job){box.textContent='Select a job.';return;}try{const r=await fetch('/api/status/'+job);const x=await r.json();box.textContent=(x.log||[]).join('\n')||x.message||'—';}catch(e){box.textContent='Log not found.';}}
async function verifySystem(){const el=document.getElementById('verifyResult');el.textContent='Checking...';try{const r=await fetch('/api/verify');const x=await r.json();el.innerHTML=x.checks.map(c=>(c.ok?'✔ ':'✘ ')+c.name+' — '+c.detail).join('<br>');}catch(e){el.textContent='Verify error: '+e.message;}}
let timer=null;
async function launch(queuedOnly){let script=document.getElementById('script').value.trim();if(!script){alert('Please paste your script first.');return;}
const mode=document.getElementById('imageMode').value;
if(mode==='gemini'&&!document.getElementById('accountSelect').value){alert('Gemini mode: connect and select a Gemini account on the Connect Gemini page — or use one of the FREE sources (Pollinations / Openverse).');switchView('connect');return;}
if(mode==='pixabay'&&!document.getElementById('pixabayKey').value.trim()){alert('Pixabay mode: enter your free Pixabay API key in the Image Source section — or use Openverse stock photos (no key needed).');return;}
if(mode==='pexels'&&!document.getElementById('pexelsKey').value.trim()){alert('Pexels mode: enter your free Pexels API key in the Image Source section — or use Openverse stock photos (no key needed).');return;}
if(mode==='cloudflare'&&(!document.getElementById('cfAccount').value.trim()||!document.getElementById('cfToken').value.trim())){alert('For Cloudflare mode, enter the Account ID and Token.');return;}
let b=document.getElementById('launch');b.disabled=true;document.getElementById('result').innerHTML='';document.getElementById('status').textContent=queuedOnly?'Adding to queue...':'Starting...';document.getElementById('progress').value=0;
try{let r=await fetch('/api/launch',{method:'POST',body:(()=>{let f=new FormData();let s=collectSettings();let vals={script:s.script,style:s.style,voice:s.voice,ratio:s.ratio,speed:s.speed,captions:s.captions,transitions:s.transitions,motion:s.motion,captionStyle:s.captionStyle,imageMode:s.imageMode,accountName:document.getElementById('accountSelect').value,cfAccount:document.getElementById('cfAccount').value,cfToken:document.getElementById('cfToken').value,pixabayKey:document.getElementById('pixabayKey').value,pexelsKey:document.getElementById('pexelsKey').value,characterBible:s.characterBible,projectName:document.getElementById('projectName').value,sceneEdits:sceneEdits.length?JSON.stringify(sceneEdits):''};for(let [k,v] of Object.entries(vals))f.append(k,v);for(let file of document.getElementById('uploads').files)f.append('images',file);for(let file of document.getElementById('sceneClips').files)f.append('sceneclips',file);const ov=document.getElementById('ownVoice').files[0];if(ov)f.append('voiceover',ov);const mf=document.getElementById('musicFile').files[0];if(mf)f.append('musicfile',mf);return f;})()});
let x=await r.json();if(!r.ok)throw Error(x.error||'Failed to start');
if(queuedOnly){b.disabled=false;document.getElementById('status').textContent='Added to the queue ✔ — check the status on the Queue page.';loadJobs();return;}
clearInterval(timer);timer=setInterval(async()=>{try{let q=await fetch('/api/status/'+x.job);let j=await q.json();document.getElementById('status').textContent=j.message;document.getElementById('progress').value=j.progress;loadJobs();
if(j.state==='done'){clearInterval(timer);b.disabled=false;document.getElementById('result').innerHTML='<p><a href="'+j.url+'" download>⬇ Download Final MP4</a></p><video controls src="'+j.url+'"></video>';loadVideos();}
if(j.state==='error'){clearInterval(timer);b.disabled=false;document.getElementById('status').textContent='Error: '+j.message;}}catch(e){clearInterval(timer);b.disabled=false;}},1200);
}catch(e){b.disabled=false;document.getElementById('status').textContent=e.message;}}
buildPresets();buildExtraStyles();loadVoices();refreshAccounts();loadProjects();loadJobs();
setInterval(()=>{loadJobs();},8000);
</script></body></html>
'''


def update(job, progress, message, state='running', url=None):
    with LOCK:
        old = JOBS.get(job, {})
        log = list(old.get('log', []))
        line = f"[{time.strftime('%H:%M:%S')}] {message}"
        if not log or log[-1].split('] ', 1)[-1] != message:
            log.append(line)
        JOBS[job] = {'progress': progress, 'message': message, 'state': state,
                     'url': url if url is not None else old.get('url'),
                     'name': old.get('name', ''), 'created': old.get('created', time.time()),
                     'log': log[-120:]}


def _queue_worker():
    while True:
        job, data = WORK_Q.get()
        try:
            run_job(job, data)
        except Exception:
            pass
        finally:
            WORK_Q.task_done()


threading.Thread(target=_queue_worker, daemon=True).start()


def split_scenes(script):
    paragraphs = [p.strip() for p in re.split(r'\n\s*\n', script) if p.strip()]
    if len(paragraphs) > 1:
        return paragraphs
    sentences = re.split(r'(?<=[.!?।؟])\s+', script.strip())
    return [' '.join(sentences[i:i+3]) for i in range(0, len(sentences), 3) if sentences[i:i+3]]


def font(size):
    for p in [r'C:\Windows\Fonts\arial.ttf', r'C:\Windows\Fonts\segoeui.ttf',
              '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf']:
        if Path(p).exists():
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def wrap_text(draw, text, fnt, max_width):
    words = text.split()
    lines, line = [], ''
    for word in words:
        candidate = (line + ' ' + word).strip()
        if draw.textbbox((0, 0), candidate, font=fnt)[2] <= max_width:
            line = candidate
        else:
            if line: lines.append(line)
            line = word
    if line: lines.append(line)
    return lines


def dimensions(ratio):
    return {'9:16': (720, 1280), '16:9': (1280, 720), '1:1': (900, 900)}.get(ratio, (720, 1280))


def scene_image(source, target, text, ratio, captions, caption_style):
    w, h = dimensions(ratio)
    with Image.open(source) as raw:
        img = ImageOps.fit(raw.convert('RGB'), (w, h), method=Image.Resampling.LANCZOS)
    if captions:
        d = ImageDraw.Draw(img, 'RGBA')
        f = font(max(26, int(w * (.049 if caption_style in ('bold', 'comic', 'highlight') else .036))))
        # Display concise per-scene subtitles, not the entire paragraph.
        snippet = re.sub(r'\s+', ' ', text).strip()[:190]
        if caption_style == 'comic':
            snippet = snippet.upper()
        lines = wrap_text(d, snippet, f, int(w*.81))[:5]
        step = max(36, int(w*.063))
        block_height = len(lines)*step + 30
        y0 = int(h*.79)-block_height
        if caption_style in ('box','bold'):
            d.rounded_rectangle((int(w*.07), y0, int(w*.93), y0+block_height), radius=18,
                                fill=(5, 9, 18, 185 if caption_style == 'box' else 130))
        txt_fill = {'green': (96, 255, 132), 'comic': (255, 236, 92)}.get(caption_style, 'white')
        y = y0+15
        for line in lines:
            box = d.textbbox((0, 0), line, font=f, stroke_width=2)
            x = (w-(box[2]-box[0]))//2
            if caption_style == 'highlight':
                d.rectangle((x-8, y+3, x+(box[2]-box[0])+8, y+step-5), fill=(255, 221, 0, 255))
                d.text((x,y), line, font=f, fill=(10, 10, 10), stroke_width=0)
            else:
                d.text((x,y), line, font=f, fill=txt_fill, stroke_width=2, stroke_fill=(0,0,0,255))
            y += step
    img.save(target, quality=93)


def _image_prompt(scene, style, ratio, continuity, character_bible=""):
    return (f'Create ONE original, detailed, beautiful {style} illustration / film still, '
            f'for a vertical or horizontal short-form story. Frame aspect ratio {ratio}. '
            'No text, no subtitles, no logos, no watermark, no split panels. '
            'Keep character identity, wardrobe and setting consistent with the story. '
            f'Character reference (maintain these traits): {character_bible[:1400]}. Overall story context: {continuity[:900]}. Scene to illustrate: {scene[:1800]}. '
            'Show a specific clear physical action and environment described by this scene.')


def _caption_overlay(text, ratio, caption_style, output):
    """Transparent PNG with only the caption block - for overlaying on Veo clips."""
    w, h = dimensions(ratio)
    img = Image.new('RGBA', (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(img, 'RGBA')
    f = font(max(26, int(w * (.049 if caption_style in ('bold', 'comic', 'highlight') else .036))))
    snippet = re.sub(r'\s+', ' ', text).strip()[:190]
    if caption_style == 'comic':
        snippet = snippet.upper()
    lines = wrap_text(d, snippet, f, int(w*.81))[:5]
    step = max(36, int(w*.063))
    block_height = len(lines)*step + 30
    y0 = int(h*.79)-block_height
    if caption_style in ('box','bold'):
        d.rounded_rectangle((int(w*.07), y0, int(w*.93), y0+block_height), radius=18,
                            fill=(5, 9, 18, 185 if caption_style == 'box' else 130))
    txt_fill = {'green': (96, 255, 132, 255), 'comic': (255, 236, 92, 255)}.get(caption_style, (255, 255, 255, 255))
    y = y0+15
    for line in lines:
        box = d.textbbox((0, 0), line, font=f, stroke_width=2)
        x = (w-(box[2]-box[0]))//2
        if caption_style == 'highlight':
            d.rectangle((x-8, y+3, x+(box[2]-box[0])+8, y+step-5), fill=(255, 221, 0, 255))
            d.text((x,y), line, font=f, fill=(10, 10, 10, 255), stroke_width=0)
        else:
            d.text((x,y), line, font=f, fill=txt_fill, stroke_width=2, stroke_fill=(0,0,0,255))
        y += step
    img.save(output)


def _placeholder_card(scene, ratio, output):
    """Local fallback card so ONE failed free image never kills the whole video."""
    w, h = dimensions(ratio)
    img = Image.new('RGB', (w, h), (16, 27, 51))
    d = ImageDraw.Draw(img)
    f = font(max(28, int(w * .05)))
    lines = wrap_text(d, re.sub(r'\s+', ' ', scene).strip()[:220], f, int(w * .84))[:8]
    y = h // 2 - len(lines) * max(38, int(w * .07)) // 2
    for line in lines:
        box = d.textbbox((0, 0), line, font=f)
        d.text(((w - (box[2] - box[0])) // 2, y), line, font=f, fill=(220, 232, 255))
        y += max(38, int(w * .07))
    img.save(output)


def generate_pollinations_image(scene, style, ratio, output, continuity, character_bible="", seed=42):
    """FREE image generation via Pollinations (no API key, no billing).

    Verified live on 2026-10-21: the anonymous endpoint returns real
    prompt-matched images. BUT the free service is flaky: it sporadically
    returns HTTP 404 / 402 / 429 even for good prompts (the first shipped
    version did not retry 404, so one bad reply killed the whole video -
    that is what Yasir hit). Strategy now:
      attempt 1-2: full prompt + model=flux
      attempt 3:   same prompt, no model/nologo params (server fallback)
      attempt 4:   SHORT prompt (style + character + scene only)
    Every HTTP error is retried with backoff. Raises only after all 4.
    Uses only the Python standard library (no new dependency).
    """
    prompt = _image_prompt(scene, style, ratio, continuity, character_bible)
    short_prompt = (f'{style} illustration, detailed, colorful. '
                    + (f'Character: {character_bible[:200]}. ' if character_bible else '')
                    + f'Scene: {scene[:500]}. No text, no watermark, no split panels.')
    w, h = {'9:16': (720, 1280), '16:9': (1280, 720), '1:1': (1024, 1024)}.get(ratio, (720, 1280))
    base = 'https://image.pollinations.ai/prompt/'
    urls = [
        base + urllib.parse.quote(prompt, safe='') + f'?width={w}&height={h}&seed={int(seed)}&model=flux&nologo=true',
        base + urllib.parse.quote(prompt, safe='') + f'?width={w}&height={h}&seed={int(seed) + 1}&model=flux&nologo=true',
        base + urllib.parse.quote(prompt, safe='') + f'?width={w}&height={h}&seed={int(seed) + 2}',
        base + urllib.parse.quote(short_prompt, safe='') + f'?width={w}&height={h}&seed={int(seed) + 3}',
    ]
    last_error = None
    for attempt, url in enumerate(urls):
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'AI-Video-Studio-Pro/1.0'})
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = resp.read()
            with Image.open(io.BytesIO(data)) as img:  # raises if reply is not an image
                img.convert('RGB').save(output)
            return
        except Exception as exc:  # ANY failure (404/402/429/5xx/network): back off and try the next variant
            last_error = exc
            if attempt < len(urls) - 1:
                time.sleep(10 * (attempt + 1))
    raise RuntimeError('Pollinations FREE image could not be created (after 4 attempts). '
                       f'Original error: {str(last_error)[:250]}')


def generate_cloudflare_image(account_id, token, scene, style, ratio, output, continuity, character_bible="", seed=42):
    """FREE image generation via Cloudflare Workers AI (FLUX.1 schnell).

    Cloudflare's official pricing gives every account 10,000 free Neurons
    per day; one flux-1-schnell image costs roughly 58 Neurons, i.e. about
    170 free images per day. Needs a free Cloudflare account: Account ID +
    API Token (token is used only for this job, never saved to disk).
    Response format per Cloudflare docs: JSON with result.image (base64).
    """
    prompt = _image_prompt(scene, style, ratio, continuity, character_bible)
    endpoint = (f'https://api.cloudflare.com/client/v4/accounts/{account_id}'
                '/ai/run/@cf/black-forest-labs/flux-1-schnell')
    payload = json.dumps({'prompt': prompt, 'seed': int(seed), 'steps': 4}).encode('utf-8')
    req = urllib.request.Request(endpoint, data=payload, method='POST',
                                 headers={'Authorization': f'Bearer {token}',
                                          'Content-Type': 'application/json'})
    try:
        with urllib.request.urlopen(req, timeout=180) as resp:
            data = json.loads(resp.read().decode('utf-8'))
    except urllib.error.HTTPError as exc:
        detail = ''
        try:
            detail = exc.read().decode('utf-8', 'replace')[:300]
        except Exception:
            pass
        if exc.code in (401, 403):
            raise RuntimeError('Cloudflare TOKEN/Account ID appears to be incorrect (HTTP ' + str(exc.code) + '). '
                               'Check the Account ID and the API Token with Workers AI permission again in the Cloudflare dashboard. ' + detail) from exc
        raise RuntimeError(f'Cloudflare FREE image error (HTTP {exc.code}). Your daily free Neurons may have run out '
                           '(you receive 10,000 free Neurons every day; please try again tomorrow), or the service may be busy. ' + detail) from exc
    image_b64 = ((data.get('result') or {}).get('image') if isinstance(data, dict) else None)
    if not image_b64:
        errors = data.get('errors') if isinstance(data, dict) else None
        raise RuntimeError('Cloudflare did not return an image. Response: ' + str(errors or data)[:300]
                           + ' - Common causes: an incorrect Account ID/token, or the daily free Neurons have run out.')
    with Image.open(io.BytesIO(base64.b64decode(image_b64))) as img:
        img.convert('RGB').save(output)


def generate_gemini_image(client, model, scene, style, ratio, output, continuity, character_bible=""):
    from google.genai import types
    model = _normalize_image_model(model)
    prompt = (f'Create ONE original, detailed, beautiful {style} illustration / film still, '
              f'for a vertical or horizontal short-form story. Frame aspect ratio {ratio}. '
              'No text, no subtitles, no logos, no watermark, no split panels. '
              'Keep character identity, wardrobe and setting consistent with the story. '
              f'Character reference (maintain these traits): {character_bible[:1400]}. Overall story context: {continuity[:900]}. Scene to illustrate: {scene[:1800]}. '
              'Show a specific clear physical action and environment described by this scene.')
    config_kwargs = {'response_modalities': ['TEXT', 'IMAGE']}
    if ratio in ('9:16', '16:9', '1:1', '4:3', '3:4'):
        try:
            config_kwargs['image_config'] = types.ImageConfig(aspect_ratio=ratio)
        except Exception:
            pass
    try:
        result = client.models.generate_content(
            model=model, contents=prompt,
            config=types.GenerateContentConfig(**config_kwargs))
    except TypeError:
        # Older google-genai SDK without image_config support: retry plain.
        result = client.models.generate_content(
            model=model, contents=prompt,
            config=types.GenerateContentConfig(response_modalities=['TEXT', 'IMAGE']))
    except Exception as exc:
        raise RuntimeError(_friendly_gemini_error(exc)) from exc
    # Collect parts from BOTH the new response.parts shortcut and the classic
    # candidates path. The old code crashed with
    # "'NoneType' object has no attribute 'parts'" whenever a candidate had no
    # content (safety block / empty reply), hiding the real reason.
    parts, texts, reasons = [], [], []
    try:
        parts.extend(list(getattr(result, 'parts', None) or []))
    except Exception:
        pass
    for candidate in getattr(result, 'candidates', None) or []:
        if getattr(candidate, 'finish_reason', None):
            reasons.append(str(candidate.finish_reason))
        content = getattr(candidate, 'content', None)
        if content is None:
            continue
        parts.extend(getattr(content, 'parts', None) or [])
    for part in parts:
        inline = getattr(part, 'inline_data', None)
        if inline is not None and getattr(inline, 'data', None):
            data = inline.data
            if isinstance(data, str):  # some SDK versions hand back base64 text
                data = base64.b64decode(data)
            with Image.open(io.BytesIO(data)) as img:
                img.convert('RGB').save(output)
            return
        txt = getattr(part, 'text', None)
        if txt:
            texts.append(str(txt).strip())
    detail = ' '.join(texts)[:400]
    raise RuntimeError(
        'Gemini did NOT return any image.'
        + (f' Finish reason: {", ".join(reasons)}.' if reasons else '')
        + (f' Model text reply: {detail}.' if detail else '')
        + ' Common causes: (1) billing is not enabled in the project that owns the key - image models are not available on the free tier, '
        + f'(2) the model is old/discontinued - try {DEFAULT_IMAGE_MODEL}, (3) a prompt safety block. '
        + 'First check with "Test Gemini Key", then try again with one short script.')


# --- FREE stock photos (Openverse / Pixabay / Pexels) ------------------------
_STOCK_STOPWORDS = {
    'the', 'and', 'for', 'with', 'from', 'that', 'this', 'then', 'than', 'but',
    'was', 'were', 'are', 'has', 'have', 'had', 'will', 'would', 'can', 'could',
    'there', 'here', 'into', 'out', 'over', 'very', 'just', 'one', 'scene',
    'shows', 'showing', 'show', 'image', 'photo', 'picture', 'video',
    'illustration', 'style', 'detailed', 'beautiful', 'create', 'original',
    'short', 'story', 'frame', 'aspect', 'ratio', 'text', 'keep', 'same',
    'every', 'his', 'her', 'its', 'their', 'they', 'she', 'him', 'them',
    'through', 'about', 'when', 'what', 'who', 'all', 'any',
}


def _stock_keywords(text, max_words=8):
    """Turn a scene prompt into a short stock-search query (meaningful words only)."""
    picked, seen = [], set()
    for word in re.findall(r"[A-Za-z][A-Za-z'-]{2,}", str(text or '')):
        low = word.lower()
        if low in _STOCK_STOPWORDS or low in seen:
            continue
        seen.add(low)
        picked.append(word)
        if len(picked) >= max_words:
            break
    return ' '.join(picked) if picked else 'nature landscape'


def _stock_orientation(ratio, provider):
    if provider == 'pixabay':
        return {'9:16': 'vertical', '16:9': 'horizontal'}.get(ratio, 'all')
    return {'9:16': 'portrait', '16:9': 'landscape', '1:1': 'square'}.get(ratio, 'landscape')


def _stock_search_urls(provider, api_key, query, ratio):
    """One search request -> list of direct image URLs (possibly empty)."""
    headers = {'User-Agent': 'AI-Video-Studio-Pro/1.0'}
    if provider == 'pixabay':
        params = {'key': api_key, 'q': query, 'image_type': 'photo', 'per_page': 5,
                  'orientation': _stock_orientation(ratio, provider), 'safesearch': 'true'}
        url = 'https://pixabay.com/api/?' + urllib.parse.urlencode(params)
    elif provider == 'pexels':
        params = {'query': query, 'per_page': 5,
                  'orientation': _stock_orientation(ratio, provider)}
        url = 'https://api.pexels.com/v1/search?' + urllib.parse.urlencode(params)
        headers['Authorization'] = api_key
    else:  # openverse - no key required
        params = {'q': query, 'page_size': 5, 'license_type': 'all'}
        url = 'https://api.openverse.org/v1/images/?' + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.loads(resp.read().decode('utf-8'))
    urls = []
    if provider == 'pixabay':
        for hit in data.get('hits') or []:
            u = hit.get('largeImageURL') or hit.get('webformatURL')
            if u:
                urls.append(u)
    elif provider == 'pexels':
        for photo in data.get('photos') or []:
            src = photo.get('src') or {}
            u = src.get('large2x') or src.get('large') or src.get('original')
            if u:
                urls.append(u)
    else:
        for item in data.get('results') or []:
            u = item.get('url') or item.get('foreign_landing_url')
            if u:
                urls.append(u)
    return urls


def _download_image_url(url, output):
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 (AI Video Studio Pro)'})
    with urllib.request.urlopen(req, timeout=90) as resp:
        data = resp.read()
    with Image.open(io.BytesIO(data)) as img:  # raises if the reply is not an image
        img.convert('RGB').save(output)


def generate_stock_photo(provider, api_key, scene_prompt, ratio, output, seed=42):
    """Fetch ONE free stock photo for a scene.

    Openverse needs no key; Pixabay/Pexels use a free key that is held only
    in memory for this job and never saved to disk. The search query is
    derived from the scene's image prompt; if a search finds nothing, retry
    once with a shorter query, then raise so the caller can fall back to a
    placeholder card (same behaviour as Pollinations failures).
    """
    label = {'openverse': 'Openverse', 'pixabay': 'Pixabay', 'pexels': 'Pexels'}[provider]
    try:
        query = _stock_keywords(scene_prompt, 8)
        urls = _stock_search_urls(provider, api_key, query, ratio)
        if not urls:
            short = ' '.join(query.split()[:3])
            if short and short != query:
                urls = _stock_search_urls(provider, api_key, short, ratio)
        if not urls:
            raise RuntimeError(f'{label} found no photo for this scene.')
        # Prefer a seed-offset result so different scenes do not all reuse the
        # same first hit; on failure walk the remaining results in order.
        start = int(seed) % len(urls)
        ordered = [urls[(start + k) % len(urls)] for k in range(len(urls))]
        last_error = None
        for u in ordered[:3]:
            try:
                _download_image_url(u, output)
                return
            except Exception as exc:
                last_error = exc
        raise RuntimeError(f'{label} photo download failed: {str(last_error)[:200]}')
    except Exception as exc:
        message = str(exc)
        if api_key:
            message = message.replace(str(api_key), '[REDACTED]')
        raise RuntimeError(f'{label} stock photo error: {message[:300]}') from exc


def probe_duration(path):
    p = subprocess.run(['ffprobe','-v','error','-show_entries','format=duration','-of','default=noprint_wrappers=1:nokey=1',str(path)], capture_output=True,text=True,timeout=30)
    if p.returncode: raise RuntimeError('ffprobe failed: '+p.stderr[-300:])
    return float(p.stdout.strip())


def run_job(job, data):
    try:
        if not shutil.which('ffmpeg') or not shutil.which('ffprobe'):
            raise RuntimeError('FFmpeg/ffprobe not found. Install FFmpeg and reopen CMD.')
        script = data['script'].strip()
        scenes = split_scenes(script)
        edits = data.get('sceneEditsParsed')
        if edits:
            scenes = [entry['narration'] for entry in edits]
        if not scenes: raise ValueError('Please enter a script.')
        if len(scenes) > MAX_SCENES: raise ValueError(f'This version allows a maximum of {MAX_SCENES} scenes in one video.')
        mode = data.get('imageMode', 'pollinations')
        if mode == 'gemini' and not data.get('apiKey'):
            raise ValueError('Select a connected Gemini account first, or choose one of the free image sources.')
        if mode == 'cloudflare' and not (data.get('cfAccount') and data.get('cfToken')):
            raise ValueError('Account ID and API Token are required for Cloudflare FREE mode (enter them in the Image Source section), or select Pollinations FREE mode.')
        if mode == 'pixabay' and not data.get('pixabayKey'):
            raise ValueError('A free Pixabay API key is required for Pixabay stock photos (enter it in the Image Source section), or select Openverse stock photos (no key needed).')
        if mode == 'pexels' and not data.get('pexelsKey'):
            raise ValueError('A free Pexels API key is required for Pexels stock photos (enter it in the Image Source section), or select Openverse stock photos (no key needed).')
        uploads = data.get('uploads', [])
        if mode == 'upload' and not uploads and not data.get('clips'):
            raise ValueError('Upload at least one picture, or choose one of the free image sources.')
        work = OUT / job
        work.mkdir(exist_ok=True)
        audio = work / 'voice.mp3'
        if data.get('voiceoverPath'):
            update(job, 5, 'Using your uploaded voiceover...')
            audio = Path(data['voiceoverPath'])
            total_duration = probe_duration(audio)
            if total_duration <= 0: raise RuntimeError('Voiceover is empty.')
            weights = [max(1,len(s)) for s in scenes]
            durations = [total_duration*w/sum(weights) for w in weights]
        else:
            # Per-scene TTS: the voice for each scene is generated separately. This means (1) even for very
            # long scripts there is no duration limit - Edge TTS does not fail on small
            # requests, (2) the timing of each scene is EXACT, and
            # picture and voice synchronization is improved.
            update(job, 5, 'Generating voiceover (scene by scene)...')
            voice_name = _resolve_voice(data.get('voice'))
            parts = []
            async def speak_all():
                for idx, sc in enumerate(scenes):
                    piece = work / f'voice_{idx:04d}.mp3'
                    await edge_tts.Communicate(sc, voice_name,
                                               rate=data.get('speed','+0%')).save(str(piece))
                    parts.append(piece)
            asyncio.run(speak_all())
            durations = []
            for piece in parts:
                try:
                    durations.append(max(0.4, probe_duration(piece)))
                except Exception:
                    durations.append(3.0)
            concat_list = work / 'voice_parts.txt'
            with concat_list.open('w', encoding='utf-8') as fh:
                for piece in parts:
                    fh.write(f"file '{piece.name}'\n")
            audio = work / 'voice_full.m4a'
            p = subprocess.run(['ffmpeg','-y','-hide_banner','-loglevel','error','-f','concat','-safe','0',
                                '-i',str(concat_list),'-c:a','aac','-b:a','128k',str(audio)],
                               capture_output=True, text=True, timeout=1800)
            if p.returncode: raise RuntimeError('FFmpeg error while joining the voiceover: '+p.stderr[-600:])
            total_duration = sum(durations)
            if total_duration <= 0: raise RuntimeError('Voiceover is empty.')
        # Reuse one Gemini client per job to reduce connection overhead.
        client = None
        if mode == 'gemini':
            from google import genai
            client = genai.Client(api_key=data['apiKey'])
        # All images are generated before encoding. Gemini/Cloudflare errors stay
        # explicit; a failed FREE image/stock scene falls back to a local card
        # (counted in the final message) so one flaky reply can't kill the video.
        placeholder_scenes = []
        veo_clips = data.get('clips', [])
        scene_kinds = []
        for i, scene in enumerate(scenes):
            image_path = work / f'image_{i:04d}.png'
            visual_prompt = edits[i]['prompt'] if edits else scene
            update(job, 10+int(45*i/len(scenes)), f'Preparing picture {i+1}/{len(scenes)}...')
            if i < len(veo_clips):
                scene_kinds.append('veo')  # this scene has a Flow/Veo clip - no image will be generated
                continue
            scene_kinds.append('image')
            seed = int(hashlib.md5(f'{job}:{i}'.encode()).hexdigest()[:6], 16) % 100000
            if mode == 'gemini':
                generate_gemini_image(client, _normalize_image_model(data.get('imageModel')),
                                      visual_prompt, data.get('style','2D Cartoon'), data.get('ratio','9:16'),
                                      image_path, script, data.get('characterBible',''))
            elif mode == 'pollinations':
                if i: time.sleep(3)  # gentle pace on the free service, to avoid hitting limits
                update(job, 10+int(45*i/len(scenes)), f'FREE picture (Pollinations) {i+1}/{len(scenes)} is being created...')
                try:
                    generate_pollinations_image(visual_prompt, data.get('style','2D Cartoon'), data.get('ratio','9:16'),
                                                image_path, script, data.get('characterBible',''), seed=seed)
                except Exception:
                    placeholder_scenes.append(i + 1)
                    _placeholder_card(scene, data.get('ratio','9:16'), image_path)
                    update(job, 10+int(45*i/len(scenes)), f'Picture {i+1} could not be created by the free service - a card has been used for that scene, and the video is continuing...')
            elif mode == 'cloudflare':
                update(job, 10+int(45*i/len(scenes)), f'FREE picture (Cloudflare FLUX) {i+1}/{len(scenes)} is being created...')
                generate_cloudflare_image(data['cfAccount'], data['cfToken'],
                                          visual_prompt, data.get('style','2D Cartoon'), data.get('ratio','9:16'),
                                          image_path, script, data.get('characterBible',''), seed=seed)
            elif mode in ('openverse', 'pixabay', 'pexels'):
                provider_label = {'openverse': 'Openverse', 'pixabay': 'Pixabay', 'pexels': 'Pexels'}[mode]
                stock_key = data.get('pixabayKey') if mode == 'pixabay' else data.get('pexelsKey')
                update(job, 10+int(45*i/len(scenes)), f'FREE stock photo ({provider_label}) {i+1}/{len(scenes)} is being fetched...')
                try:
                    generate_stock_photo(mode, stock_key, visual_prompt,
                                         data.get('ratio','9:16'), image_path, seed=seed)
                except Exception:
                    placeholder_scenes.append(i + 1)
                    _placeholder_card(scene, data.get('ratio','9:16'), image_path)
                    update(job, 10+int(45*i/len(scenes)), f'Picture {i+1} could not be fetched from {provider_label} - a card has been used for that scene, and the video is continuing...')
            else:
                shutil.copyfile(uploads[i%len(uploads)], image_path)
            scene_image(image_path, work/f'frame_{i:04d}.jpg', scene, data.get('ratio','9:16'),
                        data.get('captions',True), data.get('captionStyle','bold'))
        update(job, 60, 'Rendering pictures, motion and transitions...')
        w,h=dimensions(data.get('ratio','9:16'))
        fps=24
        segments=[]
        transition=data.get('transitions','mixed')
        for i,duration in enumerate(durations):
            # Scene video frame count follows narration duration; no fixed duration dropdown.
            frames=max(2,round(duration*fps))
            path=work/f'clip_{i:04d}.mp4'
            if scene_kinds[i] == 'veo':
                # Flow/Veo clip: loop it to the scene duration, fix size/crop,
                # overlay captions, then treat it as a segment like the other scenes.
                src = veo_clips[i]
                fade_v = transition=='fade' or (transition=='mixed' and i%2==1)
                fade_chain = ''
                if fade_v and duration>0.8:
                    fd=min(.28,duration/5)
                    fade_chain = f',fade=t=in:st=0:d={fd:.3f},fade=t=out:st={max(0,duration-fd):.3f}:d={fd:.3f}'
                base_vf = f'scale={w}:{h}:force_original_aspect_ratio=increase,crop={w}:{h},fps={fps}'
                if data.get('captions',True):
                    cap_png = work/f'cap_{i:04d}.png'
                    _caption_overlay(scenes[i], data.get('ratio','9:16'), data.get('captionStyle','bold'), cap_png)
                    fc = f'[0:v]{base_vf}[bg];[bg][1:v]overlay=0:0{fade_chain}[v]'
                    cmd=['ffmpeg','-y','-hide_banner','-loglevel','error','-stream_loop','-1','-i',str(src),
                         '-i',str(cap_png),'-filter_complex',fc,'-map','[v]','-frames:v',str(frames),
                         '-an','-c:v','libx264','-preset','ultrafast','-crf','24','-pix_fmt','yuv420p',str(path)]
                else:
                    cmd=['ffmpeg','-y','-hide_banner','-loglevel','error','-stream_loop','-1','-i',str(src),
                         '-vf',base_vf+fade_chain,
                         '-frames:v',str(frames),'-an','-c:v','libx264','-preset','ultrafast','-crf','24',
                         '-pix_fmt','yuv420p',str(path)]
                p=subprocess.run(cmd,capture_output=True,text=True,timeout=1800)
                if p.returncode: raise RuntimeError(f'Scene {i+1} Veo clip error: '+p.stderr[-1000:])
                segments.append(path)
                update(job, 60+int(30*(i+1)/len(scenes)),f'Rendered scene {i+1}/{len(scenes)} (Veo clip)')
                continue
            img=work/f'frame_{i:04d}.jpg'
            fade = transition=='fade' or (transition=='mixed' and i%2==1)
            motion_mode = data.get('motion', 'off')
            if motion_mode == 'on':
                motion_mode = 'pan'
            vf=[]
            if motion_mode in ('zoomin', 'zoomout', 'pan'):
                # Ken Burns motion via zoompan on the still frame (tested expressions).
                if motion_mode == 'zoomin':
                    zp = f"zoompan=z='1+0.0012*on':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d={frames}:s={w}x{h}:fps={fps}"
                elif motion_mode == 'zoomout':
                    zp = f"zoompan=z='max(1.05,1.25-0.0012*on)':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d={frames}:s={w}x{h}:fps={fps}"
                else:
                    zp = f"zoompan=z='1.18':x='(iw-iw/zoom)*on/{max(1,frames-1)}':y='ih/2-(ih/zoom/2)':d={frames}:s={w}x{h}:fps={fps}"
                vf += [zp]
            else:
                vf += [f'scale={w}:{h}']
            if fade and duration>0.8:
                fd=min(.28,duration/5)
                vf += [f'fade=t=in:st=0:d={fd:.3f}',f'fade=t=out:st={max(0,duration-fd):.3f}:d={fd:.3f}']
            cmd=['ffmpeg','-y','-hide_banner','-loglevel','error','-loop','1','-framerate',str(fps),
                 '-i',str(img),'-vf',','.join(vf),'-frames:v',str(frames),'-an','-c:v','libx264',
                 '-preset','ultrafast','-crf','24','-pix_fmt','yuv420p',str(path)]
            p=subprocess.run(cmd,capture_output=True,text=True,timeout=1800)
            if p.returncode: raise RuntimeError(f'Scene {i+1} FFmpeg error: '+p.stderr[-1200:])
            segments.append(path)
            update(job, 60+int(30*(i+1)/len(scenes)),f'Rendered scene {i+1}/{len(scenes)}')
        concat=work/'clips.txt'
        with concat.open('w',encoding='utf-8') as f:
            for seg in segments: f.write(f"file '{seg.name}'\n")
        result=OUT/(job+'.mp4')
        if data.get('musicPath'):
            cmd=['ffmpeg','-y','-hide_banner','-loglevel','error','-f','concat','-safe','0',
                 '-i',str(concat),'-i',str(audio),'-stream_loop','-1','-i',str(data['musicPath']),
                 '-filter_complex','[1:a]volume=1.0[a1];[2:a]volume=0.16[a2];[a1][a2]amix=inputs=2:duration=first[aout]',
                 '-map','0:v:0','-map','[aout]','-c:v','copy','-c:a','aac','-b:a','128k',
                 '-shortest','-movflags','+faststart',str(result)]
        else:
            cmd=['ffmpeg','-y','-hide_banner','-loglevel','error','-f','concat','-safe','0',
                 '-i',str(concat),'-i',str(audio),'-map','0:v:0','-map','1:a:0',
                 '-c:v','copy','-c:a','aac','-b:a','128k','-shortest','-movflags','+faststart',str(result)]
        p=subprocess.run(cmd,capture_output=True,text=True,timeout=1800)
        if p.returncode: raise RuntimeError('Final FFmpeg error: '+p.stderr[-1200:])
        finish_msg = f'Finished! Narration: {total_duration:.1f}s. Download MP4.'
        if placeholder_scenes:
            finish_msg += (f' NOTE: {len(placeholder_scenes)} scene(s) had FREE images that could not be created '
                           f'(scene {", ".join(str(n) for n in placeholder_scenes)}) - a card was used for them. '
                           'Please Launch again for those scenes, or try Cloudflare FREE mode.')
        update(job,100,finish_msg,'done','/generated/'+result.name)
    except Exception as exc:
        message=str(exc)
        if data.get('apiKey'): message=message.replace(data['apiKey'],'[REDACTED]')
        if data.get('cfToken'): message=message.replace(data['cfToken'],'[REDACTED]')
        if data.get('pixabayKey'): message=message.replace(data['pixabayKey'],'[REDACTED]')
        if data.get('pexelsKey'): message=message.replace(data['pexelsKey'],'[REDACTED]')
        update(job,0,message,'error')


@app.get('/')
def index():
    return render_template_string(HTML)


# --- Connect Gemini: saved accounts (NO vault, NO password) ------------------
# NOTE: this website build is for the owner's own password-gated site (the
# whole site already sits behind the studio login), so the user's own Gemini
# API keys are saved directly in plain JSON in the app's data folder, exactly
# like the Flow profiles. There is deliberately no vault password / unlock
# step. Keys are never returned by the API (only names + models), never
# written to job logs, and are redacted from error messages.
ACCOUNTS_FILE = ROOT / 'gemini_accounts.json'
ACCOUNTS_LOCK = threading.RLock()


def _load_accounts():
    try:
        data = json.loads(ACCOUNTS_FILE.read_text(encoding='utf-8'))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _save_accounts(accounts):
    temp = ACCOUNTS_FILE.with_suffix('.tmp')
    temp.write_text(json.dumps(accounts, indent=2), encoding='utf-8')
    os.replace(temp, ACCOUNTS_FILE)
    if os.name != 'nt':
        try:
            os.chmod(ACCOUNTS_FILE, 0o600)
        except OSError:
            pass


@app.get('/api/accounts')
def account_list():
    accounts = _load_accounts()
    return jsonify(accounts=[{'name': n, 'model': a.get('model', '')} for n, a in accounts.items()])


@app.post('/api/accounts/save')
def account_save():
    if not _same_origin(): return jsonify(error='Cross-origin request blocked.'),403
    d = request.get_json(silent=True) or {}
    name = str(d.get('name', '')).strip()
    key = str(d.get('key', '')).strip()
    model = _normalize_image_model(d.get('model', DEFAULT_IMAGE_MODEL))
    with ACCOUNTS_LOCK:
        accounts = _load_accounts()
        existing = accounts.get(name)
        if not key and existing:
            key = existing['key']  # model-only update: keep the already-saved key
        if not name or len(name) > 70 or not key or len(key) > 512 or not re.fullmatch(r'[A-Za-z0-9_.-]{3,100}', model):
            return jsonify(error='Invalid account name, API key or model. An API key is required for a new account.'),400
        accounts[name] = {'key': key, 'model': model}
        _save_accounts(accounts)
    return jsonify(ok=True)


@app.post('/api/accounts/delete')
def account_delete():
    if not _same_origin(): return jsonify(error='Cross-origin request blocked.'),403
    name = str((request.get_json(silent=True) or {}).get('name', ''))
    with ACCOUNTS_LOCK:
        accounts = _load_accounts()
        accounts.pop(name, None)
        _save_accounts(accounts)
    return jsonify(ok=True)


@app.post('/api/accounts/test')
def account_test():
    if not _same_origin(): return jsonify(error='Cross-origin request blocked.'),403
    name = str((request.get_json(silent=True) or {}).get('name', ''))
    account = _load_accounts().get(name)
    if not account: return jsonify(error='Select a saved account.'),404
    try:
        from google import genai
        client = genai.Client(api_key=account['key'])
        wanted = _normalize_image_model(account['model'])
        model = client.models.get(model=wanted)
        note = '' if wanted == account['model'] else f' NOTE: the old saved model "{account["model"]}" will be automatically converted to the new model when images are generated "{wanted}" - it is better to update the model field and save the account again (if you leave the key blank, the existing key will be kept).'
        return jsonify(message='Gemini API key accepted and model found: ' + str(model.name) + '. NOTE: this test only checks the key/model - IMAGE generation requires billing/prepaid credits in this project, because image models are not available on the free tier.' + note)
    except Exception as exc:
        return jsonify(error='Gemini model/key test failed: ' + _friendly_gemini_error(exc)[:450]),400


@app.post('/api/launch')
def launch():
    if not _same_origin(): return jsonify(error='Cross-origin request blocked.'),403
    data = request.form.to_dict()
    if data.get('imageMode', 'pollinations') == 'gemini':
        # Connect Gemini: use the saved account's key for this job only.
        account = _load_accounts().get(str(data.get('accountName', '')).strip())
        if not account:
            return jsonify(error='Gemini mode: connect and select a saved Gemini account on the Connect Gemini page, or choose one of the free image sources.'),400
        data['apiKey'] = account['key']
        data['imageModel'] = account['model']
    if data.get('imageMode') == 'cloudflare':
        # The token stays in memory only for this job; it is not saved to disk.
        data['cfAccount']=str(data.get('cfAccount','')).strip()
        data['cfToken']=str(data.get('cfToken','')).strip()
        if not data['cfAccount'] or not data['cfToken']:
            return jsonify(error='Cloudflare FREE mode: first enter the Cloudflare Account ID and API Token in the Image Source section, or select Pollinations FREE mode (nothing is required for that mode).'),400
        if len(data['cfAccount'])>64 or len(data['cfToken'])>256:
            return jsonify(error='The Cloudflare Account ID / Token length is not valid.'),400
    if data.get('imageMode') in ('pixabay', 'pexels'):
        # Free stock-photo keys: used only for this job, never saved to disk.
        field = 'pixabayKey' if data['imageMode'] == 'pixabay' else 'pexelsKey'
        label = 'Pixabay' if data['imageMode'] == 'pixabay' else 'Pexels'
        data[field]=str(data.get(field,'')).strip()
        if not data[field]:
            return jsonify(error=f'{label} stock photos: first enter your free {label} API key in the Image Source section, or select Openverse stock photos (no key needed).'),400
        if len(data[field])>256:
            return jsonify(error=f'The {label} API key length is not valid.'),400
        # The token stays in memory only for this job; it is not saved to disk.
        data['cfAccount']=str(data.get('cfAccount','')).strip()
        data['cfToken']=str(data.get('cfToken','')).strip()
        if not data['cfAccount'] or not data['cfToken']:
            return jsonify(error='Cloudflare FREE mode: first enter the Cloudflare Account ID and API Token in the Image Source section, or select Pollinations FREE mode (nothing is required for that mode).'),400
        if len(data['cfAccount'])>64 or len(data['cfToken'])>256:
            return jsonify(error='The Cloudflare Account ID / Token length is not valid.'),400
    script = str(data.get('script','')).strip()
    if not script: return jsonify(error='Paste your script first.'),400
    if len(script)>MAX_SCRIPT_CHARS: return jsonify(error='The script is too long.'),400
    if len(data.get('characterBible','')) > 4000: return jsonify(error='Character Bible too long.'),400
    if data.get('sceneEdits'):
        try:
            edits=json.loads(data['sceneEdits'])
            if not isinstance(edits,list) or not 1<=len(edits)<=MAX_SCENES: raise ValueError()
            if any(not isinstance(e,dict) or not isinstance(e.get('narration'),str) or not isinstance(e.get('prompt'),str) or not e['narration'].strip() or len(e['narration'])>10000 or len(e['prompt'])>10000 for e in edits): raise ValueError()
            data['sceneEditsParsed']=edits
            data['script']='\n\n'.join(e['narration'] for e in edits)
        except (ValueError,TypeError):
            return jsonify(error='Invalid scene timeline; refresh scenes and retry.'),400
    data['captions'] = data.get('captions') == 'true'
    files=request.files.getlist('images')
    if len(files)>100: return jsonify(error='Maximum 100 uploaded pictures.'),400
    job=uuid.uuid4().hex
    folder=UPLOADS/job
    folder.mkdir(exist_ok=True)
    uploads=[]
    for i,file in enumerate(files):
        if not file.filename: continue
        if file.mimetype not in ('image/png','image/jpeg','image/webp'):
            return jsonify(error='Only PNG, JPG and WEBP images are supported.'),400
        p=folder/f'upload_{i:04d}.png'
        try:
            # Verify image bytes and remove metadata; cap dimensions for local memory.
            with Image.open(file.stream) as img:
                img.verify()
            file.stream.seek(0)
            with Image.open(file.stream) as img:
                im=img.convert('RGB')
                im.thumbnail((2400,2400))
                im.save(p)
        except Exception:
            return jsonify(error='One uploaded picture is invalid.'),400
        uploads.append(p)
    data['uploads']=uploads
    # Veo / Flow scene clips (optional): the first clips take the place of the first scenes
    clips = []
    for i, f in enumerate(request.files.getlist('sceneclips')):
        if not f or not f.filename:
            continue
        ext = Path(f.filename).suffix.lower()
        if ext not in ('.mp4', '.mov', '.webm', '.mkv'):
            return jsonify(error='Scene clips can only be MP4 / MOV / WEBM.'),400
        dest = folder / f'veoclip_{i:04d}{ext}'
        f.save(dest)
        if dest.stat().st_size > 200 * 1024 * 1024:
            return jsonify(error='A scene clip is too large (max 200MB).'),400
        clips.append(str(dest))
    if len(clips) > MAX_SCENES:
        return jsonify(error='There are too many clips.'),400
    data['clips'] = clips
    # Own voiceover / background music (optional uploads)
    for field, key, exts in (('voiceover', 'voiceoverPath', ('.mp3', '.wav', '.m4a', '.ogg')),
                             ('musicfile', 'musicPath', ('.mp3', '.wav', '.m4a', '.ogg'))):
        f = request.files.get(field)
        if f and f.filename:
            ext = Path(f.filename).suffix.lower()
            if ext not in exts:
                return jsonify(error='Audio can only be MP3 / WAV / M4A / OGG.'),400
            dest = folder / (key + ext)
            f.save(dest)
            if dest.stat().st_size > 60 * 1024 * 1024:
                return jsonify(error='The audio file is too large (max 60MB).'),400
            data[key] = str(dest)
    update(job,0,'Queued...')
    with LOCK:
        if job in JOBS:
            JOBS[job]['name'] = (str(data.get('projectName','')).strip()
                                 or str(data.get('script','')).strip()[:48])
            JOBS[job]['state'] = 'queued'
    WORK_Q.put((job, data))
    return jsonify(job=job)


@app.get('/api/status/<job>')
def status(job):
    with LOCK: state = JOBS.get(job)
    return jsonify(state) if state else (jsonify(error='Job not found'),404)


@app.get('/api/jobs')
def jobs_list():
    with LOCK:
        items = [{'job': j, 'state': s.get('state'), 'progress': s.get('progress'),
                  'message': s.get('message'), 'url': s.get('url'), 'name': s.get('name', ''),
                  'created': s.get('created', 0)} for j, s in JOBS.items()]
    items.sort(key=lambda x: x['created'], reverse=True)
    return jsonify(jobs=items)


@app.get('/api/videos')
def videos_list():
    out = []
    for p in OUT.glob('*.mp4'):
        st = p.stat()
        out.append({'name': p.name, 'size_mb': round(st.st_size / 1048576, 2),
                    'mtime': st.st_mtime, 'url': '/generated/' + p.name})
    out.sort(key=lambda x: x['mtime'], reverse=True)
    return jsonify(videos=out)


@app.get('/api/assets/<job>')
def assets_list(job):
    if not re.fullmatch(r'[a-f0-9]{32}', job or ''):
        return jsonify(error='Invalid job'),400
    folder = OUT / job
    files = []
    if folder.is_dir():
        for p in sorted(folder.iterdir()):
            if re.fullmatch(r'(frame|image)_\d{4}\.(jpg|png)', p.name):
                files.append({'name': p.name, 'url': f'/api/asset/{job}/{p.name}'})
    return jsonify(files=files)


@app.get('/api/asset/<job>/<name>')
def asset_file(job, name):
    if not re.fullmatch(r'[a-f0-9]{32}', job or '') or not re.fullmatch(r'(frame|image)_\d{4}\.(jpg|png)', name or ''):
        return jsonify(error='Invalid asset'),400
    return send_from_directory(OUT / job, name, as_attachment=False)


def _load_projects():
    try:
        data = json.loads(PROJECTS_FILE.read_text(encoding='utf-8'))
        return data if isinstance(data, list) else []
    except Exception:
        return []


def _save_projects(projects):
    tmp = PROJECTS_FILE.with_suffix('.tmp')
    tmp.write_text(json.dumps(projects, indent=1), encoding='utf-8')
    os.replace(tmp, PROJECTS_FILE)


@app.get('/api/projects')
def projects_list():
    with PROJECTS_LOCK:
        projects = _load_projects()
    return jsonify(projects=[{'id': p['id'], 'name': p['name'], 'updated': p.get('updated', 0),
                              'chars': len(p.get('data', {}).get('script', ''))} for p in projects])


@app.get('/api/projects/<pid>')
def project_get(pid):
    with PROJECTS_LOCK:
        projects = _load_projects()
    for p in projects:
        if p['id'] == pid:
            return jsonify(project=p)
    return jsonify(error='Project not found'),404


@app.post('/api/projects/save')
def project_save():
    if not _same_origin(): return jsonify(error='Cross-origin request blocked.'),403
    d = request.get_json(silent=True) or {}
    name = str(d.get('name', '')).strip()
    pdata = d.get('data') if isinstance(d.get('data'), dict) else {}
    if not name or len(name) > 120:
        return jsonify(error='A project name is required (max 120 characters).'),400
    if len(str(pdata.get('script', ''))) > MAX_SCRIPT_CHARS:
        return jsonify(error='The script is too long.'),400
    with PROJECTS_LOCK:
        projects = _load_projects()
        pid = str(d.get('id', '')).strip()
        found = next((p for p in projects if p['id'] == pid), None) if pid else None
        if found:
            found.update({'name': name, 'data': pdata, 'updated': time.time()})
            pid = found['id']
        else:
            pid = uuid.uuid4().hex
            projects.insert(0, {'id': pid, 'name': name, 'data': pdata, 'updated': time.time()})
        _save_projects(projects[:100])
    return jsonify(ok=True, id=pid)


@app.post('/api/projects/delete')
def project_delete():
    if not _same_origin(): return jsonify(error='Cross-origin request blocked.'),403
    pid = str((request.get_json(silent=True) or {}).get('id', ''))
    with PROJECTS_LOCK:
        projects = [p for p in _load_projects() if p['id'] != pid]
        _save_projects(projects)
    return jsonify(ok=True)


@app.post('/api/voice-preview')
def voice_preview():
    if not _same_origin(): return jsonify(error='Cross-origin request blocked.'),403
    voice = str((request.get_json(silent=True) or {}).get('voice', 'English Female'))
    resolved = _resolve_voice(voice)
    name = 'preview_' + hashlib.md5(resolved.encode()).hexdigest()[:10] + '.mp3'
    target = OUT / name
    if not target.exists():
        sample = ('Hello! This is a preview of the voice for your video.'
                  if voice.startswith(('Urdu', 'Hindi')) or resolved.startswith(('ur-', 'hi-')) else
                  'Hello! This is a preview of the voice for your video.')
        async def speak():
            await edge_tts.Communicate(sample, resolved).save(str(target))
        try:
            asyncio.run(speak())
        except Exception as exc:
            return jsonify(error='The preview could not be created (check the internet/Edge TTS): ' + str(exc)[:200]),400
    return jsonify(url='/generated/' + name)


@app.post('/api/writer')
def writer():
    if not _same_origin(): return jsonify(error='Cross-origin request blocked.'),403
    d = request.get_json(silent=True) or {}
    topic = str(d.get('topic', '')).strip()
    language = str(d.get('language', 'English')).strip()[:30] or 'English'
    if not topic or len(topic) > 500:
        return jsonify(error='First enter a topic: what the story should be about.'),400
    prompt = (f'Write a complete voiceover narration script for a short story video. Topic: {topic}. '
              f'Language: {language}. Write 6 to 10 short paragraphs, one scene per paragraph, separated by a blank line. '
              'Simple, engaging storytelling for listeners. No title, no headings, no scene numbers, no notes - only the narration text.')
    url = 'https://text.pollinations.ai/' + urllib.parse.quote(prompt, safe='')
    last = None
    for _ in range(2):
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'AI-Video-Studio-Pro/1.0'})
            with urllib.request.urlopen(req, timeout=90) as resp:
                text = resp.read().decode('utf-8', 'replace').strip()
            if len(text) >= 80:
                return jsonify(script=text)
            last = 'The response was too short'
        except Exception as exc:
            last = exc
    return jsonify(error='The FREE AI Writer could not respond right now (the free service may be busy). Please try again in a little while. ' + str(last)[:150]),400


@app.get('/api/verify')
def verify():
    checks = []
    checks.append({'name': 'FFmpeg', 'ok': bool(shutil.which('ffmpeg')), 'detail': shutil.which('ffmpeg') or 'not found - please install FFmpeg'})
    checks.append({'name': 'FFprobe', 'ok': bool(shutil.which('ffprobe')), 'detail': shutil.which('ffprobe') or 'not found'})
    try:
        probe = OUT / '.write_test'
        probe.write_text('ok')
        probe.unlink()
        checks.append({'name': 'Generated folder is writable', 'ok': True, 'detail': str(OUT)})
    except Exception as exc:
        checks.append({'name': 'Generated folder is writable', 'ok': False, 'detail': str(exc)[:120]})
    try:
        req = urllib.request.Request('https://image.pollinations.ai/models', headers={'User-Agent': 'AI-Video-Studio-Pro/1.0'})
        with urllib.request.urlopen(req, timeout=12) as resp:
            checks.append({'name': 'Pollinations FREE service (internet)', 'ok': resp.status == 200, 'detail': f'HTTP {resp.status}'})
    except Exception as exc:
        checks.append({'name': 'Pollinations FREE service (internet)', 'ok': False, 'detail': str(exc)[:120]})
    return jsonify(checks=checks)


@app.get('/api/voices')
def voices_catalog():
    """The COMPLETE free Edge TTS catalog (hundreds of voices) - downloaded from the internet the first time
    and cached, then available instantly afterwards."""
    try:
        if VOICES_CACHE_FILE.exists() and time.time() - VOICES_CACHE_FILE.stat().st_mtime < 7 * 86400:
            cached = json.loads(VOICES_CACHE_FILE.read_text(encoding='utf-8'))
            if isinstance(cached, list) and cached:
                return jsonify(voices=cached, cached=True)
    except Exception:
        pass
    try:
        async def fetch():
            return await edge_tts.list_voices()
        raw = asyncio.run(fetch())
        out = [{'short': v.get('ShortName', ''), 'locale': v.get('Locale', ''),
                'gender': v.get('Gender', ''), 'name': v.get('FriendlyName', v.get('ShortName', ''))}
               for v in raw if v.get('ShortName')]
        out.sort(key=lambda x: (x['locale'], x['gender'], x['short']))
        VOICES_CACHE_FILE.write_text(json.dumps(out), encoding='utf-8')
        return jsonify(voices=out, cached=False)
    except Exception as exc:
        fallback = [{'short': code, 'locale': code.split('-')[0] + '-' + code.split('-')[1],
                     'gender': '', 'name': label} for label, code in VOICES.items()]
        return jsonify(voices=fallback, cached=True, note='The catalog could not be downloaded: ' + str(exc)[:120])


def _load_flow_profiles():
    try:
        data = json.loads(FLOW_PROFILES_FILE.read_text(encoding='utf-8'))
        return data if isinstance(data, list) else []
    except Exception:
        return []


@app.get('/api/flow/profiles')
def flow_profiles_list():
    return jsonify(profiles=_load_flow_profiles())


@app.post('/api/flow/profiles/save')
def flow_profile_save():
    if not _same_origin(): return jsonify(error='Cross-origin request blocked.'),403
    d = request.get_json(silent=True) or {}
    name = str(d.get('name', '')).strip()
    note = str(d.get('note', '')).strip()[:200]
    if not name or len(name) > 60:
        return jsonify(error='A profile name is required (for example, acc1 - my Gmail).'),400
    profiles = _load_flow_profiles()
    pid = str(d.get('id', '')).strip()
    found = next((p for p in profiles if p['id'] == pid), None) if pid else None
    if found:
        found.update({'name': name, 'note': note})
        pid = found['id']
    else:
        pid = uuid.uuid4().hex
        profiles.append({'id': pid, 'name': name, 'note': note})
    FLOW_PROFILES_FILE.write_text(json.dumps(profiles[:20], indent=1), encoding='utf-8')
    return jsonify(ok=True, id=pid)


@app.post('/api/flow/profiles/delete')
def flow_profile_delete():
    if not _same_origin(): return jsonify(error='Cross-origin request blocked.'),403
    pid = str((request.get_json(silent=True) or {}).get('id', ''))
    profiles = [p for p in _load_flow_profiles() if p['id'] != pid]
    FLOW_PROFILES_FILE.write_text(json.dumps(profiles, indent=1), encoding='utf-8')
    return jsonify(ok=True)


@app.post('/api/flow/open')
def flow_open():
    """Opens Google Flow in the browser on this PC (the user signs in themselves)."""
    if not _same_origin(): return jsonify(error='Cross-origin request blocked.'),403
    import webbrowser
    try:
        webbrowser.open('https://labs.google/fx/tools/flow')
        return jsonify(ok=True)
    except Exception as exc:
        return jsonify(error='The browser could not be opened: ' + str(exc)[:150]),400


@app.get('/generated/<path:name>')
def generated(name):
    if '/' in name or '\\' in name or not name.endswith(('.mp4', '.mp3')):
        return jsonify(error='Invalid file'),400
    return send_from_directory(OUT,name,as_attachment=False)


if __name__ == '__main__':
    if LAN_MODE and len(LAN_PASSWORD) < 12:
        raise SystemExit('Mobile/LAN mode requires STUDIO_ACCESS_PASSWORD with at least 12 characters.')
    app.run(host='0.0.0.0' if LAN_MODE else '127.0.0.1',port=5001,debug=False,threaded=True)
