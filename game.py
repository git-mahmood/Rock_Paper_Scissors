import cv2
import mediapipe as mp
import time
import os
import urllib.request
import numpy as np
import collections
import math
import threading

# --- Download Model ---
MODEL_PATH = "hand_landmarker.task"
if not os.path.exists(MODEL_PATH):
    print("Downloading hand model...")
    url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
    urllib.request.urlretrieve(url, MODEL_PATH)
    print("Done!")

BaseOptions           = mp.tasks.BaseOptions
HandLandmarker        = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode     = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.IMAGE,
    num_hands=1,
    min_hand_detection_confidence=0.15,
    min_hand_presence_confidence=0.15,
    min_tracking_confidence=0.15
)

WINS = {"Rock": "Paper", "Paper": "Scissors", "Scissors": "Rock"}
CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17)
]

# =============================================
# THREADING — camera never waits for detection
# =============================================
latest_frame      = None
latest_landmarks  = None
latest_wrist_y    = None   # updated inside thread too
frame_lock        = threading.Lock()
landmark_lock     = threading.Lock()
detection_running = True

def detection_thread():
    global latest_landmarks, latest_wrist_y, detection_running
    with HandLandmarker.create_from_options(options) as detector:
        while detection_running:
            with frame_lock:
                f = latest_frame
            if f is None:
                time.sleep(0.003)
                continue
            try:
                rgb    = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
                mp_img = mp.Image(
                    image_format=mp.ImageFormat.SRGB, data=rgb)
                res = detector.detect(mp_img)
                lms = res.hand_landmarks[0] \
                      if res.hand_landmarks else None
                with landmark_lock:
                    latest_landmarks = lms
                    if lms:
                        latest_wrist_y = lms[0].y * f.shape[0]
                    else:
                        latest_wrist_y = None
            except:
                pass

# =============================================
# IMPROVED GESTURE — handles ANY angle/position
# Specifically fixed for side-view scissors
# =============================================
def dist3d(a, b):
    return math.sqrt(
        (a.x-b.x)**2 +
        (a.y-b.y)**2 +
        (a.z-b.z)**2
    )

def finger_angle(a, b, c):
    """Angle at joint b between points a-b-c"""
    v1 = np.array([a.x-b.x, a.y-b.y, a.z-b.z])
    v2 = np.array([c.x-b.x, c.y-b.y, c.z-b.z])
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return 180.0
    cos = np.dot(v1,v2) / (n1*n2)
    return math.degrees(math.acos(np.clip(cos,-1,1)))

def is_finger_extended(lms, tip, pip, mcp):
    """
    True if finger is extended.
    Uses BOTH angle AND distance — works at any angle.
    """
    # Method 1: joint angle — straight finger > 150 deg
    ang = finger_angle(lms[mcp], lms[pip], lms[tip])
    angle_extended = ang > 150

    # Method 2: tip distance from wrist vs pip distance
    palm = dist3d(lms[0], lms[9])
    if palm < 0.001:
        return angle_extended
    tip_dist = dist3d(lms[tip], lms[0]) / palm
    pip_dist = dist3d(lms[pip], lms[0]) / palm
    dist_extended = tip_dist > pip_dist * 1.2

    # Method 3: z-depth — extended finger tip is closer to cam
    z_extended = (lms[mcp].z - lms[tip].z) > 0.02

    # Vote — 2 of 3 must agree
    votes = sum([angle_extended, dist_extended, z_extended])
    return votes >= 2

def detect_gesture(lms):
    # Check each finger independently
    index  = is_finger_extended(lms,  8,  6,  5)
    middle = is_finger_extended(lms, 12, 10,  9)
    ring   = is_finger_extended(lms, 16, 14, 13)
    pinky  = is_finger_extended(lms, 20, 18, 17)

    open_count = sum([index, middle, ring, pinky])

    # Scissors: ONLY index + middle up
    # Extra check: ring and pinky must be clearly down
    if index and middle and not ring and not pinky:
        return "Scissors"

    # Paper: 3 or 4 fingers extended
    if open_count >= 3:
        return "Paper"

    # Rock: all closed
    if open_count == 0:
        return "Rock"

    # 1 finger only = still treat as rock (thumb variations)
    if open_count == 1:
        return "Rock"

    # 2 fingers but not index+middle = paper attempt
    if open_count == 2:
        if index and middle:
            return "Scissors"
        return "Paper"

    return "Rock"

# =============================================
# GAME STATE
# =============================================
score_player    = 0
score_ai        = 0
result_text     = ""
ai_choice       = ""
player_gesture  = "..."
final_player    = ""

# Shake detection — fixed timing
wrist_y_history  = collections.deque(maxlen=25)
shake_count      = 0
last_peak_time   = 0
last_wrist_y     = None
wrist_direction  = 0
stillness_start  = None

# Tuned constants — fixes jumping shake count
MIN_SHAKE_SPAN  = 18    # need bigger movement to count
MIN_PEAK_GAP    = 0.30  # 300ms between peaks — prevents double count
SHOOT_STILLNESS = 8     # pixel stillness threshold
SHOOT_PAUSE     = 0.08  # 80ms still = shoot

game_phase  = "waiting"
phase_start = 0

def get_result(player, ai):
    if player == ai:       return "Draw!"
    if WINS[player] == ai: return "AI Wins!"
    return "You Win!"

# =============================================
# UI HELPERS
# =============================================
def rounded_rect(frame, x, y, w, h, r, color, thick=-1):
    cv2.rectangle(frame,(x+r,y),(x+w-r,y+h),color,thick)
    cv2.rectangle(frame,(x,y+r),(x+w,y+h-r),color,thick)
    for cx2,cy2 in [(x+r,y+r),(x+w-r,y+r),
                    (x+r,y+h-r),(x+w-r,y+h-r)]:
        cv2.circle(frame,(cx2,cy2),r,color,thick)

def draw_centered(frame, text, cx, y, scale, color, thick=2):
    (tw,_),_ = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    cv2.putText(frame, text, (cx-tw//2, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, thick, cv2.LINE_AA)

def draw_pill(frame, text, cx, cy,
              bg=(30,30,50), fg=(255,255,255)):
    (tw,th),_ = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
    pad=14; h=th+16
    x=cx-tw//2-pad; y=cy-h//2
    rounded_rect(frame,x,y,tw+pad*2,h,h//2,bg,-1)
    cv2.putText(frame, text, (x+pad, cy+th//2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65, fg, 2, cv2.LINE_AA)

def draw_shake_dots(frame, count, cx, cy):
    labels   = ["Rock","Paper","Scissors"]
    cols_on  = [(100,200,255),(100,255,150),(255,180,80)]
    cols_off = [(35,35,55),(35,35,55),(35,35,55)]
    for i in range(3):
        x   = cx - 110 + i*110
        col = cols_on[i]  if i < count else cols_off[i]
        bdr = cols_on[i]  if i < count else (60,60,80)
        cv2.circle(frame,(x,cy),22,col,-1)
        cv2.circle(frame,(x,cy),22,bdr,2)
        nc = (15,15,25) if i < count else (70,70,90)
        draw_centered(frame,str(i+1),x,cy+7,0.65,nc,2)
        lc = cols_on[i] if i < count else (60,60,80)
        draw_centered(frame,labels[i],x,cy+42,0.42,lc,1)

# =============================================
# AI HAND DRAWING
# =============================================
def draw_ai_hand(frame, gesture, cx, cy, s=1.0):
    if gesture == "Rock":
        c,c2 = (70,100,240),(45,70,180)
        cv2.ellipse(frame,(cx,cy+int(10*s)),
                    (int(40*s),int(34*s)),0,0,360,c,-1)
        for kx in [cx-int(22*s),cx-int(8*s),
                   cx+int(8*s),cx+int(22*s)]:
            cv2.circle(frame,(kx,cy-int(14*s)),int(11*s),c,-1)
        cv2.ellipse(frame,(cx-int(40*s),cy+int(6*s)),
                    (int(13*s),int(10*s)),-30,0,360,c,-1)
        cv2.ellipse(frame,(cx,cy+int(10*s)),
                    (int(40*s),int(34*s)),0,0,360,c2,2)
        draw_centered(frame,"ROCK",cx,cy+int(62*s),
                      0.6*s,(150,180,255),2)
    elif gesture == "Paper":
        c,c2 = (60,190,100),(40,145,75)
        cv2.rectangle(frame,
                      (cx-int(36*s),cy),
                      (cx+int(36*s),cy+int(36*s)),c,-1)
        for fx in [cx-int(27*s),cx-int(9*s),
                   cx+int(9*s),cx+int(27*s)]:
            cv2.rectangle(frame,
                          (fx-int(9*s),cy-int(58*s)),
                          (fx+int(9*s),cy+int(6*s)),c,-1)
            cv2.ellipse(frame,(fx,cy-int(58*s)),
                        (int(9*s),int(10*s)),0,180,360,c,-1)
        cv2.ellipse(frame,(cx-int(50*s),cy+int(10*s)),
                    (int(19*s),int(11*s)),-20,0,360,c,-1)
        cv2.rectangle(frame,
                      (cx-int(36*s),cy),
                      (cx+int(36*s),cy+int(36*s)),c2,2)
        draw_centered(frame,"PAPER",cx,cy+int(62*s),
                      0.6*s,(150,255,180),2)
    elif gesture == "Scissors":
        c,c2 = (240,150,50),(185,110,30)
        cv2.rectangle(frame,
                      (cx-int(32*s),cy),
                      (cx+int(32*s),cy+int(36*s)),c,-1)
        for fx in [cx-int(14*s),cx+int(14*s)]:
            cv2.rectangle(frame,
                          (fx-int(9*s),cy-int(58*s)),
                          (fx+int(9*s),cy+int(6*s)),c,-1)
            cv2.ellipse(frame,(fx,cy-int(58*s)),
                        (int(9*s),int(10*s)),0,180,360,c,-1)
        for fx in [cx+int(28*s),cx+int(40*s)]:
            cv2.ellipse(frame,(fx,cy-int(9*s)),
                        (int(9*s),int(15*s)),0,0,360,c,-1)
        cv2.ellipse(frame,(cx-int(43*s),cy+int(10*s)),
                    (int(17*s),int(10*s)),-20,0,360,c,-1)
        cv2.rectangle(frame,
                      (cx-int(32*s),cy),
                      (cx+int(32*s),cy+int(36*s)),c2,2)
        draw_centered(frame,"SCISSORS",cx,cy+int(62*s),
                      0.5*s,(255,200,120),2)

def draw_hand(frame, lms, fw, fh):
    pts = [(int(l.x*fw),int(l.y*fh)) for l in lms]
    for a,b in CONNECTIONS:
        cv2.line(frame,pts[a],pts[b],(0,220,120),2,cv2.LINE_AA)
    for i,pt in enumerate(pts):
        col  = (255,70,70) if i in [4,8,12,16,20] else (255,200,0)
        size = 7            if i in [4,8,12,16,20] else 4
        cv2.circle(frame,pt,size,col,-1,cv2.LINE_AA)

# =============================================
# START
# =============================================
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

det_thread = threading.Thread(
    target=detection_thread, daemon=True)
det_thread.start()

print("RPS AI  |  Q=quit  |  R=reset")
print("Shake your fist 3 times then SHOOT!")

try:
    while True:
        ret, frame = cap.read()
        if not ret: break

        frame  = cv2.flip(frame,1)
        fh, fw = frame.shape[:2]
        now    = time.time()

        # Send frame to detection thread
        with frame_lock:
            latest_frame = frame.copy()

        # Get results from thread
        with landmark_lock:
            lms     = latest_landmarks
            wrist_y = latest_wrist_y

        hand_detected   = lms is not None
        current_gesture = None

        if hand_detected:
            draw_hand(frame, lms, fw, fh)
            g = detect_gesture(lms)
            if g:
                current_gesture = g
                player_gesture  = g
        else:
            player_gesture = "..."
            wrist_y        = None

        # =============================================
        # SHAKE ENGINE — fixed
        # =============================================
        if game_phase in ("waiting","shaking") \
                and hand_detected and wrist_y is not None:

            wrist_y_history.append((now, wrist_y))

            if last_wrist_y is not None:
                move    = wrist_y - last_wrist_y
                new_dir = 1 if move > 2.0 else \
                         (-1 if move < -2.0 else 0)

                if (new_dir != 0
                        and wrist_direction != 0
                        and new_dir != wrist_direction):

                    # Check span of recent movement
                    recent_y = [y for _,y in wrist_y_history]
                    span = max(recent_y) - min(recent_y) \
                           if recent_y else 0

                    if (span >= MIN_SHAKE_SPAN
                            and now - last_peak_time > MIN_PEAK_GAP):
                        shake_count    = min(shake_count+1, 3)
                        last_peak_time = now
                        game_phase     = "shaking"
                        stillness_start = None
                        # Clear history after counting shake
                        wrist_y_history.clear()

                if new_dir != 0:
                    wrist_direction = new_dir

            last_wrist_y = wrist_y

            # SHOOT — detect stillness after 3 shakes
            if shake_count >= 3 and game_phase == "shaking":
                recent_y = [y for _,y in
                            list(wrist_y_history)[-6:]]
                if len(recent_y) >= 4:
                    span = max(recent_y) - min(recent_y)
                    if span < SHOOT_STILLNESS:
                        if stillness_start is None:
                            stillness_start = now
                        elif now - stillness_start >= SHOOT_PAUSE:
                            if current_gesture:
                                final_player = current_gesture
                                ai_choice    = WINS[final_player]
                                result_text  = get_result(
                                    final_player, ai_choice)
                                if "AI Wins"   in result_text:
                                    score_ai     += 1
                                elif "You Win" in result_text:
                                    score_player += 1
                                game_phase  = "result"
                                phase_start = now
                    else:
                        stillness_start = None

        # Reset after result
        if game_phase == "result" and now - phase_start > 3.2:
            game_phase      = "waiting"
            shake_count     = 0
            stillness_start = None
            last_wrist_y    = None
            wrist_direction = 0
            wrist_y_history.clear()
            result_text     = ""

        # =============================================
        # UI
        # =============================================
        # Top bar
        ov = frame.copy()
        cv2.rectangle(ov,(0,0),(fw,72),(8,8,20),-1)
        cv2.addWeighted(ov,0.93,frame,0.07,0,frame)
        cv2.line(frame,(0,72),(fw,72),(50,50,100),1)

        draw_centered(frame,"ROCK  PAPER  SCISSORS",
                      fw//2,26,0.75,(255,215,0),2)
        draw_pill(frame,f"YOU  {score_player}",
                  fw//2-95,52,(15,55,15))
        draw_pill(frame,f"{score_ai}  AI",
                  fw//2+95,52,(55,15,15))

        # Phase UI
        if game_phase == "waiting":
            draw_centered(frame,
                          "Make a FIST and shake 3 times!",
                          fw//2, 108, 0.68,
                          (160,160,210), 1)
            draw_shake_dots(frame, 0, fw//2, 152)

        elif game_phase == "shaking":
            if shake_count < 3:
                msgs = {1:"Rock...",
                        2:"Paper...",
                        3:"Scissors..."}
                draw_centered(frame,
                              msgs.get(shake_count,"Shake!"),
                              fw//2, 108, 0.9,
                              (255,200,60), 2)
            else:
                flash = int(now*5) % 2 == 0
                col   = (80,255,80) if flash else (30,160,30)
                draw_centered(frame,">>  SHOOT!  <<",
                              fw//2, 108, 1.1, col, 3)
            draw_shake_dots(frame, shake_count, fw//2, 152)

        # Detected gesture badge
        if hand_detected and game_phase != "result":
            g_col = (100,230,120) if player_gesture=="Rock"     else \
                    (100,190,255) if player_gesture=="Paper"    else \
                    (255,170,80)  if player_gesture=="Scissors" else \
                    (120,120,150)
            draw_pill(frame,
                      f"Detected: {player_gesture}",
                      160, fh-28,
                      (12,12,28), g_col)

        # =============================================
        # RESULT PANEL
        # =============================================
        if game_phase == "result":
            ph  = 250
            py  = fh - ph
            ov2 = frame.copy()
            cv2.rectangle(ov2,(0,py),(fw,fh),(8,8,20),-1)
            cv2.addWeighted(ov2,0.93,frame,0.07,0,frame)
            cv2.line(frame,(0,py),(fw,py),(50,50,100),1)

            mid = fw//2

            # Player
            draw_centered(frame,"YOU",
                          mid-165,py+30,0.8,
                          (100,200,255),2)
            p   = final_player
            pc  = (100,200,255)
            pcx = mid-165
            pcy = py+85

            if p=="Rock":
                cv2.ellipse(frame,(pcx,pcy+10),
                            (44,36),0,0,360,pc,3)
                for kx in [pcx-19,pcx-7,pcx+7,pcx+19]:
                    cv2.circle(frame,(kx,pcy-11),7,pc,2)
            elif p=="Paper":
                cv2.rectangle(frame,
                              (pcx-40,pcy),
                              (pcx+40,pcy+38),pc,3)
                for fx in [pcx-25,pcx-9,pcx+9,pcx+25]:
                    cv2.rectangle(frame,
                                  (fx-8,pcy-50),
                                  (fx+8,pcy+4),pc,2)
            elif p=="Scissors":
                cv2.rectangle(frame,
                              (pcx-34,pcy),
                              (pcx+34,pcy+36),pc,3)
                for fx in [pcx-13,pcx+13]:
                    cv2.rectangle(frame,
                                  (fx-8,pcy-50),
                                  (fx+8,pcy+4),pc,2)
                for fx in [pcx+28,pcx+40]:
                    cv2.ellipse(frame,(fx,pcy-7),
                                (8,12),0,0,360,pc,2)

            draw_centered(frame,p.upper(),
                          pcx,py+195,0.7,pc,2)

            # Divider
            cv2.line(frame,(mid,py+15),(mid,fh-15),
                     (45,45,80),1)
            draw_centered(frame,"VS",mid,
                          py+125,1.0,(150,150,200),2)

            # AI
            draw_centered(frame,"AI",
                          mid+165,py+30,0.8,
                          (255,100,100),2)
            draw_ai_hand(frame,ai_choice,
                         cx=mid+165,cy=py+68,s=1.1)

            # Result banner
            rc  = (80,255,100)  if "You Win" in result_text else \
                  (100,100,255) if "AI Wins" in result_text else \
                  (200,200,220)
            rbg = (8,45,8)      if "You Win" in result_text else \
                  (8,8,55)      if "AI Wins" in result_text else \
                  (28,28,42)
            (rw,_),_ = cv2.getTextSize(
                result_text,
                cv2.FONT_HERSHEY_SIMPLEX,1.2,3)
            rx = mid-rw//2
            rounded_rect(frame,rx-18,fh-50,
                         rw+36,40,10,rbg,-1)
            cv2.putText(frame,result_text,(rx,fh-18),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.2,rc,3,cv2.LINE_AA)

        if not hand_detected:
            draw_centered(frame,
                          "Show your hand to the camera",
                          fw//2,fh-22,0.65,
                          (200,80,80),1)

        cv2.imshow("Rock Paper Scissors AI", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): break
        elif key == ord('r'):
            score_player = score_ai = 0
            result_text  = ai_choice = ""
            final_player = ""
            game_phase   = "waiting"
            shake_count  = 0
            wrist_y_history.clear()

finally:
    detection_running = False
    cap.release()
    cv2.destroyAllWindows()
