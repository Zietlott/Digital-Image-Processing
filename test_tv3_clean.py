# -*- coding: utf-8 -*-
"""
Phat hien, theo doi bien bao tron trong video va ghi video ket qua.
Mon: Introduction to Digital Image Processing (505060) - Essay giua ky

Chi dung OpenCV, NumPy, math. Khong dung model pre-trained.
Cung mot thuat toan va cung mot bo tham so cho moi video dau vao.
Khong hardcode theo frame hay video cu the.

PIPELINE 2 LUOT (two-pass)
  Luot 1  : doc tung frame -> tim ung vien bien tron
            -> ghep ung vien giua cac frame thanh "track"
  Hau xu ly track:
            - loai track khong phai bien bao (qua ngan, khong tron, nhieu...)
            - noi track bi dut, xoa track trung nhau (unique)
            - noi suy box o frame bi mat detection + lam muot box (stable)
  Luot 2  : doc lai video -> ve box + MSSV len moi frame -> ghi MP4
            (cung kich thuoc, cung FPS, cung he mau BGR voi video goc)
"""
import math

import cv2
import numpy as np

# =============================================================================
# 0. CAU HINH CHUNG
# =============================================================================
# MSSV cac thanh vien. Ten file output va chu hien tren frame lay tu day.
STUDENT_IDS = ["52400062", "52400200", "52400248", "52400063"]

# Video dau vao (dat cung thu muc voi file code) va video ket qua duoc ghi vao cung thu muc.
INPUT_VIDEOS = ("video1.mp4", "video2.mp4")

SHOW_PREVIEW = False      # True: hien cua so xem truoc (KHONG dung phim ESC de dung giua chung)

# ---- Mau (HSV cua OpenCV: H 0..180, S 0..255, V 0..255) ----
# Mau do nam hai dau vong Hue. Dai Hue rong de chiu duoc thay doi anh sang.
RED_H_LOW, RED_H_HIGH = 5, 150
RED_S_MIN, RED_V_MIN = 70, 50
# Mau xanh bien bao: S cao va V cao de tach khoi vung xanh toi co cung Hue.
BLUE_H_MIN, BLUE_H_MAX = 98, 130
BLUE_S_MIN, BLUE_V_MIN = 135, 115

# ---- Vung quan tam (ti le theo kich thuoc frame) ----
ROI_Y_MAX = 0.55          # chi tim bien o phan tren frame

# ---- Loc hinh dang ung vien (moi frame) ----
CAND_MIN_AREA = 60
CAND_MAX_AREA = 90000
CAND_AR_RANGE = (0.60, 1.60)
# Do do tron theo elip noi tiep HCN bao: ellipse_fill = S_bao_loi / (pi/4 * w * h)
#   tron/elip ~ 1.0 | vuong/chu nhat ~ 1.27 | tam giac ~ 0.64
CAND_ELLIPSE_FILL = (0.82, 1.12)
CAND_MIN_SOLID = 0.70     # dien tich / dien tich bao loi

# ---- Tracker ----
TRK_MAX_GAP = 15          # so frame toi da duoc mat detection ma track van song
TRK_GATE_DIST = 0.8       # khoang cach tam toi da (tinh theo kich thuoc box) de ghep
TRK_GATE_SCALE = 1.6      # ti le kich thuoc toi da giua 2 frame lien tiep

# ---- Phan loai track (hau xu ly) ----
# Moi nguong deu ap len TRUNG VI cua ca track, khong ap len 1 frame rieng le.
# "near" = 30% quan sat co kich thuoc LON nhat (luc doi tuong o gan, ro net).
FRAG_MIN_OBS = 4
TRACK_MIN_OBS = 15
TRACK_MIN_DENSITY = 0.40
TRACK_X_MIN = 0.33        # tam track (trung vi) o phan giua/phai
TRACK_Y_MAX = 0.50        # tam track (trung vi) o nua tren frame
TRACK_NEAR_CIRC = 0.90    # do tron bao loi luc gan
TRACK_MIN_SIZE = 18       # duong kinh toi thieu (px) it nhat mot luc trong track
# Nguong cho ung vien vien do:
RED_NEAR_EFILL = (0.92, 1.06)
RED_NEAR_AR = (0.65, 1.15)
RED_INNER_MIN = 0.25      # long bien co mau khac mau vien
RED_INNER_WB_MIN = 0.20   # long bien co vung trang hoac xanh
RED_MEANV_MIN = 100       # long bien du sang
RED_OUTER_DARK_MAX = 0.10 # loai vung xung quanh rat toi
RED_OUTER_BLUE_MAX = 0.30 # loai vung xung quanh toan xanh
RED_OUTER_RED_MAX = 0.15  # loai vung xung quanh van do
# Nguong cho ung vien mau xanh / dia xanh co vien do:
BLUE_MEANV_MIN = 120
BLUE_PURE_BORDER = 0.80
BLUE_PURE_HUE = 103
BLUE_RING_RED = 0.10
BLUE_NEAR_EFILL = (0.84, 1.02)
BLUE_NEAR_AR = (0.55, 1.40)
MERGE_MAX_GAP = 30        # noi 2 manh neu cach nhau <= 30 frame
MERGE_STATIC_GAP = 150    # hoac <= 150 frame neu vi tri gan nhu dung yen
MERGE_OVERLAP = 10
MERGE_SCALE = 1.8         # ti le kich thuoc toi da khi noi manh
MEDIAN_WIN = 7            # cua so loc trung vi box (so le)
SMOOTH_WIN = 5            # cua so trung binh truot box (so le)

# ---- Ve ----
BOX_COLOR = (0, 255, 0)


# =============================================================================
# 1. TIEN XU LY + PHAT HIEN UNG VIEN HINH TRON
# =============================================================================
def build_color_masks(hsv):
    """Tra ve 2 mask nhi phan (0/255): mau do va mau xanh bien bao."""
    red = cv2.inRange(hsv, (0, RED_S_MIN, RED_V_MIN), (RED_H_LOW, 255, 255))
    red |= cv2.inRange(hsv, (RED_H_HIGH, RED_S_MIN, RED_V_MIN), (180, 255, 255))
    blue = cv2.inRange(hsv, (BLUE_H_MIN, BLUE_S_MIN, BLUE_V_MIN),
                       (BLUE_H_MAX, 255, 255))
    return red, blue


def ellipse_features(hsv, red, blue, cx, cy, a, b):
    """
    Dac trung noi dung cua ung vien elip tam (cx, cy), ban truc a (ngang), b (doc).
    Toa do chuan hoa d = sqrt(((x-cx)/a)^2 + ((y-cy)/b)^2): d=1 la mep bien,
    dung duoc cho ca bien nhin thang (tron) lan nhin xien (elip).
      border_red/blue : ti le pixel do/xanh tren vanh d = 0.70..0.95
      inner_other     : ti le pixel khac mau vien trong long d <= 0.55
      mean_v          : do sang trung binh trong long bien
      outer_dark      : ti le pixel rat toi o vanh ngoai d = 1.2..1.6
      outer_blue      : ti le pixel xanh o vanh ngoai
      outer_red       : ti le pixel do o vanh ngoai
      ring_red        : ti le pixel do o vanh d = 1.05..1.35
    """
    h, w = red.shape
    rx, ry = int(math.ceil(a * 1.65)) + 2, int(math.ceil(b * 1.65)) + 2
    x0, y0 = max(0, int(cx) - rx), max(0, int(cy) - ry)
    x1, y1 = min(w, int(cx) + rx + 1), min(h, int(cy) + ry + 1)
    yy, xx = np.mgrid[y0:y1, x0:x1]
    d = np.sqrt(((xx - cx) / max(a, 1.0)) ** 2 + ((yy - cy) / max(b, 1.0)) ** 2)

    rr = red[y0:y1, x0:x1] > 0
    bb = blue[y0:y1, x0:x1] > 0
    v = hsv[y0:y1, x0:x1, 2]

    border = (d >= 0.70) & (d <= 0.95)
    inner = d <= 0.55
    outer = (d >= 1.20) & (d <= 1.60)
    ring = (d >= 1.05) & (d <= 1.35)
    nb, ni = max(1, int(border.sum())), max(1, int(inner.sum()))
    no, nr = max(1, int(outer.sum())), max(1, int(ring.sum()))

    border_red = float((rr & border).sum()) / nb
    border_blue = float((bb & border).sum()) / nb
    dom = rr if border_red >= border_blue else bb
    s_ch = hsv[y0:y1, x0:x1, 1]
    h_ch = hsv[y0:y1, x0:x1, 0]
    disk = (d <= 0.95) & dom
    white = (s_ch < 90) & (v > 110)
    return {
        "hue_med": float(np.median(h_ch[disk])) if disk.any() else 0.0,
        "inner_wb": float(((white | bb) & inner).sum()) / ni,
        "border_red": border_red,
        "border_blue": border_blue,
        "inner_other": float((~dom & inner).sum()) / ni,
        "mean_v": float(v[inner].mean()) if inner.any() else 0.0,
        "outer_dark": float(((v < 60) & outer).sum()) / no,
        "outer_blue": float((bb & outer).sum()) / no,
        "outer_red": float((rr & outer).sum()) / no,
        "ring_red": float((rr & ring).sum()) / nr,
    }


def iou_min(a, b):
    """Giao / dien tich box nho hon. =1 khi box nho nam tron trong box lon."""
    xa, ya = max(a[0], b[0]), max(a[1], b[1])
    xb, yb = min(a[0] + a[2], b[0] + b[2]), min(a[1] + a[3], b[1] + b[3])
    inter = max(0, xb - xa) * max(0, yb - ya)
    return inter / float(min(a[2] * a[3], b[2] * b[3]) + 1e-6)


def detect_candidates(frame):
    """
    Tra ve danh sach ung vien bien tron/elip trong 1 frame.
    Moi ung vien: {'box': (x, y, w, h), 'cx', 'cy', 'r', 'efill', ... dac trung}
    """
    fh, fw = frame.shape[:2]
    y_lim = int(fh * ROI_Y_MAX)
    roi = frame[:y_lim]

    blurred = cv2.GaussianBlur(roi, (5, 5), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    red, blue = build_color_masks(hsv)
    k_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    k_big = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))

    cands = []
    # 4 mask sau phep dong hinh thai: do, xanh, do|xanh, va do voi kernel lon
    # (kernel lon noi cac manh bi dut do vach trang cat ngang vien).
    for mask, kernel in ((red, k_small), (blue, k_small),
                         (cv2.bitwise_or(red, blue), k_small), (red, k_big)):
        m = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if not (CAND_MIN_AREA <= area <= CAND_MAX_AREA):
                continue
            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            if hull_area <= 0:
                continue
            x, y, w, h = cv2.boundingRect(hull)
            ar = w / float(h)
            if not (CAND_AR_RANGE[0] <= ar <= CAND_AR_RANGE[1]):
                continue
            efill = hull_area / (math.pi / 4.0 * w * h)
            per = cv2.arcLength(hull, True)
            circ = 4.0 * math.pi * hull_area / (per * per + 1e-6)
            solid = area / hull_area
            if not (CAND_ELLIPSE_FILL[0] <= efill <= CAND_ELLIPSE_FILL[1]):
                continue
            if solid < CAND_MIN_SOLID:
                continue

            cx, cy, a, b = x + w / 2.0, y + h / 2.0, w / 2.0, h / 2.0
            f = ellipse_features(hsv, red, blue, cx, cy, a, b)
            # Neu dia xanh nam trong vien do mong: no box ra de bao ca vien.
            if f["border_blue"] > f["border_red"] and f["ring_red"] >= 0.35:
                a, b = a * 1.22, b * 1.22
            bx, by = max(0, int(round(cx - a))), max(0, int(round(cy - b)))
            bw = min(int(round(2 * a)), fw - bx)
            bh = min(int(round(2 * b)), y_lim - by)
            if bw < 6 or bh < 6:
                continue
            f.update(box=(bx, by, bw, bh), cx=cx, cy=cy, r=math.sqrt(a * b),
                     efill=efill, solid=solid, ar=ar, circ=circ)
            cands.append(f)

    # Bo khoi dinh: 1 ung vien lon chua >= 2 ung vien nho rieng biet ben trong.
    def _area(c):
        return c["box"][2] * c["box"][3]
    merged_blobs = set()
    for i, c in enumerate(cands):
        inside = [o for o in cands if o is not c and _area(o) < 0.7 * _area(c)
                  and iou_min(o["box"], c["box"]) >= 0.8]
        if any(iou_min(p["box"], q["box"]) < 0.2
               for k, p in enumerate(inside) for q in inside[k + 1:]):
            merged_blobs.add(i)
    cands = [c for i, c in enumerate(cands) if i not in merged_blobs]

    # Gop ung vien trung nhau: giu ung vien lon nhat.
    cands.sort(key=_area, reverse=True)
    kept = []
    for c in cands:
        if all(iou_min(c["box"], k["box"]) < 0.5 for k in kept):
            kept.append(c)
    return kept


# =============================================================================
# 2. TRACKER: ON DINH (stable) + DUY NHAT (unique)
# =============================================================================
def _box_to_cs(b):
    """(x, y, w, h) -> (cx, cy, s) voi s = kich thuoc dac trung."""
    return b[0] + b[2] / 2.0, b[1] + b[3] / 2.0, math.sqrt(b[2] * b[3])


class Track:
    """1 doi tuong duoc theo doi qua nhieu frame."""
    FIT_LAST = 10

    def __init__(self, tid, frame_idx, det):
        self.id = tid
        self.obs = [(frame_idx, det)]

    @property
    def last_frame(self):
        return self.obs[-1][0]

    def predict(self, frame_idx):
        """Du doan (cx, cy, s) o frame_idx bang hoi quy tuyen tinh tren FIT_LAST quan sat cuoi."""
        pts = self.obs[-self.FIT_LAST:]
        f = np.array([p[0] for p in pts], dtype=np.float64)
        v = np.array([_box_to_cs(p[1]["box"]) for p in pts], dtype=np.float64)
        s_med = float(np.median(v[-5:, 2]))
        if len(pts) < 3 or f[-1] - f[0] < 2:
            return v[-1, 0], v[-1, 1], s_med
        out = []
        for k in range(3):
            slope, icpt = np.polyfit(f, v[:, k], 1)
            out.append(slope * frame_idx + icpt)
        s_pred = min(max(out[2], 0.7 * s_med), 1.5 * s_med)
        return out[0], out[1], max(4.0, s_pred)

    def add(self, frame_idx, det):
        self.obs.append((frame_idx, det))


class SignTracker:
    """
    Online: ghep ung vien moi frame vao track bang du doan chuyen dong + gating.
    Offline (finalize): phan loai track, noi track dut, xoa trung, noi suy, lam muot.
    """

    def __init__(self, frame_w, frame_h):
        self.fw, self.fh = frame_w, frame_h
        self.active = []
        self.finished = []
        self.next_id = 1

    def _leaving_frame(self, t, frame_idx):
        """Doi tuong da troi ra khoi khung hinh -> ket thuc track."""
        cx, cy, s = t.predict(frame_idx)
        m = 0.2 * s
        return cx + m > self.fw or cx - m < 0 or cy - m < 0

    # ---------------- online ----------------
    def update(self, frame_idx, detections):
        still = []
        for t in self.active:
            (self.finished if self._leaving_frame(t, frame_idx) else still).append(t)
        self.active = still
        pairs = []
        for ti, t in enumerate(self.active):
            pcx, pcy, ps = t.predict(frame_idx)
            gap = frame_idx - t.last_frame
            for di, d in enumerate(detections):
                cx, cy, s = _box_to_cs(d["box"])
                dist = math.hypot(cx - pcx, cy - pcy) / max(ps, 8.0)
                scale = max(s, ps) / max(1.0, min(s, ps))
                if dist <= TRK_GATE_DIST + 0.05 * gap and scale <= TRK_GATE_SCALE:
                    pairs.append((dist + 0.5 * (scale - 1.0), ti, di))
        pairs.sort()
        used_t, used_d = set(), set()
        for _, ti, di in pairs:
            if ti in used_t or di in used_d:
                continue
            used_t.add(ti); used_d.add(di)
            self.active[ti].add(frame_idx, detections[di])
        for di, d in enumerate(detections):
            if di not in used_d:
                self.active.append(Track(self.next_id, frame_idx, d))
                self.next_id += 1
        still = []
        for t in self.active:
            (still if frame_idx - t.last_frame <= TRK_MAX_GAP else self.finished).append(t)
        self.active = still

    # ---------------- offline ----------------
    @staticmethod
    def _is_fragment(t):
        """Manh track du dai va du day moi duoc xet tiep."""
        n = len(t.obs)
        span = t.last_frame - t.obs[0][0] + 1
        return n >= FRAG_MIN_OBS and n / float(span) >= TRACK_MIN_DENSITY

    @staticmethod
    def _is_traffic_sign(t, fw, fh):
        """Phan loai 1 track da noi: vi tri + do ben + hinh dang luc gan + mau/noi dung."""
        dets = [d for _, d in t.obs]
        n = len(dets)
        if n < TRACK_MIN_OBS:
            return False
        med = lambda k, ds=dets: float(np.median([d[k] for d in ds]))
        if med("cx") / fw < TRACK_X_MIN or med("cy") / fh > TRACK_Y_MAX:
            return False
        if 2 * max(d["r"] for d in dets) < TRACK_MIN_SIZE:
            return False
        near = sorted(dets, key=lambda d: d["r"])[int(n * 0.7):]
        efill, ar = med("efill", near), med("ar", near)
        if med("circ", near) < TRACK_NEAR_CIRC:
            return False
        if med("border_red") >= med("border_blue"):
            return (RED_NEAR_EFILL[0] <= efill <= RED_NEAR_EFILL[1]
                    and RED_NEAR_AR[0] <= ar <= RED_NEAR_AR[1]
                    and med("inner_other") >= RED_INNER_MIN
                    and med("inner_wb") >= RED_INNER_WB_MIN
                    and med("mean_v") >= RED_MEANV_MIN
                    and med("outer_dark") <= RED_OUTER_DARK_MAX
                    and med("outer_blue") <= RED_OUTER_BLUE_MAX
                    and med("outer_red") <= RED_OUTER_RED_MAX)
        if not (BLUE_NEAR_EFILL[0] <= efill <= BLUE_NEAR_EFILL[1]
                and BLUE_NEAR_AR[0] <= ar <= BLUE_NEAR_AR[1]
                and med("mean_v") >= BLUE_MEANV_MIN):
            return False
        pure = med("border_blue", near) >= BLUE_PURE_BORDER and med("hue_med") >= BLUE_PURE_HUE
        ringed = med("border_red", near) >= BLUE_RING_RED
        return pure or ringed

    @staticmethod
    def _box_at(t, f):
        """Box cua track t tai frame gan f nhat."""
        return min(t.obs, key=lambda o: abs(o[0] - f))[1]["box"]

    @classmethod
    def _same_sign(cls, a, b):
        """Manh b (bat dau sau) co phai la phan tiep theo cua manh a khong?"""
        fb = b.obs[0][0]
        gap = fb - a.last_frame
        bcx, bcy, bs = _box_to_cs(b.obs[0][1]["box"])
        if -MERGE_OVERLAP <= gap <= 0:
            acx, acy, as_ = _box_to_cs(cls._box_at(a, fb))
            return (math.hypot(bcx - acx, bcy - acy) <= 0.8 * max(as_, bs)
                    and max(bs, as_) / max(1.0, min(bs, as_)) <= MERGE_SCALE)
        if 0 < gap <= MERGE_MAX_GAP:
            pcx, pcy, ps = a.predict(fb)
            return (math.hypot(bcx - pcx, bcy - pcy) <= 1.0 * max(ps, 8.0)
                    and max(bs, ps) / max(1.0, min(bs, ps)) <= MERGE_SCALE)
        if 0 < gap <= MERGE_STATIC_GAP:
            ab = a.obs[-1][1]["box"]
            acx, acy, as_ = _box_to_cs(ab)
            return (iou_min(ab, b.obs[0][1]["box"]) >= 0.3
                    and max(bs, as_) / max(1.0, min(bs, as_)) <= 1.5)
        return False

    @classmethod
    def _merge_fragments(cls, tracks):
        """Noi cac manh cua cung mot doi tuong (bi dut do mat detection) thanh 1 track."""
        tracks = sorted(tracks, key=lambda t: t.obs[0][0])
        merged = True
        while merged:
            merged = False
            for a in tracks:
                cands = [b for b in tracks if b is not a and b.obs[0][0] > a.obs[0][0]
                         and cls._same_sign(a, b)]
                if not cands:
                    continue
                b = min(cands, key=lambda t: t.obs[0][0])
                by_f = {}
                for f, d in a.obs + b.obs:
                    if f not in by_f or d["box"][2] * d["box"][3] > by_f[f]["box"][2] * by_f[f]["box"][3]:
                        by_f[f] = d
                a.obs = sorted(by_f.items(), key=lambda o: o[0])
                tracks.remove(b)
                merged = True
                break
        return tracks

    @staticmethod
    def _harmonize_ringed(t):
        """
        Voi doi tuong co dia xanh + vien do: o xa thuong chi bat duoc dia xanh,
        o gan bat duoc ca vien -> box nhay kich thuoc. Can chinh lai ti le o phan xa
        de box on dinh tu xa den gan.
        """
        is_b = [d["border_blue"] > d["border_red"] for _, d in t.obs]
        fb = [f for (f, _), b in zip(t.obs, is_b) if b]
        fr = [f for (f, _), b in zip(t.obs, is_b) if not b]
        if len(fb) < 5 or len(fr) < 5 or np.median(fb) >= np.median(fr):
            return
        red_part = [d for (_, d), b in zip(t.obs, is_b) if not b]
        if float(np.median([d["inner_wb"] for d in red_part])) < 0.4:
            return
        last_b = [o for o, b in zip(t.obs, is_b) if b][-8:]
        first_r = [o for o, b in zip(t.obs, is_b) if not b][:8]
        def fit_at(obs, f):
            ff = np.array([o[0] for o in obs], dtype=np.float64)
            ls = np.log([_box_to_cs(o[1]["box"])[2] for o in obs])
            if len(obs) < 3 or ff[-1] - ff[0] < 2:
                return float(np.exp(np.median(ls)))
            k, c = np.polyfit(ff, ls, 1)
            return float(np.exp(k * f + c))
        fj = 0.5 * (last_b[-1][0] + first_r[0][0])
        ratio = fit_at(first_r, fj) / max(1.0, fit_at(last_b, fj))
        ratio = min(max(ratio, 1.0), MERGE_SCALE)
        new_obs = []
        for (f, d), b in zip(t.obs, is_b):
            if b:
                x, y, w, h = d["box"]
                cx, cy = x + w / 2.0, y + h / 2.0
                w2, h2 = w * ratio, h * ratio
                d = dict(d, box=(int(round(cx - w2 / 2)), int(round(cy - h2 / 2)),
                                 int(round(w2)), int(round(h2))))
            new_obs.append((f, d))
        t.obs = new_obs

    @staticmethod
    def _dense_boxes(t):
        """Box cho moi frame tu dau den cuoi track: noi suy + loc trung vi + trung binh truot."""
        f0, f1 = t.obs[0][0], t.last_frame
        known_f = np.array([f for f, _ in t.obs], dtype=np.float64)
        known = np.array([_box_to_cs(d["box"]) + (d["box"][2] / float(max(1, d["box"][3])),)
                          for _, d in t.obs], dtype=np.float64)
        frames = np.arange(f0, f1 + 1)
        dense = np.stack([np.interp(frames, known_f, known[:, k]) for k in range(4)], axis=1)
        if len(frames) >= MEDIAN_WIN:
            pad = MEDIAN_WIN // 2
            for c in range(4):
                col = np.pad(dense[:, c], pad, mode="edge")
                win = np.stack([col[i:i + len(frames)] for i in range(MEDIAN_WIN)], axis=1)
                dense[:, c] = np.median(win, axis=1)
        if len(frames) >= SMOOTH_WIN:
            k = np.ones(SMOOTH_WIN) / SMOOTH_WIN
            pad = SMOOTH_WIN // 2
            for c in range(4):
                col = np.pad(dense[:, c], pad, mode="edge")
                dense[:, c] = np.convolve(col, k, mode="valid")
        out = {}
        for f, (cx, cy, s, ar) in zip(frames, dense):
            w = s * math.sqrt(ar); h = s / math.sqrt(ar)
            out[int(f)] = (int(round(cx - w / 2)), int(round(cy - h / 2)),
                           int(round(w)), int(round(h)))
        return out

    def finalize(self, fw, fh):
        """Tra ve dict {frame_idx: [box, ...]} da on dinh + duy nhat."""
        tracks = self.finished + self.active
        self.active, self.finished = [], tracks

        frags = [t for t in tracks if self._is_fragment(t)]
        signs = [t for t in self._merge_fragments(frags) if self._is_traffic_sign(t, fw, fh)]
        for t in signs:
            self._harmonize_ringed(t)
        dense = [(t, self._dense_boxes(t)) for t in signs]

        dense.sort(key=lambda td: len(td[0].obs), reverse=True)
        kept = []
        for t, boxes in dense:
            dup = False
            for _, kb in kept:
                common = [f for f in boxes if f in kb]
                if common:
                    ov = sum(iou_min(boxes[f], kb[f]) >= 0.5 for f in common)
                    if ov >= 0.5 * len(common):
                        dup = True
                        break
            if not dup:
                kept.append((t, boxes))

        per_frame = {}
        for t, boxes in kept:
            for f, (x, y, w, h) in boxes.items():
                x, y = max(0, x), max(0, y)
                w, h = min(w, fw - x), min(h, fh - y)
                if w <= 2 or h <= 2:
                    continue
                lst = per_frame.setdefault(f, [])
                if all(iou_min((x, y, w, h), o) < 0.5 for o in lst):
                    lst.append((x, y, w, h))
        self.kept_tracks = [t for t, _ in kept]
        return per_frame


# =============================================================================
# 3. XU LY VIDEO: DOC / GHI MP4 / CHEN MSSV / KIEM TRA OUTPUT
# =============================================================================
def draw_student_ids(frame, text):
    """Chen MSSV goc tren-trai; co chu + nen tu dong theo kich thuoc frame."""
    fh = frame.shape[0]
    scale = max(0.5, fh / 1080.0 * 0.9)
    thick = max(1, int(round(fh / 1080.0 * 2)))
    (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    pad = int(10 * fh / 1080.0) + 4
    cv2.rectangle(frame, (pad, pad), (pad + tw + 2 * pad, pad + th + base + 2 * pad),
                  (0, 0, 0), -1)
    cv2.putText(frame, text, (2 * pad, pad + pad + th), cv2.FONT_HERSHEY_SIMPLEX,
                scale, (255, 255, 255), thick, cv2.LINE_AA)


def open_writer(path, fps, size):
    """Mo VideoWriter MP4 (codec mp4v). Kiem tra isOpened() de khong ghi file rong."""
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, size, True)
    if not writer.isOpened():
        raise IOError("Khong tao duoc file output: " + path)
    return writer


def verify_output(in_path, out_path):
    """So sanh output voi input: so frame, FPS, kich thuoc, so kenh mau."""
    a, b = cv2.VideoCapture(in_path), cv2.VideoCapture(out_path)
    info = []
    for cap in (a, b):
        ok, fr = cap.read()
        info.append(dict(
            w=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), h=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            fps=cap.get(cv2.CAP_PROP_FPS), n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            ch=fr.shape[2] if ok and fr.ndim == 3 else 1))
        cap.release()
    i, o = info
    print("  Kiem tra output:")
    print("    kich thuoc : %dx%d -> %dx%d  %s" % (i["w"], i["h"], o["w"], o["h"],
          "OK" if (i["w"], i["h"]) == (o["w"], o["h"]) else "SAI"))
    print("    FPS        : %.3f -> %.3f  %s" % (i["fps"], o["fps"],
          "OK" if abs(i["fps"] - o["fps"]) < 0.01 else "SAI"))
    print("    so frame   : %d -> %d  %s" % (i["n"], o["n"], "OK" if i["n"] == o["n"] else "SAI"))
    print("    he mau     : %d kenh (BGR) -> %d kenh  %s" % (i["ch"], o["ch"],
          "OK" if i["ch"] == o["ch"] else "SAI"))
    return i, o


def now_seconds():
    """Thoi gian hien tai (giay) bang ham cua OpenCV."""
    return cv2.getTickCount() / cv2.getTickFrequency()


def process_video(input_path, output_path, ids_text):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise IOError("Khong mo duoc video: " + input_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps != fps or fps <= 0:
        fps = 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print("[%s] %d frame, %.3f fps" % (input_path, total, fps))

    # ---------------- LUOT 1: detect + track ----------------
    t0 = now_seconds()
    tracker = None
    idx, fw, fh = 0, None, None
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if fw is None:
            fh, fw = frame.shape[:2]
            tracker = SignTracker(fw, fh)
        tracker.update(idx, detect_candidates(frame))
        idx += 1
        if idx % 300 == 0:
            print("  luot 1: %d/%d frame (%.0fs)" % (idx, total, now_seconds() - t0))
    cap.release()
    n_frames = idx
    if tracker is None:
        raise IOError("Video khong co frame nao: " + input_path)
    boxes_per_frame = tracker.finalize(fw, fh)
    print("  -> %d track bien bao sau hau xu ly" % len(tracker.kept_tracks))

    # ---------------- LUOT 2: ve + ghi video ----------------
    cap = cv2.VideoCapture(input_path)
    writer = open_writer(output_path, fps, (fw, fh))
    written = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        for (x, y, w, h) in boxes_per_frame.get(written, []):
            cv2.rectangle(frame, (x, y), (x + w, y + h), BOX_COLOR, 2)
        draw_student_ids(frame, ids_text)
        writer.write(frame)
        written += 1
        if SHOW_PREVIEW:
            cv2.imshow("preview", frame)
            cv2.waitKey(1)
    cap.release()
    writer.release()
    if SHOW_PREVIEW:
        cv2.destroyAllWindows()
    if written != n_frames:
        print("  CANH BAO: luot 2 ghi %d frame, luot 1 doc %d frame" % (written, n_frames))
    print("  Da ghi %s (%d frame, %.0fs)" % (output_path, written, now_seconds() - t0))
    verify_output(input_path, output_path)
    return tracker


def main():
    ids_text = " - ".join(STUDENT_IDS)
    prefix = "_".join(STUDENT_IDS)
    for name in INPUT_VIDEOS:                    # cung mot code + cung mot bo tham so cho moi video
        process_video(name, "%s_%s" % (prefix, name), ids_text)


if __name__ == "__main__":
    main()
