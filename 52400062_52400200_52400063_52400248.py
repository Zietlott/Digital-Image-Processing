# -*- coding: utf-8 -*-
"""
Phat hien, theo doi va NHAN DANG NOI DUNG bien bao giao thong trong video,
ghi video ket qua (box + noi dung bien + MSSV).
Mon: Introduction to Digital Image Processing (505060) - Final Report

Chi dung OpenCV, NumPy, math. Khong dung model pre-trained.
Cung mot thuat toan va cung mot bo tham so cho moi video dau vao.
Khong hardcode theo frame hay video cu the.
Mau (template) dung de nhan dang mui ten / ky hieu duoc VE BANG CODE
(cv2.line, cv2.circle, ...) ngay trong chuong trinh, khong cat tu video.

PIPELINE 2 LUOT (two-pass)
  Luot 1  : doc tung frame -> tim ung vien 3 dang hinh hoc:
              * TRON      (bien cam / bien hieu lenh: vien do, nen xanh)
              * TAM GIAC  (bien canh bao: nen vang, vien do)
              * CHU NHAT  (bien chi dan: nen xanh)
            -> ghep ung vien giua cac frame thanh "track" (cung dang hinh)
  Hau xu ly track:
            - loai track khong phai bien bao (qua ngan, sai hinh, nhieu...)
            - noi track bi dut, xoa track trung nhau (unique)
            - noi suy box o frame bi mat detection + lam muot box (stable)
            - NHAN DANG NOI DUNG: bo phieu tren cac quan sat lon nhat cua track
              (ti le mau + vi tri mau + so khop mau ve bang code)
  Luot 2  : doc lai video -> ve box + noi dung + MSSV len moi frame -> ghi MP4
            (cung kich thuoc, cung FPS, cung he mau BGR voi video goc)

"""
import math

import cv2
import numpy as np

# =============================================================================
# 0. CAU HINH CHUNG
# =============================================================================
# MSSV cac thanh vien. Ten file output va chu hien tren frame lay tu day.
STUDENT_IDS = ["52400062", "52400200", "52400063", "52400248"]

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
# Mau vang bien canh bao.
YEL_H_MIN, YEL_H_MAX = 14, 34
YEL_S_MIN, YEL_V_MIN = 110, 120

# ---- Bien TAM GIAC canh bao (long vang, vien do, dinh huong len) ----
TRI_AR = (0.80, 1.60)       # w / h cua long vang (tam giac deu ~ 1.15)
TRI_FILL = (0.38, 0.66)     # dien tich bao loi / (w*h): tam giac ly tuong = 0.5
TRI_APEX_DX = 0.22          # dinh tren lech tam toi da (ti le w)
TRI_EXPAND = 1.30           # no box long vang ra de bao ca vien do
# ---- Bien CHU NHAT chi dan (nen xanh) ----
RECT_AR = (0.70, 2.60)      # w / h
RECT_FILL = 0.82            # dien tich bao loi / (w*h): chu nhat ~ 1, tron ~ 0.785
RECT_CORNER = 0.15          # o vuong goc (ti le canh) phai duoc bao loi phu kin (>= 3/4 goc)
RECT_BLUE_MIN = 0.35        # ti le pixel xanh toi thieu trong HCN
RECT_MIN_AREA = 120         # px (he goc): bien chu nhat qua nho khong doc duoc noi dung
PATCH = 64                  # kich thuoc anh chuan hoa cua 1 ung vien (de nhan dang)
PATCH_KEEP = 6              # moi track giu PATCH_KEEP anh lon nhat de bo phieu noi dung

# ---- Vung quan tam (ti le theo kich thuoc frame) ----
ROI_Y_MAX = 0.55          # chi tim bien o phan tren frame

# ---- [C] Tien xu ly chong nhieu ----
USE_BILATERAL = True      # loc song phuong: giam nhieu nhung giu bien (vanh do manh)
BILATERAL_D, BILATERAL_SC, BILATERAL_SS = 5, 40, 5
USE_CLAHE = True          # can bang tuong phan cuc bo tren kenh V (chi dung de tao mask)
CLAHE_CLIP, CLAHE_GRID = 2.0, (8, 8)
OPEN_K = 2                # nhan MORPH_OPEN (0 = tat). Nho de khong xoa vanh mong cua bien xa

# ---- Loc hinh dang ung vien (moi frame) ----
CAND_MIN_AREA = 30        # [B] tinh theo he toa do GOC (da quy ve ti le 1x)
CAND_MAX_AREA = 90000
CAND_AR_RANGE = (0.60, 1.60)
# Do do tron theo elip noi tiep HCN bao: ellipse_fill = S_bao_loi / (pi/4 * w * h)
#   tron/elip ~ 1.0 | vuong/chu nhat ~ 1.27 | tam giac ~ 0.64
CAND_ELLIPSE_FILL = (0.82, 1.12)
CAND_MIN_SOLID = 0.70     # dien tich / dien tich bao loi
# [A] Bien nho (xa): anh bi nhoe, hinh dang dao dong nhieu -> noi long nguong.
SMALL_R = 10.0            # ban kinh (px, he goc) duoi nguong nay coi la "nho"
CAND_ELLIPSE_FILL_SMALL = (0.75, 1.18)
CAND_MIN_SOLID_SMALL = 0.62
CAND_MIN_BOX = 5          # canh box toi thieu (px, he goc)

# ---- [D] Multi-scale cho bien xa ----
MS_ENABLE = True
MS_SCALE = 2.0            # he so phong to
MS_Y_MAX = 0.40           # chi phong to dai tren cua frame (ti le chieu cao frame)
MS_X_MIN = 0.25           # va tu 25% chieu rong tro di (khop voi TRACK_X_MIN)
MS_MAX_R = 14.0           # chi giu ung vien nho (r goc < nguong) tu luot phong to

# ---- [N] Tang phong to x3 cho bien RAT XA. Dung CHINH XAC cac nguong mau / hinh dang
# / phan loai track cua ban goc => khong them nguy co nhan nham; chi giup track bat dau
# SOM HON (box xuat hien khi bien con rat nho).
FAR_ENABLE = True
FAR_SCALE = 3.0
FAR_Y_MAX = 0.32
FAR_X_MIN = 0.30
FAR_MAX_R = 8.0

# ---- [M] Debug: ve them box VANG manh cho cac track bi LOAI (de thay vat the nao bi
# loai / bi nhan nham). Video debug duoc ghi ra file "debug_..." rieng. Nop bai: False.
DEBUG_OVERLAY = False

# ---- [O] Khong bo ung vien lon neu vanh cua no lien mach 1 mau (bien co hoa tiet trong) ----
BLOB_KEEP_BORDER = 0.65

# ---- Tracker ----
TRK_MAX_GAP = 20          # [F] so frame toi da duoc mat detection ma track van song
TRK_GATE_DIST = 0.8       # khoang cach tam toi da (tinh theo kich thuoc box) de ghep
TRK_GATE_SCALE = 1.6      # ti le kich thuoc toi da giua 2 frame lien tiep

# ---- Phan loai track (hau xu ly) ----
# Moi nguong deu ap len TRUNG VI cua ca track, khong ap len 1 frame rieng le.
# "near" = 30% quan sat co kich thuoc LON nhat (luc doi tuong o gan, ro net).
FRAG_MIN_OBS = 4
TRACK_MIN_OBS = 15
TRACK_MIN_DENSITY = 0.40
TRACK_X_MIN = 0.33        # tam track (trung vi) o phan giua/phai
TRACK_Y_MIN = 0.20        # bo qua cac bien bao treo qua cao o phia tren (loai tru bien tren gia long mon)
TRACK_Y_MAX = 0.50        # tam track (trung vi) o nua tren frame
TRACK_NEAR_CIRC = 0.90    # do tron bao loi luc gan
TRACK_MIN_SIZE = 18       # duong kinh toi thieu (px) it nhat mot luc trong track
# [E] Kiem tra track "to dan" theo thoi gian (loai nhieu ngau nhien)
GROWTH_CHECK = True
GROWTH_MIN_SPAN = 10      # track phai keo dai it nhat so frame nay moi kiem tra
GROWTH_MIN_CORR = 0.40    # tuong quan toi thieu giua frame va log(r)
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
# Nguong cho track tam giac / chu nhat:
TRI_RED_MIN = 0.20        # vanh ngoai long vang phai co mau do
TRI_DARK = (0.03, 0.45)   # ky hieu den trong long (khong rong, khong toi het)
RECT_WHITE = (0.06, 0.60) # chu / ky hieu trang trong bien chi dan
RECT_BLUE_TRACK = 0.40
RECT_MAX_SIZE = 260       # px: bang quang cao lon hon nhieu
RECT_MIN_OBS = 40         # bien chi dan lon, phai thay on dinh lau hon (~1.3 s)
MERGE_MAX_GAP = 30        # noi 2 manh neu cach nhau <= 30 frame
MERGE_STATIC_GAP = 150    # hoac <= 150 frame neu vi tri gan nhu dung yen
MERGE_OVERLAP = 10
MERGE_SCALE = 1.8         # ti le kich thuoc toi da khi noi manh
MEDIAN_WIN = 7            # cua so loc trung vi box (so le)
SMOOTH_WIN = 5            # cua so trung binh truot box (so le)

# ---- Ve ----
BOX_COLOR = (0, 255, 0)

_CLAHE = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=CLAHE_GRID)


# =============================================================================
# 1. TIEN XU LY + PHAT HIEN UNG VIEN (TRON / TAM GIAC / CHU NHAT)
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


def _preprocess(img, scale=1.0):
    """
    Tra ve (hsv_goc, hsv_mask).
      hsv_goc  : dung de tinh dac trung (giu nguyen do sang that).
      hsv_mask : kenh V da CLAHE, chi dung de tao mask mau (on dinh voi anh sang yeu/chen).
    """
    if scale >= 3.0:          # anh phong to lon: bilateral qua cham, Gaussian nhe la du
        img = cv2.GaussianBlur(img, (3, 3), 0)
    elif USE_BILATERAL:
        img = cv2.bilateralFilter(img, BILATERAL_D, BILATERAL_SC, BILATERAL_SS)
    else:
        img = cv2.GaussianBlur(img, (5, 5), 0)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    if USE_CLAHE:
        hsv_mask = hsv.copy()
        hsv_mask[:, :, 2] = _CLAHE.apply(hsv[:, :, 2])
    else:
        hsv_mask = hsv
    return hsv, hsv_mask


def _find_candidates(img, scale=1.0, max_r=None):
    """
    Tim ung vien tron/elip tren anh `img` (co the da phong to `scale` lan).
    Tra ve ung vien o he toa do cua `img` (chua quy ve goc); box/cx/cy/r
    se duoc quy ve goc boi ham goi.
    max_r: neu co, chi giu ung vien co ban kinh GOC < max_r.
    """
    ih, iw = img.shape[:2]
    hsv, hsv_mask = _preprocess(img, scale)
    red, blue = build_color_masks(hsv_mask)
    k_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    k_big = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    k_open = (cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (OPEN_K, OPEN_K))
              if OPEN_K >= 2 else None)

    area_min = CAND_MIN_AREA * scale * scale
    area_max = CAND_MAX_AREA * scale * scale
    min_box = CAND_MIN_BOX * scale

    cands = []
    # 4 mask sau phep dong hinh thai: do, xanh, do|xanh, va do voi kernel lon
    # (kernel lon noi cac manh bi dut do vach trang cat ngang vien).
    for mask, kernel in ((red, k_small), (blue, k_small),
                         (cv2.bitwise_or(red, blue), k_small), (red, k_big)):
        m = mask
        if k_open is not None:
            m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k_open)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if not (area_min <= area <= area_max):
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

            r_orig = math.sqrt(w * h) / 2.0 / scale
            if max_r is not None and r_orig >= max_r:
                continue
            # [A] nguong phu thuoc kich thuoc
            if r_orig < SMALL_R:
                ef_rng, solid_min = CAND_ELLIPSE_FILL_SMALL, CAND_MIN_SOLID_SMALL
            else:
                ef_rng, solid_min = CAND_ELLIPSE_FILL, CAND_MIN_SOLID
            if not (ef_rng[0] <= efill <= ef_rng[1]):
                continue
            if solid < solid_min:
                continue

            cx, cy, a, b = x + w / 2.0, y + h / 2.0, w / 2.0, h / 2.0
            f = ellipse_features(hsv, red, blue, cx, cy, a, b)
            # Neu dia xanh nam trong vien do mong: no box ra de bao ca vien.
            if f["border_blue"] > f["border_red"] and f["ring_red"] >= 0.35:
                a, b = a * 1.22, b * 1.22
            bx, by = max(0, int(round(cx - a))), max(0, int(round(cy - b)))
            bw = min(int(round(2 * a)), iw - bx)
            bh = min(int(round(2 * b)), ih - by)
            if bw < min_box or bh < min_box:
                continue
            f.update(box=(bx, by, bw, bh), cx=cx, cy=cy, r=math.sqrt(a * b),
                     efill=efill, solid=solid, ar=ar, circ=circ, shape="circle")
            cands.append(f)

    cands += _find_triangles(hsv, hsv_mask, red, scale, max_r, k_small)
    cands += _find_rectangles(hsv, blue, scale, max_r, k_small)
    return cands


def _find_triangles(hsv, hsv_mask, red, scale, max_r, kernel):
    """
    Bien canh bao: long VANG hinh tam giac deu, dinh huong len, vien DO bao ngoai.
    Tim contour tren mask vang -> xap xi da giac -> kiem tra 3 dinh + vi tri dinh tren.
    """
    ih, iw = red.shape
    yel = cv2.inRange(hsv_mask, (YEL_H_MIN, YEL_S_MIN, YEL_V_MIN), (YEL_H_MAX, 255, 255))
    yel = cv2.morphologyEx(yel, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(yel, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if not (CAND_MIN_AREA * 0.6 * scale * scale <= area <= CAND_MAX_AREA * scale * scale):
            continue
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        x, y, w, h = cv2.boundingRect(hull)
        if hull_area <= 0 or h == 0:
            continue
        ar, fill = w / float(h), hull_area / float(w * h)
        if not (TRI_AR[0] <= ar <= TRI_AR[1] and TRI_FILL[0] <= fill <= TRI_FILL[1]):
            continue
        per = cv2.arcLength(hull, True)
        approx = cv2.approxPolyDP(hull, 0.07 * per, True)
        if len(approx) != 3:
            continue
        pts = approx.reshape(-1, 2).astype(np.float64)
        top = pts[np.argmin(pts[:, 1])]
        if abs(top[0] - (x + w / 2.0)) > TRI_APEX_DX * w or top[1] > y + 0.2 * h:
            continue                               # phai co 1 dinh o tren, chinh giua
        if np.sum(pts[:, 1] > y + 0.7 * h) != 2:
            continue                               # va 2 dinh o day
        r_orig = math.sqrt(w * h) / 2.0 / scale
        if max_r is not None and r_orig >= max_r:
            continue
        # no box quanh trong tam (2/3 chieu cao tinh tu dinh) de bao ca vien do
        gx, gy = x + w / 2.0, y + 2.0 * h / 3.0
        w2, h2 = w * TRI_EXPAND, h * TRI_EXPAND
        bx, by = int(round(gx - w2 / 2)), int(round(gy - 2.0 * h2 / 3.0))
        bx, by = max(0, bx), max(0, by)
        bw, bh = min(int(round(w2)), iw - bx), min(int(round(h2)), ih - by)
        if bw < CAND_MIN_BOX * scale or bh < CAND_MIN_BOX * scale:
            continue
        # dac trung: ti le do o vanh (giua tam giac lon va long vang), ti le toi trong long
        tri_big = np.zeros((bh, bw), np.uint8)
        apex = (int(gx - bx), 0)
        cv2.fillConvexPoly(tri_big, np.array([apex, (0, bh - 1), (bw - 1, bh - 1)], np.int32), 255)
        inner = np.zeros((bh, bw), np.uint8)
        cv2.fillConvexPoly(inner, (pts - [bx, by]).astype(np.int32), 255)
        ring = (tri_big > 0) & (inner == 0)
        rr = red[by:by + bh, bx:bx + bw] > 0
        v = hsv[by:by + bh, bx:bx + bw, 2]
        inn = inner > 0
        out.append(dict(
            shape="tri", box=(bx, by, bw, bh), cx=bx + bw / 2.0, cy=by + bh / 2.0,
            r=math.sqrt(bw * bh) / 2.0, efill=fill, solid=area / hull_area, ar=ar, circ=0.0,
            tri_red=float((rr & ring).sum()) / max(1, int(ring.sum())),
            mean_v=float(v[inn].mean()) if inn.any() else 0.0))
    return out


def _find_rectangles(hsv, blue, scale, max_r, kernel):
    """
    Bien chi dan: hinh CHU NHAT nen XANH, ky hieu / chu mau trang ben trong.
    Contour ngoai cua mask xanh lap day gan tron HCN bao (fill >= RECT_FILL).
    """
    ih, iw = blue.shape
    m = cv2.morphologyEx(blue, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if not (RECT_MIN_AREA * scale * scale <= area <= CAND_MAX_AREA * scale * scale):
            continue
        hull = cv2.convexHull(cnt)
        x, y, w, h = cv2.boundingRect(hull)
        ar, fill = w / float(h), cv2.contourArea(hull) / float(w * h)
        if not (RECT_AR[0] <= ar <= RECT_AR[1]) or fill < RECT_FILL:
            continue
        r_orig = math.sqrt(w * h) / 2.0 / scale
        if max_r is not None and r_orig >= max_r:
            continue
        # 4 goc cua HCN bao phai duoc lap day (hinh tron / elip de trong 4 goc)
        hm = np.zeros((h, w), np.uint8)
        cv2.fillConvexPoly(hm, (hull.reshape(-1, 2) - [x, y]).astype(np.int32), 1)
        cw, ch = max(1, int(w * RECT_CORNER)), max(1, int(h * RECT_CORNER))
        corners = (hm[:ch, :cw], hm[:ch, -cw:], hm[-ch:, :cw], hm[-ch:, -cw:])
        if sorted(float(c.mean()) for c in corners)[1] < 0.35:
            continue                               # >= 3/4 goc day (cho phep 1 goc bi che)
        blue_frac = float((blue[y:y + h, x:x + w] > 0).mean())
        if blue_frac < RECT_BLUE_MIN:
            continue
        mx, my = max(1, w // 10), max(1, h // 10)
        sub = hsv[y + my:y + h - my, x + mx:x + w - mx]
        if sub.size == 0:
            continue
        # o trang ngay BEN TRAI bien (hinh minh hoa cua bien canh bao dang chu nhat,
        # vd "Doan duong hay xay ra tai nan"): nguong trang tuong doi theo nen xanh
        box_hsv = hsv[y:y + h, x:x + w]
        bmask = blue[y:y + h, x:x + w] > 0
        s_blue = float(np.median(box_hsv[:, :, 1][bmask])) if bmask.any() else 150.0
        v_ref = float(np.percentile(box_hsv[:, :, 2], 97))
        lx = max(0, x - int(0.8 * h))
        strip = hsv[y + my:y + h - my, lx:x]
        left_white = float(((strip[:, :, 1] < max(80.0, 0.55 * s_blue))
                            & (strip[:, :, 2] >= max(110.0, 0.60 * v_ref))).mean()) \
            if strip.size else 0.0
        out.append(dict(
            shape="rect", box=(x, y, w, h), cx=x + w / 2.0, cy=y + h / 2.0,
            r=math.sqrt(w * h) / 2.0, efill=fill, solid=1.0, ar=ar, circ=0.0,
            rect_blue=blue_frac, left_white=left_white,
            mean_v=float(sub[:, :, 2].mean())))
    return out


def _to_original(c, scale, ox, oy):
    """Quy ung vien tu anh crop+phong to ve he toa do frame goc."""
    if scale == 1.0 and ox == 0 and oy == 0:
        return c
    x, y, w, h = c["box"]
    c = dict(c)
    c["box"] = (int(round(x / scale)) + ox, int(round(y / scale)) + oy,
                max(1, int(round(w / scale))), max(1, int(round(h / scale))))
    c["cx"] = c["cx"] / scale + ox
    c["cy"] = c["cy"] / scale + oy
    c["r"] = c["r"] / scale
    return c


def detect_candidates(frame):
    """
    Tra ve danh sach ung vien bien tron/elip trong 1 frame.
    Moi ung vien: {'box': (x, y, w, h), 'cx', 'cy', 'r', 'efill', ... dac trung}
    """
    fh, fw = frame.shape[:2]
    y_lim = int(fh * ROI_Y_MAX)
    roi = frame[:y_lim]

    cands = _find_candidates(roi, 1.0)

    # [D] Multi-scale: phong to dai tren cua frame de bat bien rat nho (xa).
    if MS_ENABLE:
        my = int(fh * MS_Y_MAX)
        mx = int(fw * MS_X_MIN)
        crop = frame[:my, mx:]
        if crop.size > 0:
            up = cv2.resize(crop, None, fx=MS_SCALE, fy=MS_SCALE,
                            interpolation=cv2.INTER_CUBIC)
            for c in _find_candidates(up, MS_SCALE, max_r=MS_MAX_R):
                cands.append(_to_original(c, MS_SCALE, mx, 0))

    # [N] tang x3 cho bien rat xa (cung nguong voi ban goc)
    if FAR_ENABLE:
        my3, mx3 = int(fh * FAR_Y_MAX), int(fw * FAR_X_MIN)
        crop3 = frame[:my3, mx3:]
        if crop3.size > 0:
            up3 = cv2.resize(crop3, None, fx=FAR_SCALE, fy=FAR_SCALE,
                             interpolation=cv2.INTER_CUBIC)
            for c in _find_candidates(up3, FAR_SCALE, max_r=FAR_MAX_R):
                cands.append(_to_original(c, FAR_SCALE, mx3, 0))

    # Bo khoi dinh: 1 ung vien lon chua >= 2 ung vien nho rieng biet ben trong.
    def _area(c):
        return c["box"][2] * c["box"][3]
    merged_blobs = set()
    for i, c in enumerate(cands):
        inside = [o for o in cands if o is not c and _area(o) < 0.7 * _area(c)
                  and iou_min(o["box"], c["box"]) >= 0.8]
        # [O] Bien co hoa tiet ben trong (vd: bien cam dung/do co gach cheo do chia long
        # thanh 4 manh xanh roi nhau) KHONG phai khoi dinh: vanh ngoai cua no lien mach
        # 1 mau (>= BLOB_KEEP_BORDER vanh d=0.7..0.95 cung mau), con khoi dinh that
        # (2 bien chong len nhau, 2 mau) thi khong dat.
        coherent = (c["shape"] != "circle"
                    or max(c["border_red"], c["border_blue"]) >= BLOB_KEEP_BORDER)
        if not coherent and any(iou_min(p["box"], q["box"]) < 0.2
               for k, p in enumerate(inside) for q in inside[k + 1:]):
            merged_blobs.add(i)
    cands = [c for i, c in enumerate(cands) if i not in merged_blobs]

    # Gop ung vien trung nhau: giu ung vien lon nhat.
    cands.sort(key=_area, reverse=True)
    kept = []
    for c in cands:
        if all(iou_min(c["box"], k["box"]) < 0.5 for k in kept):
            kept.append(c)
    # Cat anh chuan hoa PATCH x PATCH cua moi ung vien (dung cho buoc nhan dang noi dung).
    for c in kept:
        x, y, w, h = c["box"]
        roi_c = frame[max(0, y):min(fh, y + h), max(0, x):min(fw, x + w)]
        c["patch"] = (cv2.resize(roi_c, (PATCH, PATCH), interpolation=cv2.INTER_AREA
                                 if w > PATCH else cv2.INTER_CUBIC) if roi_c.size else None)
    return kept


# =============================================================================
# 2. TRACKER: ON DINH (stable) + DUY NHAT (unique)
# =============================================================================
def _box_to_cs(b):
    """(x, y, w, h) -> (cx, cy, s) voi s = kich thuoc dac trung."""
    return b[0] + b[2] / 2.0, b[1] + b[3] / 2.0, math.sqrt(b[2] * b[3])


def _patch_priority(d):
    """
    Uu tien giu patch cua quan sat LON (gan) va box DUNG TI LE (bien tron: w/h ~ 1).
    Box meo (vd dinh them vat the canh ben) cho anh nhan dang sai -> giam uu tien.
    """
    if d["shape"] == "circle" and not (0.80 <= d["box"][2] / float(max(1, d["box"][3])) <= 1.25):
        return d["r"] * 0.1
    return d["r"]


class Track:
    """1 doi tuong duoc theo doi qua nhieu frame."""
    FIT_LAST = 10

    def __init__(self, tid, frame_idx, det):
        self.id = tid
        self.shape = det["shape"]          # circle / tri / rect: chi ghep cung dang hinh
        self.obs = [(frame_idx, det)]
        self.label = ""

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
        self.trim_patches()

    def trim_patches(self):
        """Chi giu PATCH_KEEP patch cua cac quan sat lon nhat (tiet kiem bo nho)."""
        withp = [d for _, d in self.obs if d.get("patch") is not None]
        if len(withp) > PATCH_KEEP:
            for d in sorted(withp, key=_patch_priority)[:-PATCH_KEEP]:
                d["patch"] = None


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
        self.rejected = {}

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
                if d["shape"] != t.shape:
                    continue
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
    def _grows_over_time(dets_obs):
        """
        [E] Bien tren xe dang tien toi phai to dan: log(r) tang tuyen tinh theo frame.
        Tra ve True neu he so goc > 0 va tuong quan >= GROWTH_MIN_CORR.
        Track qua ngan (< GROWTH_MIN_SPAN frame) thi bo qua kiem tra.
        """
        fr = np.array([f for f, _ in dets_obs], dtype=np.float64)
        if np.ptp(fr) < GROWTH_MIN_SPAN:
            return True
        ls = np.log(np.array([max(1.0, d["r"]) for _, d in dets_obs], dtype=np.float64))
        if np.std(ls) < 1e-9 or np.std(fr) < 1e-9:
            return False
        slope = np.polyfit(fr, ls, 1)[0]
        corr = float(np.corrcoef(fr, ls)[0, 1])
        return slope > 0 and corr >= GROWTH_MIN_CORR

    @classmethod
    def _is_traffic_sign(cls, t, fw, fh):
        """Phan loai 1 track da noi: vi tri + do ben + hinh dang luc gan + mau/noi dung."""
        dets = [d for _, d in t.obs]
        n = len(dets)
        if n < TRACK_MIN_OBS:
            return False
        med = lambda k, ds=dets: float(np.median([d[k] for d in ds]))
        
        # Đã cập nhật dòng này: Thêm điều kiện med("cy") / fh < TRACK_Y_MIN
        if med("cx") / fw < TRACK_X_MIN or med("cy") / fh < TRACK_Y_MIN or med("cy") / fh > TRACK_Y_MAX:
            return False
            
        if 2 * max(d["r"] for d in dets) < TRACK_MIN_SIZE:
            return False
        if GROWTH_CHECK and not cls._grows_over_time(t.obs):
            return False
        near = sorted(dets, key=lambda d: d["r"])[int(n * 0.7):]
        if t.shape == "tri":
            # long vang + vien do + co ky hieu toi ben trong (tinh tren patch gan nhat)
            return (med("tri_red", near) >= TRI_RED_MIN
                    and TRI_DARK[0] <= _patch_median(t, _tri_symbol_frac) <= TRI_DARK[1])
        if t.shape == "rect":
            if n < RECT_MIN_OBS:
                return False
            # nen xanh, co chu / ky hieu trang, khong qua lon (bang quang cao)
            return (RECT_WHITE[0] <= _patch_median(t, _rect_white_frac) <= RECT_WHITE[1]
                    and med("rect_blue", near) >= RECT_BLUE_TRACK
                    and 2 * max(d["r"] for d in dets) <= RECT_MAX_SIZE
                    and med("mean_v", near) >= BLUE_MEANV_MIN)
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
        if a.shape != b.shape:
            return False
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
                a.trim_patches()
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
        """Tra ve dict {frame_idx: [(box, noi_dung), ...]} da on dinh + duy nhat."""
        tracks = self.finished + self.active
        self.active, self.finished = [], tracks

        frags = [t for t in tracks if self._is_fragment(t)]
        merged = self._merge_fragments(frags)
        signs = [t for t in merged if self._is_traffic_sign(t, fw, fh)]
        self.rejected = {}
        if DEBUG_OVERLAY:
            for t in merged:
                if t not in signs and len(t.obs) >= 8:
                    for f, b in self._dense_boxes(t).items():
                        self.rejected.setdefault(f, []).append(b)
        for t in signs:
            if t.shape == "circle":
                self._harmonize_ringed(t)
            t.label = recognize_track(t)              # nhan dang noi dung 1 lan / track
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
                if all(iou_min((x, y, w, h), o[0]) < 0.5 for o in lst):
                    lst.append(((x, y, w, h), t.label))
        self.kept_tracks = [t for t, _ in kept]
        return per_frame


# =============================================================================
# 3. NHAN DANG NOI DUNG BIEN BAO
#    Dau vao: anh chuan hoa PATCH x PATCH cua bien (box -> resize).
#    Buoc 1: tach mask mau DO / XANH / VANG / TRANG / DEN trong patch.
#    Buoc 2: dua vao dang hinh + mau vien + mau nen -> nhom bien.
#    Buoc 3: trong tung nhom, phan biet noi dung bang ti le mau theo vung
#            (vach ngang, duong cheo, nua tren/duoi...) hoac SO KHOP MAU
#            (template matching) voi ky hieu ve bang code.
#    Track: bo phieu tren PATCH_KEEP quan sat lon nhat (gan nhat, ro nhat).
# =============================================================================
_U, _V = np.meshgrid((np.arange(PATCH) - (PATCH - 1) / 2.0) / (PATCH / 2.0),
                     (np.arange(PATCH) - (PATCH - 1) / 2.0) / (PATCH / 2.0))
_D = np.sqrt(_U ** 2 + _V ** 2)          # khoang cach chuan hoa toi tam patch (1 = mep)
_SLASH = np.abs(_U - _V) < 0.20          # dai cheo "\" (vach cam cua bien cam)
_SLASH2 = np.abs(_U + _V) < 0.20         # dai cheo "/"


def patch_masks(patch):
    """
    Mask mau (bool) cua 1 patch BGR.
    TRANG / DEN lay nguong TUONG DOI so voi mau nen cua chinh bien (do/xanh/vang):
    video bi nen mau (chroma subsampling) nen net trang manh tren nen xanh van co
    do bao hoa kha cao -> nguong co dinh S < 80 se bo sot ky hieu.
    """
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1].astype(np.int32), hsv[:, :, 2].astype(np.int32)
    red = ((h <= 10) | (h >= 150)) & (s >= 70) & (v >= 50)
    blue = (h >= 95) & (h <= 135) & (s >= 70) & (v >= 40)
    yellow = (h >= YEL_H_MIN) & (h <= YEL_H_MAX) & (s >= 90) & (v >= 100)
    col = red | blue | yellow
    s_col = float(np.median(s[col])) if col.sum() > 20 else 150.0
    v_col = float(np.median(v[col])) if col.sum() > 20 else 150.0
    vmax = float(np.percentile(v, 97))
    white = (s < max(80.0, 0.55 * s_col)) & (v >= max(110.0, 0.60 * vmax))
    black = (v < max(60.0, 0.60 * v_col)) & ~col
    return dict(red=red, blue=blue, yellow=yellow, white=white, black=black)


def _frac(mask, region):
    n = int(region.sum())
    return float((mask & region).sum()) / n if n else 0.0


# ---------- Ky hieu ve bang code (khong lay tu video) ----------
def _pt(u, v):
    """Toa do chuan hoa (-1..1) -> pixel cua patch."""
    return (int(round((u + 1) * PATCH / 2.0 - 0.5)), int(round((v + 1) * PATCH / 2.0 - 0.5)))


def _arrow_template(angle_deg):
    """Mui ten trang tren bien hieu lenh, huong angle (0 = sang phai, 90 = len tren)."""
    t = np.zeros((PATCH, PATCH), np.uint8)
    a = math.radians(angle_deg)
    du, dv = math.cos(a), -math.sin(a)
    tail, tip = (-0.48 * du, -0.48 * dv), (0.50 * du, 0.50 * dv)
    neck = (0.18 * du, 0.18 * dv)
    cv2.line(t, _pt(*tail), _pt(*neck), 255, max(2, PATCH // 10))
    pu, pv = -dv, du                                   # vector vuong goc
    head = [_pt(*tip), _pt(neck[0] + 0.28 * pu, neck[1] + 0.28 * pv),
            _pt(neck[0] - 0.28 * pu, neck[1] - 0.28 * pv)]
    cv2.fillConvexPoly(t, np.array(head, np.int32), 255)
    return t


def _turn_template(kind):
    """Mui ten den cua bien cam re: 'right', 'left' (doi xung), 'uturn'."""
    t = np.zeros((PATCH, PATCH), np.uint8)
    th = max(2, PATCH // 11)
    if kind == "uturn":
        cv2.line(t, _pt(0.22, 0.50), _pt(0.22, -0.10), 255, th)
        cv2.ellipse(t, _pt(0.0, -0.10), (int(0.22 * PATCH / 2), int(0.22 * PATCH / 2)),
                    0, 180, 360, 255, th)
        cv2.line(t, _pt(-0.22, -0.10), _pt(-0.22, 0.15), 255, th)
        cv2.fillConvexPoly(t, np.array([_pt(-0.22, 0.38), _pt(-0.42, 0.12), _pt(-0.02, 0.12)],
                                       np.int32), 255)
        return t
    cv2.line(t, _pt(-0.18, 0.55), _pt(-0.18, -0.12), 255, th)
    cv2.ellipse(t, _pt(0.0, -0.12), (int(0.18 * PATCH / 2), int(0.18 * PATCH / 2)),
                0, 180, 270, 255, th)
    cv2.line(t, _pt(0.0, -0.30), _pt(0.18, -0.30), 255, th)
    cv2.fillConvexPoly(t, np.array([_pt(0.48, -0.30), _pt(0.16, -0.52), _pt(0.16, -0.08)],
                                   np.int32), 255)
    return t if kind == "right" else cv2.flip(t, 1)


# Bang mau: (ten hien thi, anh mau). Bien hieu lenh: 8 huong mui ten.
_MANDATORY = [("Turn right", 0), ("Go straight or right", 45), ("Go straight", 90),
              ("Go straight or left", 135), ("Turn left", 180), ("Keep left", 225),
              ("Keep right", 315)]
_MANDATORY_T = [(name, _arrow_template(ang)) for name, ang in _MANDATORY]
_TURN_T = [("No right turn", _turn_template("right")), ("No left turn", _turn_template("left")),
           ("No U-turn", _turn_template("uturn"))]
TEMPLATE_MIN_SCORE = 0.30


def _match(mask, templates, valid):
    """
    So khop mask nhi phan voi cac mau bang cv2.matchTemplate (TM_CCOEFF_NORMED).
    Lam mo 2 anh de chiu lech nho; cat vien 4 px cua mau de cho phep dich +-4 px.
    valid: vung pixel duoc tinh (vd: bo dai vach cheo do che ky hieu).
    Tra ve (ten tot nhat, diem, do chenh voi mau thu 2).
    """
    img = cv2.GaussianBlur((mask & valid).astype(np.float32), (7, 7), 0)
    scores = []
    for name, t in templates:
        tt = cv2.GaussianBlur(((t > 0) & valid).astype(np.float32), (7, 7), 0)[4:-4, 4:-4]
        scores.append((float(cv2.matchTemplate(img, tt, cv2.TM_CCOEFF_NORMED).max()), name))
    scores.sort(reverse=True)
    margin = scores[0][0] - scores[1][0] if len(scores) > 1 else scores[0][0]
    return scores[0][1], scores[0][0], margin


def _tri_inner():
    """Mask long tam giac (thu nho quanh trong tam de bo vien do) tren patch."""
    tri = np.zeros((PATCH, PATCH), np.uint8)
    g = np.array([PATCH / 2.0, PATCH * 2.0 / 3.0])
    pts = np.array([[PATCH / 2.0, 0], [0, PATCH - 1], [PATCH - 1, PATCH - 1]])
    cv2.fillConvexPoly(tri, ((pts - g) * 0.72 + g).astype(np.int32), 1)
    return tri > 0


_TRI_IN = _tri_inner()


def _tri_symbol(patch, m):
    """Ky hieu den trong long bien canh bao = pixel toi hon han nen vang."""
    v = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)[:, :, 2].astype(np.float64)
    yel = m["yellow"] & _TRI_IN
    if yel.sum() < 20:
        return np.zeros_like(yel)
    vy = float(np.median(v[yel]))
    return _TRI_IN & ~m["yellow"] & ~m["red"] & (v < 0.70 * vy)


def _tri_symbol_frac(patch):
    return _frac(_tri_symbol(patch, patch_masks(patch)), _TRI_IN)


def _rect_white_frac(patch):
    return _frac(patch_masks(patch)["white"], (np.abs(_U) < 0.85) & (np.abs(_V) < 0.85))


def _patch_median(t, fn):
    """Trung vi cua dac trung fn tinh tren cac patch con giu cua track (quan sat gan nhat)."""
    vals = [fn(d["patch"]) for _, d in t.obs if d.get("patch") is not None]
    return float(np.median(vals)) if vals else 0.0


def recognize_circle(patch):
    """Bien tron: bien cam (vien do) hoac bien hieu lenh (nen xanh)."""
    m = patch_masks(patch)
    ring = (_D >= 0.72) & (_D <= 0.98)
    inner = _D <= 0.62
    ring_red, ring_blue = _frac(m["red"], ring), _frac(m["blue"], ring)
    in_red, in_blue = _frac(m["red"], inner), _frac(m["blue"], inner)
    in_white = _frac(m["white"], inner)
    if ring_red >= ring_blue or in_red > 0.35:
        # (a) Nen do + vach trang nam ngang -> Cam di nguoc chieu
        bar = (np.abs(_V) < 0.20) & (np.abs(_U) < 0.55)
        off_bar = (np.abs(_V) > 0.38) & (_D < 0.75)
        if (in_red > 0.30 and _frac(m["white"], bar) > 0.35
                and _frac(m["red"], off_bar) > 0.45):
            return "No entry", 1.0
        # (b) Nen xanh + vien do: dem vach do theo 2 duong cheo
        if in_blue > 0.25:
            # dinh vi lai dia xanh trong patch (box co the bi lech / dinh vat the do ben canh)
            # (vach do chia dia xanh thanh 2 hoac 4 manh -> lay HCN bao cua cac manh xanh)
            n, _, st, cen = cv2.connectedComponentsWithStats(m["blue"].astype(np.uint8), 8)
            if n > 1:
                k = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
                reach = 0.9 * max(st[k, 2], st[k, 3])        # cac manh xanh lan can cua dia
                big = np.array([st[i] for i in range(1, n)
                                if st[i, 4] >= 0.25 * st[k, 4]
                                and np.hypot(*(cen[i] - cen[k])) <= reach])
            else:
                big = np.array([[0, 0, PATCH, PATCH, 0]])
            x, y = big[:, 0].min(), big[:, 1].min()
            w = (big[:, 0] + big[:, 2]).max() - x
            h = (big[:, 1] + big[:, 3]).max() - y
            u = (np.arange(PATCH)[None, :] - (x + w / 2.0)) / max(1.0, w / 2.0)
            v = (np.arange(PATCH)[:, None] - (y + h / 2.0)) / max(1.0, h / 2.0)
            dd = np.sqrt(u ** 2 + v ** 2)
            disk = (dd < 0.75) & (dd > 0.40)                # bo tam (vach day giao nhau o tam)
            d1 = _frac(m["red"], (np.abs(u - v) < 0.20) & disk)
            d2 = _frac(m["red"], (np.abs(u + v) < 0.20) & disk)
            lo, hi = min(d1, d2), max(d1, d2)
            if lo >= 0.12 and lo >= 0.45 * hi:
                return "No stopping & parking", 1.0   # 2 vach cheo (X)
            return "No parking", 1.0                  # 1 vach cheo
        # (c) Nen trang + ky hieu den + vach cheo -> bien cam re (so khop mau)
        vch = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)[:, :, 2].astype(np.float64)
        v_bg = float(np.percentile(vch[inner], 90))      # do sang nen trang (co the bi toi)
        dark = (vch < 0.60 * v_bg) & ~m["red"] & ~m["blue"]
        if _frac(dark, inner) > 0.04:                   # nen sang + ky hieu toi
            valid = (_D < 0.80) & ~_SLASH
            name, sc, mg = _match(dark, _TURN_T, valid)
            if sc >= TEMPLATE_MIN_SCORE or (sc >= 0.15 and mg >= 0.10):
                return name, max(sc, 0.35)
            return "Prohibitory sign", 0.3
        return "Prohibitory sign", 0.2
    # (d) Bien hieu lenh nen xanh: so khop mui ten trang voi 7 huong
    if in_white > 0.05:
        name, sc, mg = _match(m["white"], _MANDATORY_T, _D < 0.80)
        if sc >= TEMPLATE_MIN_SCORE:
            return name, sc
    return "Mandatory sign", 0.2


def recognize_triangle(patch):
    """Bien canh bao tam giac: phan biet ky hieu CHU (2 dong) va HINH NGUOI."""
    m = patch_masks(patch)
    if _frac(m["yellow"], _TRI_IN) < 0.25:
        return "Warning sign", 0.2
    sym = _tri_symbol(patch, m)
    if sym.sum() < 8:
        return "Warning sign", 0.2
    rows = sym.sum(axis=1) > 0
    # dem so "dai dong" co ky hieu (bo khe 1 px)
    runs, prev, gap = 0, False, 0
    for r in rows:
        if r and not prev and gap != 1:
            runs += 1
        gap = 0 if r else gap + 1
        prev = r
    n_cc, _, stats, _ = cv2.connectedComponentsWithStats(sym.astype(np.uint8), 8)
    blobs = [s for s in stats[1:] if s[cv2.CC_STAT_AREA] >= 4]
    if runs >= 2 and len(blobs) >= 4:
        return "Slow down", 1.0                      # chu "DI CHAM" xep 2 dong
    big = max(s[cv2.CC_STAT_AREA] for s in blobs) if blobs else 0
    if runs == 1 and 1 <= len(blobs) <= 4 and big >= 0.03 * PATCH * PATCH:
        return "Children crossing", 0.8              # hinh ve (2 tre em): 1 khoi den lon
    return "Warning sign", 0.3


def recognize_rectangle(patch, ar, left_white=0.0):
    """Bien chu nhat nen xanh: phan biet theo VI TRI + HINH DANG vung trang."""
    m = patch_masks(patch)
    white = m["white"]
    # Phan tich thanh phan lien thong cua ky hieu trang
    n, _, st, _ = cv2.connectedComponentsWithStats(white.astype(np.uint8), 8)
    comps = [s for s in st[1:] if s[cv2.CC_STAT_AREA] >= 6]
    H = W = float(PATCH)
    tall = [s for s in comps if s[cv2.CC_STAT_HEIGHT] > 0.40 * H
            and s[cv2.CC_STAT_WIDTH] < 0.35 * W]
    if ar >= 1.4 and len(tall) >= 2:
        return "Lane direction", 1.0                 # cac mui ten / vach doc chay het chieu cao
    arrow = [s for s in comps if s[cv2.CC_STAT_HEIGHT] > 0.50 * H
             and s[cv2.CC_STAT_WIDTH] < 0.50 * W]
    if tall or arrow:
        return "Direction sign", 0.6                 # co mui ten lon + chu (bien chi huong)
    # O trang lon sat BEN TRAI bien (hinh minh hoa) + chi co chu trang -> canh bao tai nan
    if left_white > 0.45:
        return "Accident-prone area", 1.0
    # hinh camera: 1 khoi trang DAC (khong phai dong chu dai, mong) o nua tren
    icon = [s for s in comps if s[cv2.CC_STAT_TOP] + s[cv2.CC_STAT_HEIGHT] < 0.50 * H
            and 0.25 * W <= s[cv2.CC_STAT_WIDTH] <= 0.75 * W
            and s[cv2.CC_STAT_HEIGHT] >= 0.15 * H
            and s[cv2.CC_STAT_AREA] >= 0.30 * s[cv2.CC_STAT_WIDTH] * s[cv2.CC_STAT_HEIGHT]]
    text = [s for s in comps if s[cv2.CC_STAT_TOP] > 0.40 * H]
    if icon and len(text) >= 3:
        return "Camera enforcement zone", 0.8       # hinh camera o tren + chu o duoi
    return "Information sign", 0.3


def recognize_track(t):
    """Bo phieu nhan dang tren cac patch lon nhat cua track (co trong so = diem)."""
    obs = sorted((d for _, d in t.obs if d.get("patch") is not None),
                 key=_patch_priority, reverse=True)[:PATCH_KEEP]
    votes = {}
    for d in obs:
        if t.shape == "circle":
            name, sc = recognize_circle(d["patch"])
        elif t.shape == "tri":
            name, sc = recognize_triangle(d["patch"])
        else:
            name, sc = recognize_rectangle(d["patch"], d["ar"], d.get("left_white", 0.0))
        votes[name] = votes.get(name, 0.0) + sc
    if not votes:
        return {"circle": "Traffic sign", "tri": "Warning sign"}.get(t.shape, "Information sign")
    return max(votes.items(), key=lambda kv: kv[1])[0]


# =============================================================================
# 4. XU LY VIDEO: DOC / GHI MP4 / CHEN MSSV / KIEM TRA OUTPUT
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


def draw_label(frame, text, box, others=()):
    """
    Ghi noi dung bien bao ngay TREN box; neu sat mep tren hoac de len box cua bien
    khac (2 bien gan tren cung 1 cot) thi ghi DUOI box. Nen cung mau box, chu den.
    """
    fh, fw = frame.shape[:2]
    x, y, w, h = box
    scale = max(0.45, fh / 1080.0 * 0.62)
    thick = max(1, int(round(fh / 1080.0 * 1.6)))
    (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    pad = 3
    lw, lh = tw + 2 * pad, th + base + 2 * pad
    tx = min(max(0, x + w // 2 - lw // 2), max(0, fw - lw))

    def hits(ty):
        r = (tx, ty, lw, lh)
        return any(iou_min(r, o) > 0.15 for o in others)

    above, below = y - 4 - lh, y + h + 4
    if above >= 0 and not hits(above):
        ty = above
    elif below + lh <= fh and not hits(below):
        ty = below
    else:
        ty = max(0, above) if above >= 0 else min(below, fh - lh)
    cv2.rectangle(frame, (tx, ty), (tx + lw, ty + lh), BOX_COLOR, -1)
    cv2.putText(frame, text, (tx + pad, ty + pad + th), cv2.FONT_HERSHEY_SIMPLEX, scale,
                (0, 0, 0), thick, cv2.LINE_AA)


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
    if DEBUG_OVERLAY:
        output_path = "debug_" + output_path
    cap = cv2.VideoCapture(input_path)
    writer = open_writer(output_path, fps, (fw, fh))
    written = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if DEBUG_OVERLAY:
            for (x, y, w, h) in tracker.rejected.get(written, []):
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 1)
        items = boxes_per_frame.get(written, [])
        for (x, y, w, h), _ in items:
            cv2.rectangle(frame, (x, y), (x + w, y + h), BOX_COLOR, 2)
        for i, (b, label) in enumerate(items):
            draw_label(frame, label, b, [o for j, (o, _) in enumerate(items) if j != i])
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
