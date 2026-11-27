import collections
import ctypes
import json
import math
import os
import queue
import re
import shutil
import struct
import sys
import threading
import time
import warnings
import zlib
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import font as tkfont
from typing import List, Optional, Tuple

# Make coordinates consistent on high-DPI Windows screens (must run before Tk is created)
if sys.platform.startswith("win"):
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except Exception:
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass

try:
    import chess
    import chess.engine
    import mss
    import numpy as np
    import pyautogui
except ImportError as exc:
    raise SystemExit(
        f"Missing library: {exc.name}\nInstall everything with:\n"
        "    pip install chess mss numpy pyautogui"
    ) from exc

pyautogui.FAILSAFE = True   # slam the mouse into a screen corner to abort
pyautogui.PAUSE = 0.02

GAME_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chess_bot_last_game.json")


# ============================================================================
# BUILT-IN ENGINE (Negamax + Alpha-Beta + Quiescence + Piece-Square Tables)
# ============================================================================
PIECE_VALUES = {
    chess.PAWN: 100, chess.KNIGHT: 320, chess.BISHOP: 330,
    chess.ROOK: 500, chess.QUEEN: 900, chess.KING: 20000,
}

PAWN_PST = [
     0,  0,  0,  0,  0,  0,  0,  0,
    50, 50, 50, 50, 50, 50, 50, 50,
    10, 10, 20, 30, 30, 20, 10, 10,
     5,  5, 10, 25, 25, 10,  5,  5,
     0,  0,  0, 20, 20,  0,  0,  0,
     5, -5,-10,  0,  0,-10, -5,  5,
     5, 10, 10,-20,-20, 10, 10,  5,
     0,  0,  0,  0,  0,  0,  0,  0,
]
KNIGHT_PST = [
    -50,-40,-30,-30,-30,-30,-40,-50,
    -40,-20,  0,  0,  0,  0,-20,-40,
    -30,  0, 10, 15, 15, 10,  0,-30,
    -30,  5, 15, 20, 20, 15,  5,-30,
    -30,  0, 15, 20, 20, 15,  0,-30,
    -30,  5, 10, 15, 15, 10,  5,-30,
    -40,-20,  0,  5,  5,  0,-20,-40,
    -50,-40,-30,-30,-30,-30,-40,-50,
]
BISHOP_PST = [
    -20,-10,-10,-10,-10,-10,-10,-20,
    -10,  0,  0,  0,  0,  0,  0,-10,
    -10,  0,  5, 10, 10,  5,  0,-10,
    -10,  5,  5, 10, 10,  5,  5,-10,
    -10,  0, 10, 10, 10, 10,  0,-10,
    -10, 10, 10, 10, 10, 10, 10,-10,
    -10,  5,  0,  0,  0,  0,  5,-10,
    -20,-10,-10,-10,-10,-10,-10,-20,
]
ROOK_PST = [
      0,  0,  0,  0,  0,  0,  0,  0,
      5, 10, 10, 10, 10, 10, 10,  5,
     -5,  0,  0,  0,  0,  0,  0, -5,
     -5,  0,  0,  0,  0,  0,  0, -5,
     -5,  0,  0,  0,  0,  0,  0, -5,
     -5,  0,  0,  0,  0,  0,  0, -5,
     -5,  0,  0,  0,  0,  0,  0, -5,
      0,  0,  0,  5,  5,  0,  0,  0,
]
QUEEN_PST = [
    -20,-10,-10, -5, -5,-10,-10,-20,
    -10,  0,  0,  0,  0,  0,  0,-10,
    -10,  0,  5,  5,  5,  5,  0,-10,
     -5,  0,  5,  5,  5,  5,  0, -5,
      0,  0,  5,  5,  5,  5,  0, -5,
    -10,  5,  5,  5,  5,  5,  0,-10,
    -10,  0,  5,  0,  0,  0,  0,-10,
    -20,-10,-10, -5, -5,-10,-10,-20,
]
KING_PST = [
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -20,-30,-30,-40,-40,-30,-30,-20,
    -10,-20,-20,-20,-20,-20,-20,-10,
     20, 20,  0,  0,  0,  0, 20, 20,
     20, 30, 10,  0,  0, 10, 30, 20,
]
PST_MAP = {
    chess.PAWN: PAWN_PST, chess.KNIGHT: KNIGHT_PST, chess.BISHOP: BISHOP_PST,
    chess.ROOK: ROOK_PST, chess.QUEEN: QUEEN_PST, chess.KING: KING_PST,
}


class BuiltinAlphaBetaEngine:
    """Pure-Python Negamax + Alpha-Beta + MVV-LVA ordering + Quiescence search."""

    def __init__(self):
        self._stop = False
        self._deadline = 0.0

    def evaluate(self, board: chess.Board) -> int:
        if board.is_checkmate():
            return -99999
        if board.is_stalemate() or board.is_insufficient_material() or board.can_claim_draw():
            return 0
        score = 0
        for piece_type, base_val in PIECE_VALUES.items():
            pst = PST_MAP[piece_type]
            for sq in board.pieces(piece_type, chess.WHITE):
                score += base_val + pst[chess.square_mirror(sq)]
            for sq in board.pieces(piece_type, chess.BLACK):
                score -= base_val + pst[sq]
        return score if board.turn == chess.WHITE else -score

    def _score_move(self, board, move) -> int:
        score = 0
        if board.is_capture(move):
            victim = board.piece_type_at(move.to_square)
            attacker = board.piece_type_at(move.from_square)
            score += 10 * PIECE_VALUES.get(victim, 100) - PIECE_VALUES.get(attacker, 100) + 1000
        if move.promotion:
            score += PIECE_VALUES.get(move.promotion, 0) + 800
        if board.gives_check(move):
            score += 300
        return score

    def _ordered_moves(self, board, captures_only=False):
        moves = [m for m in board.legal_moves
                 if (not captures_only or board.is_capture(m) or m.promotion)]
        moves.sort(key=lambda m: self._score_move(board, m), reverse=True)
        return moves

    def _quiesce(self, board, alpha, beta) -> int:
        stand_pat = self.evaluate(board)
        if stand_pat >= beta:
            return beta
        if alpha < stand_pat:
            alpha = stand_pat
        for move in self._ordered_moves(board, captures_only=True):
            if time.perf_counter() > self._deadline:
                self._stop = True
                break
            board.push(move)
            score = -self._quiesce(board, -beta, -alpha)
            board.pop()
            if self._stop:
                return alpha
            if score >= beta:
                return beta
            if score > alpha:
                alpha = score
        return alpha

    def _negamax(self, board, depth, alpha, beta) -> int:
        if time.perf_counter() > self._deadline:
            self._stop = True
            return self.evaluate(board)
        if board.is_game_over():
            return self.evaluate(board)
        if depth <= 0:
            return self._quiesce(board, alpha, beta)
        best = -100000
        for move in self._ordered_moves(board):
            board.push(move)
            score = -self._negamax(board, depth - 1, -beta, -alpha)
            board.pop()
            if self._stop:
                break
            if score > best:
                best = score
            if score > alpha:
                alpha = score
            if alpha >= beta:
                break
        return best

    def find_best_move(self, board, time_limit=1.5, max_depth=4) -> Optional[chess.Move]:
        legal_moves = self._ordered_moves(board)
        if not legal_moves:
            return None
        if len(legal_moves) == 1:
            return legal_moves[0]
        self._stop = False
        self._deadline = time.perf_counter() + max(0.2, time_limit)
        best_move = legal_moves[0]
        for depth in range(1, max_depth + 1):
            current_best, current_best_score = None, -100000
            alpha, beta = -100000, 100000
            for move in legal_moves:
                board.push(move)
                score = -self._negamax(board, depth - 1, -beta, -alpha)
                board.pop()
                if self._stop:
                    break
                if score > current_best_score:
                    current_best_score, current_best = score, move
                if score > alpha:
                    alpha = score
            if not self._stop and current_best is not None:
                best_move = current_best
                legal_moves.remove(best_move)
                legal_moves.insert(0, best_move)
            else:
                break
        return best_move


# ============================================================================
# SCREEN READING: empty / white piece / black piece for each square
# ============================================================================
EMPTY, WHITE_P, BLACK_P = 0, 1, 2
_OBS_NAME = {EMPTY: "empty", WHITE_P: "a white piece", BLACK_P: "a black piece"}


class PositionError(Exception):
    """The position on screen could not be matched to a known game."""


class NeedGameText(PositionError):
    """The screen can't be identified on its own; a FEN / PGN is needed once (the bot then learns)."""


def square_to_rc(sq: int, white_bottom: bool):
    """python-chess square -> (row, col) on screen (row 0 = top)."""
    f, r = chess.square_file(sq), chess.square_rank(sq)
    return (7 - r, f) if white_bottom else (r, 7 - f)


def rc_to_square(row: int, col: int, white_bottom: bool) -> int:
    return chess.square(col, 7 - row) if white_bottom else chess.square(7 - col, row)


# ---- feature extraction ------------------------------------------------------------------------
# Different sites / themes need different settings, so the settings live in a FeatureParams tuple and
# the bot AUTOMATICALLY picks the best one for your screen (see smart_match).
#   dev    : how different from the square's own background a pixel must be to count as "piece"
#   win    : centre part of the square used for the occupancy test
#   bwin   : centre part used to measure piece brightness (only piece pixels are measured)
#   corner : size of the corner patches that measure the square's background (pieces never reach them)
#   cmode  : "abs" = absolute brightness, "rel" = brightness relative to the square's background
FeatureParams = collections.namedtuple("FeatureParams", "dev win bwin corner cmode")
DEFAULT_FP = FeatureParams(40, (0.18, 0.82), (0.30, 0.70), 0.14, "abs")
FP_VARIANTS = [
    DEFAULT_FP,
    FeatureParams(40, (0.18, 0.82), (0.30, 0.70), 0.14, "rel"),
    FeatureParams(60, (0.18, 0.82), (0.30, 0.70), 0.14, "abs"),
    FeatureParams(60, (0.18, 0.82), (0.30, 0.70), 0.14, "rel"),
    FeatureParams(25, (0.18, 0.82), (0.30, 0.70), 0.14, "abs"),
    FeatureParams(80, (0.18, 0.82), (0.30, 0.70), 0.14, "abs"),
    FeatureParams(40, (0.25, 0.75), (0.35, 0.65), 0.14, "abs"),
    FeatureParams(40, (0.25, 0.75), (0.35, 0.65), 0.14, "rel"),
    FeatureParams(40, (0.10, 0.90), (0.25, 0.75), 0.08, "abs"),
    FeatureParams(60, (0.25, 0.75), (0.30, 0.70), 0.10, "rel"),
    FeatureParams(60, (0.12, 0.88), (0.25, 0.75), 0.08, "abs"),
]

_RC_CACHE = {}


def _rc_squares(white_bottom: bool) -> np.ndarray:
    if white_bottom not in _RC_CACHE:
        _RC_CACHE[white_bottom] = np.array(
            [rc_to_square(r, c, white_bottom) for r in range(8) for c in range(8)])
    return _RC_CACHE[white_bottom]


def _to_sq(arr_rc, white_bottom: bool):
    out = np.zeros(64, np.float64)
    out[_rc_squares(white_bottom)] = np.asarray(arr_rc, np.float64).reshape(-1)
    return out.tolist()


def _squares(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    sh, sw = h // 8, w // 8
    if sh < 6 or sw < 6:
        raise PositionError("The calibrated board area is too small. Re-calibrate the board corners.")
    a = img[:sh * 8, :sw * 8, :3]
    sq = a.reshape(8, sh, 8, sw, 3).transpose(0, 2, 1, 3, 4)          # (row, col, y, x, rgb)
    st = max(1, min(sh, sw) // 48)                                     # sub-sample big boards for speed
    return sq[:, :, ::st, ::st]


def features_rc(img: np.ndarray, fp: FeatureParams = DEFAULT_FP):
    """Vectorised: (metric[8][8], brightness[8][8]) in screen layout (row 0 = top)."""
    sq = _squares(img)
    sh, sw = sq.shape[2], sq.shape[3]
    k = max(2, int(min(sh, sw) * fp.corner))
    corners = np.concatenate([
        sq[:, :, :k, :k].reshape(8, 8, -1, 3), sq[:, :, :k, -k:].reshape(8, 8, -1, 3),
        sq[:, :, -k:, :k].reshape(8, 8, -1, 3), sq[:, :, -k:, -k:].reshape(8, 8, -1, 3)], axis=2)
    ref = np.median(corners.astype(np.float32), axis=2)                # (8, 8, 3) background colour per square
    y0, y1, x0, x1 = int(fp.win[0] * sh), max(int(fp.win[1] * sh), int(fp.win[0] * sh) + 1), \
        int(fp.win[0] * sw), max(int(fp.win[1] * sw), int(fp.win[0] * sw) + 1)
    cen = sq[:, :, y0:y1, x0:x1].astype(np.float32)
    dev = np.abs(cen - ref[:, :, None, None, :]).max(axis=-1) > fp.dev
    metric = dev.mean(axis=(2, 3))
    by0, by1, bx0, bx1 = int(fp.bwin[0] * sh), max(int(fp.bwin[1] * sh), int(fp.bwin[0] * sh) + 1), \
        int(fp.bwin[0] * sw), max(int(fp.bwin[1] * sw), int(fp.bwin[0] * sw) + 1)
    bc = sq[:, :, by0:by1, bx0:bx1].astype(np.float32)
    lum = bc.mean(axis=-1)
    bdev = np.abs(bc - ref[:, :, None, None, :]).max(axis=-1) > fp.dev
    cnt = bdev.sum(axis=(2, 3))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        med = np.nanmedian(np.where(bdev, lum, np.nan).reshape(8, 8, -1), axis=2)
    fallback = np.median(lum.reshape(8, 8, -1), axis=2)
    bright = np.where(cnt >= 6, med, fallback)
    if fp.cmode == "rel":
        bright = bright - ref.mean(axis=2)
    return metric, bright


def extract_features(img: np.ndarray, white_bottom: bool, fp: FeatureParams = DEFAULT_FP):
    """Returns (metric[64], brightness[64]) indexed by python-chess square."""
    m, b = features_rc(img, fp)
    return _to_sq(m, white_bottom), _to_sq(b, white_bottom)


def _largest_gap_threshold(values, lo=None, hi=None):
    """Midpoint of the widest gap between neighbouring sorted values (optionally within lo..hi)."""
    vals = sorted(values)
    best_gap, thr = -1.0, None
    for a, b in zip(vals, vals[1:]):
        mid = (a + b) / 2.0
        if lo is not None and mid < lo:
            continue
        if hi is not None and mid > hi:
            continue
        if b - a > best_gap:
            best_gap, thr = b - a, mid
    return thr, best_gap


def auto_thresholds(metric, bright):
    """Learns the empty/occupied and white/black thresholds from the screen alone
    (no known position needed)."""
    occ_thr, _ = _largest_gap_threshold(metric, 0.05, 0.92)
    if occ_thr is None:
        raise PositionError("Could not find the pieces on the board. Re-calibrate the corners "
                            "and make sure nothing covers the board.")
    occupied = [s for s in range(64) if metric[s] > occ_thr]
    if len(occupied) < 2:
        raise PositionError("Could not see enough pieces. Re-calibrate the board corners.")
    col_thr, gap = _largest_gap_threshold([bright[s] for s in occupied])
    if col_thr is None:
        raise PositionError("Could not tell white and black pieces apart.")
    return occ_thr, col_thr


def refine_thresholds(metric, bright, board: chess.Board):
    """With the true contents known, centre the thresholds between the two groups.
    None only if occupied / empty squares can't be separated."""
    occ = [metric[s] for s in chess.SQUARES if board.piece_at(s)]
    emp = [metric[s] for s in chess.SQUARES if not board.piece_at(s)]
    wb = [bright[s] for s in chess.SQUARES if board.piece_at(s) and board.color_at(s) == chess.WHITE]
    bb = [bright[s] for s in chess.SQUARES if board.piece_at(s) and board.color_at(s) == chess.BLACK]
    if not occ or not emp or not wb or not bb or min(occ) - max(emp) < 0.02:
        return None
    if min(wb) > max(bb):
        col = (min(wb) + max(bb)) / 2.0
    else:                                   # colour overlap: still usable as a tie-breaker only
        col = (float(np.mean(wb)) + float(np.mean(bb))) / 2.0
    return (min(occ) + max(emp)) / 2.0, col


def classify_from_features(metric, bright, occ_thr, col_thr):
    obs = []
    for sq in chess.SQUARES:
        if metric[sq] <= occ_thr:
            obs.append(EMPTY)
        else:
            obs.append(WHITE_P if bright[sq] > col_thr else BLACK_P)
    return tuple(obs)


def classify(img, white_bottom, occ_thr, col_thr, fp: FeatureParams = DEFAULT_FP):
    metric, bright = extract_features(img, white_bottom, fp)
    return classify_from_features(metric, bright, occ_thr, col_thr)


def board_signature(board: chess.Board):
    out = []
    for sq in chess.SQUARES:
        p = board.piece_at(sq)
        out.append(EMPTY if p is None else (WHITE_P if p.color == chess.WHITE else BLACK_P))
    return tuple(out)


def obs_masks(obs):
    wm = bm = 0
    for sq, v in enumerate(obs):
        if v == WHITE_P:
            wm |= 1 << sq
        elif v == BLACK_P:
            bm |= 1 << sq
    return wm, bm


def occ_mask(obs) -> int:
    m = 0
    for sq, v in enumerate(obs):
        if v != EMPTY:
            m |= 1 << sq
    return m


def occ_equal(obs, sig) -> bool:
    """Same squares occupied? (colour is deliberately ignored - it is the least reliable signal)"""
    return occ_mask(obs) == occ_mask(sig)


def _colour_mismatches(board: chess.Board, obs) -> int:
    wm, bm = obs_masks(obs)
    occ = board.occupied
    return bin(((board.occupied_co[chess.WHITE] ^ wm) | (board.occupied_co[chess.BLACK] ^ bm)) & occ).count("1")


def find_move_from_observation(board: chess.Board, obs) -> Optional[chess.Move]:
    """The legal move whose resulting OCCUPANCY matches the screen. Piece colour is only used to break
    ties (queen preferred for promotions), so a single misread colour can't lose the game."""
    target = occ_mask(obs)
    matches = []
    for mv in list(board.legal_moves):
        board.push(mv)
        if board.occupied == target:
            matches.append((_colour_mismatches(board, obs), mv))
        board.pop()
    if not matches:
        return None
    matches.sort(key=lambda t: (t[0], 0 if t[1].promotion in (None, chess.QUEEN) else 1))
    return matches[0][1]


def describe_mismatch(root: chess.Board, obs, limit=4) -> str:
    exp = board_signature(root)
    diffs = []
    for sq in chess.SQUARES:
        if (obs[sq] != EMPTY) != (exp[sq] != EMPTY):
            diffs.append(f"{chess.square_name(sq)}: screen shows {_OBS_NAME[obs[sq]]}, expected {_OBS_NAME[exp[sq]]}")
    if not diffs:
        return "none"
    more = f" (+{len(diffs) - limit} more)" if len(diffs) > limit else ""
    return "; ".join(diffs[:limit]) + more


# ---- candidate positions + automatic troubleshooting match --------------------------------------
class _Cands:
    def __init__(self):
        self.depth, self.wm, self.bm, self.paths = [], [], [], []


_CAND_CACHE = {}


def enumerate_candidates(root: chess.Board, max_plies: int, cap: int = 60000) -> _Cands:
    """`root` and every position up to `max_plies` legal moves after it."""
    key = (root.fen(), max_plies)
    if key in _CAND_CACHE:
        return _CAND_CACHE[key]
    work = root.copy(stack=False)
    out = _Cands()
    path: List[chess.Move] = []

    def dfs(left, d):
        if len(out.paths) >= cap:
            return
        out.depth.append(d)
        out.wm.append(work.occupied_co[chess.WHITE])
        out.bm.append(work.occupied_co[chess.BLACK])
        out.paths.append(list(path))
        if left == 0:
            return
        for mv in list(work.legal_moves):
            work.push(mv)
            path.append(mv)
            dfs(left - 1, d + 1)
            path.pop()
            work.pop()

    dfs(max_plies, 0)
    out.depth = np.array(out.depth)
    if len(_CAND_CACHE) > 4:
        _CAND_CACHE.clear()
    _CAND_CACHE[key] = out
    return out


def score_candidates(wmasks, bmasks, metric, bright):
    """Vectorised check of many candidate positions against the measured screen features.
    For each candidate (white / black occupancy bitboards):
      gap  = how cleanly the occupancy metric separates its occupied from its empty squares,
      sep  = same for piece brightness (white pieces vs black pieces),
      msep = smooth version of gap (mean occupied minus mean empty) - lets alignment searches climb.
    A candidate that really is on screen has a large positive gap and sep."""
    sh = np.arange(64, dtype=np.uint64)
    one = np.uint64(1)
    wb = ((np.array(wmasks, dtype=np.uint64)[:, None] >> sh) & one).astype(bool)
    bb = ((np.array(bmasks, dtype=np.uint64)[:, None] >> sh) & one).astype(bool)
    occ = wb | bb
    m, br, big = np.asarray(metric, np.float64), np.asarray(bright, np.float64), 1e9
    gap = np.where(occ, m, big).min(axis=1) - np.where(~occ, m, -big).max(axis=1)
    sep = np.where(wb, br, big).min(axis=1) - np.where(bb, br, -big).max(axis=1)
    n_occ = np.maximum(occ.sum(axis=1), 1)
    n_emp = np.maximum((~occ).sum(axis=1), 1)
    msep = (np.where(occ, m, 0).sum(axis=1) / n_occ) - (np.where(~occ, m, 0).sum(axis=1) / n_emp)
    return gap, sep, msep


def crop_window(raw, win):
    """raw = (image, ox, oy, board_w_px, board_h_px); win = (L, T, W, H) as fractions of the board."""
    img, ox, oy, bw, bh = raw
    L, T, W, H = win
    x0, y0 = int(round(ox + L * bw)), int(round(oy + T * bh))
    x1, y1 = int(round(x0 + W * bw)), int(round(y0 + H * bh))
    if x0 < 0 or y0 < 0 or x1 > img.shape[1] or y1 > img.shape[0] or x1 - x0 < 48 or y1 - y0 < 48:
        return None
    return img[y0:y1, x0:x1]


def _window_candidates():
    yield (0.0, 0.0, 1.0, 1.0)
    for s in (-0.06, -0.03, 0.03, 0.06):                    # board area a little too big / too small
        yield (-s / 2, -s / 2, 1 + s, 1 + s)
    for w in ((0.04, 0.0, 0.96, 0.96), (0.0, 0.0, 0.96, 0.96), (0.0, 0.04, 0.96, 0.96), (0.04, 0.04, 0.96, 0.96),
              (-0.04, -0.04, 1.04, 1.04), (0.0, 0.0, 1.04, 1.04)):                    # labels inside / outside the area
        yield w


class Match:
    pass


def smart_match(raw, white_bottom: bool, root: chess.Board, max_plies: int):
    """AUTOMATIC TROUBLESHOOTING. Finds `root` (or a position up to `max_plies` moves after it) on screen,
    searching: board-area alignment, both board orientations (a flipped 'I play' side is corrected) and
    several detection settings - scoring each against the known contents. Returns (Match or None, report)."""
    cands = enumerate_candidates(root, max_plies)

    def evaluate(wb, win, fp):
        img = crop_window(raw, win)
        if img is None:
            return None
        try:
            m_rc, b_rc = features_rc(img, fp)
        except PositionError:
            return None
        metric, bright = _to_sq(m_rc, wb), _to_sq(b_rc, wb)
        gap, sep, msep = score_candidates(cands.wm, cands.bm, metric, bright)
        total = (np.minimum(gap / 0.10, 1.0) + 0.3 * np.clip(gap, -1, 1) + 0.6 * np.clip(msep, -1, 1)
                 + 0.5 * np.clip(sep / 60.0, -1, 1) - 0.04 * cands.depth)
        if wb != white_bottom:
            total = total - 0.30                                    # prefer the side you selected
        i = int(np.argmax(total))
        return float(total[i]), i, float(gap[i]), float(sep[i]), metric, bright

    best = None

    def consider(wb, win, fp):
        nonlocal best
        r = evaluate(wb, win, fp)
        if r is not None and (best is None or r[0] > best[0]):
            best = (r[0], r[1], r[2], r[3], r[4], r[5], wb, win, fp)

    consider(white_bottom, (0.0, 0.0, 1.0, 1.0), DEFAULT_FP)
    good = best is not None and best[2] >= 0.10 and best[3] >= 25
    if not good:
        for wb in (white_bottom, not white_bottom):             # 1) alignment x orientation
            for win in _window_candidates():
                consider(wb, win, DEFAULT_FP)
        for step in (0.03, 0.015, 0.0075):                      # 2) hill-climb the alignment (position + size)
            for _ in range(10):
                if best is None:
                    break
                before = best[0]
                _, _, _, _, _, _, wb0, (L, T, W, H), fp0 = best
                h = step / 2.0
                for nb in ((L + step, T, W, H), (L - step, T, W, H), (L, T + step, W, H), (L, T - step, W, H),
                           (L, T, W + step, H + step), (L, T, W - step, H - step),
                           (L - h, T - h, W + step, H + step), (L + h, T + h, W - step, H - step)):
                    consider(wb0, nb, fp0)
                if best[0] <= before + 1e-9:
                    break
        if best is not None:                                    # 3) best detection settings for this screen
            _, _, _, _, _, _, wb0, win0, _ = best
            for fp in FP_VARIANTS[1:]:
                consider(wb0, win0, fp)

    if best is not None and best[2] >= 0.03:
        total, i, gap, sep, metric, bright, wb, win, fp = best
        m = Match()
        m.board = root.copy()
        for mv in cands.paths[i]:
            m.board.push(mv)
        m.plies, m.white_bottom, m.win, m.fp = len(cands.paths[i]), wb, win, fp
        refined = refine_thresholds(metric, bright, m.board)
        if refined is None:
            return None, "found the position but couldn't separate pieces from squares"
        m.occ_thr, m.col_thr = refined
        m.flipped = wb != white_bottom
        return m, "ok"

    # nothing fitted: report the closest candidate and what differs
    try:
        img = crop_window(raw, (0.0, 0.0, 1.0, 1.0))
        metric, bright = extract_features(img, white_bottom)
        occ_thr, col_thr = auto_thresholds(metric, bright)
        obs = classify_from_features(metric, bright, occ_thr, col_thr)
        target = occ_mask(obs)
        sh = np.arange(64, dtype=np.uint64)
        xor = np.array([w | b for w, b in zip(cands.wm, cands.bm)], dtype=np.uint64) ^ np.uint64(target)
        ham = ((xor[:, None] >> sh) & np.uint64(1)).sum(axis=1)
        j = int(np.argmin(ham + 0.5 * cands.depth))
        b = root.copy()
        for mv in cands.paths[j]:
            b.push(mv)
        return None, f"closest candidate still differs on {int(ham[j])} square(s): {describe_mismatch(b, obs)}"
    except Exception as exc:
        return None, f"nothing on screen fitted ({exc})"


# ---- debug picture (so problems can be inspected) ------------------------------------------------
def save_png(path: str, arr: np.ndarray):
    a = np.ascontiguousarray(arr[:, :, :3].astype(np.uint8))
    h, w = a.shape[:2]
    raw = b"".join(b"\x00" + a[y].tobytes() for y in range(h))

    def chunk(tag, data):
        c = struct.pack(">I", len(data)) + tag + data
        return c + struct.pack(">I", zlib.crc32(tag + data) & 0xffffffff)

    png = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)) \
        + chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b"")
    with open(path, "wb") as f:
        f.write(png)


def write_debug_images(raw, white_bottom: bool, folder: str):
    """Saves what the bot sees: the board with a grid, and a marker on every square it thinks holds a
    piece (white dot = white piece, black dot = black piece)."""
    try:
        img = crop_window(raw, (0.0, 0.0, 1.0, 1.0))
        save_png(os.path.join(folder, "chess_bot_debug_raw.png"), img)
        metric, bright = extract_features(img, white_bottom)
        occ_thr, col_thr = auto_thresholds(metric, bright)
        obs = classify_from_features(metric, bright, occ_thr, col_thr)
        out = img[:, :, ::-1].copy()                                  # BGR -> RGB
        h, w = out.shape[:2]
        for i in range(9):
            out[min(h - 1, int(i * h / 8)), :] = (255, 0, 255)
            out[:, min(w - 1, int(i * w / 8))] = (255, 0, 255)
        sq = max(6, min(h, w) // 8 // 4)
        for s in chess.SQUARES:
            if obs[s] == EMPTY:
                continue
            r, c = square_to_rc(s, white_bottom)
            y, x = int(r * h / 8) + 3, int(c * w / 8) + 3
            out[y:y + sq, x:x + sq] = (255, 255, 255) if obs[s] == WHITE_P else (0, 0, 0)
            out[y:y + 1, x:x + sq] = (255, 0, 0)
            out[y + sq - 1:y + sq, x:x + sq] = (255, 0, 0)
        path = os.path.join(folder, "chess_bot_debug.png")
        save_png(path, out)
        return path
    except Exception:
        return None


# ============================================================================
# PIECE LEARNING + SCREEN-ONLY RECOGNITION  (lets a running game be continued with no FEN / PGN)
# ============================================================================
TEMPLATE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chess_bot_pieces.json")
FEAT_N = 12
_SYMBOLS = "PNBRQKpnbrqk"


def square_image(img: np.ndarray, sq: int, white_bottom: bool) -> np.ndarray:
    h, w = img.shape[:2]
    row, col = square_to_rc(sq, white_bottom)
    y0, y1 = int(round(row * h / 8)), int(round((row + 1) * h / 8))
    x0, x1 = int(round(col * w / 8)), int(round((col + 1) * w / 8))
    return img[y0:y1, x0:x1]


def square_bg(sq_img: np.ndarray) -> np.ndarray:
    h, w = sq_img.shape[:2]
    k = max(2, int(min(h, w) * 0.14))
    corners = np.concatenate([
        sq_img[:k, :k].reshape(-1, 3), sq_img[:k, -k:].reshape(-1, 3),
        sq_img[-k:, :k].reshape(-1, 3), sq_img[-k:, -k:].reshape(-1, 3),
    ]).astype(np.float32)
    return np.median(corners, axis=0)


def piece_feature(sq_img: np.ndarray) -> np.ndarray:
    """Size- and square-colour-independent description of what is drawn on a square:
    a 12x12 map of 'differs from the background' plus a 12x12 map of lighter/darker-than-background."""
    sq_img = sq_img[:, :, :3]
    h, w = sq_img.shape[:2]
    ref = square_bg(sq_img)
    crop = sq_img[int(0.06 * h):int(0.94 * h), int(0.06 * w):int(0.94 * w)].astype(np.float32)
    lum = crop.mean(axis=2) - float(ref.mean())
    dev = (np.abs(crop - ref).max(axis=2) > 40).astype(np.float32)
    ys = np.linspace(0, lum.shape[0], FEAT_N + 1).astype(int)
    xs = np.linspace(0, lum.shape[1], FEAT_N + 1).astype(int)
    out = np.empty((2, FEAT_N, FEAT_N), np.float32)
    for i in range(FEAT_N):
        y0, y1 = ys[i], max(ys[i + 1], ys[i] + 1)
        for j in range(FEAT_N):
            x0, x1 = xs[j], max(xs[j + 1], xs[j] + 1)
            out[0, i, j] = dev[y0:y1, x0:x1].mean()
            out[1, i, j] = np.clip(lum[y0:y1, x0:x1].mean() / 128.0, -1, 1) * 0.5
    return out.reshape(-1)


def _rms(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


class PieceTemplates:
    """Remembers what the 12 piece kinds look like. Several PIECE SETS (themes / sites) are kept side by
    side; the one that fits the screen is chosen automatically. Learned from any known position."""
    CAP = 14
    MAX_PROFILES = 6
    SAME_SET = 0.09

    def __init__(self):
        self.profiles: List[dict] = []

    @classmethod
    def load(cls):
        t = cls()
        try:
            with open(TEMPLATE_FILE, "r", encoding="utf-8") as f:
                d = json.load(f)
            profs = d.get("profiles")
            if profs is None and "samples" in d:
                profs = [d["samples"]]
            for p in profs or []:
                t.profiles.append({k: [np.asarray(v, np.float32) / 1000.0 for v in p.get(k, [])] for k in _SYMBOLS})
        except Exception:
            pass
        return t

    def save(self):
        try:
            data = {"v": 2, "profiles": [{k: [[int(round(x * 1000)) for x in s] for s in v] for k, v in p.items()}
                                         for p in self.profiles]}
            with open(TEMPLATE_FILE, "w", encoding="utf-8") as f:
                json.dump(data, f)
        except Exception:
            pass

    @staticmethod
    def _complete(p) -> bool:
        return all(p[k] for k in _SYMBOLS)

    def ready(self) -> bool:
        return any(self._complete(p) for p in self.profiles)

    @staticmethod
    def _fit(p, items) -> float:
        """Mean distance of known (key, feature) pairs to profile p's samples of the same key."""
        ds = [min(_rms(f, g) for g in p[k]) for k, f in items if p[k]]
        return sum(ds) / len(ds) if ds else 9.0

    def learn(self, img: np.ndarray, white_bottom: bool, pieces) -> bool:
        """pieces: {square: 'P'|'n'|...} known to be on the board in this image."""
        items = [(key, piece_feature(square_image(img, sq, white_bottom))) for sq, key in pieces.items()]
        if not items:
            return False
        target, best = None, self.SAME_SET
        for p in self.profiles:
            d = self._fit(p, items)
            if d < best:
                target, best = p, d
        changed = False
        if target is None:
            target = {k: [] for k in _SYMBOLS}
            self.profiles.append(target)
            if len(self.profiles) > self.MAX_PROFILES:
                self.profiles.pop(0)
            changed = True
        for key, f in items:
            lst = target[key]
            if lst and min(_rms(f, g) for g in lst) < 0.04:
                continue
            lst.append(f)
            changed = True
            if len(lst) > self.CAP:
                lst.pop(0)
        return changed

    @staticmethod
    def rank(profile, feat: np.ndarray):
        out = [(min(_rms(feat, g) for g in profile[k]), k) for k in _SYMBOLS if profile[k]]
        out.sort()
        return out

    def pick_profile(self, feats):
        """The complete piece set that best explains these piece images -> (profile, mean distance)."""
        best, best_d = None, 9.0
        for p in self.profiles:
            if not self._complete(p):
                continue
            d = sum(self.rank(p, f)[0][0] for f in feats) / max(1, len(feats))
            if d < best_d:
                best, best_d = p, d
        return best, best_d


def detect_last_mover(img: np.ndarray, white_bottom: bool, assign) -> Optional[bool]:
    """Looks for the highlighted from/to squares of the last move. The colour of the piece standing
    on a highlighted square = the side that just moved. Returns True (white) / False (black) / None."""
    bg = {sq: square_bg(square_image(img, sq, white_bottom)) for sq in range(64)}
    med = {}
    for par in (0, 1):
        sqs = [s for s in range(64) if (chess.square_file(s) + chess.square_rank(s)) % 2 == par]
        med[par] = np.median(np.stack([bg[s] for s in sqs]), axis=0)
    hl = [s for s in range(64)
          if float(np.abs(bg[s] - med[(chess.square_file(s) + chess.square_rank(s)) % 2]).max()) > 28]
    if not 1 <= len(hl) <= 4:
        return None
    on = [s for s in hl if s in assign]
    if not on:
        return None
    cols = {assign[s].isupper() for s in on}
    return cols.pop() if len(cols) == 1 else None


def recognise_position(img: np.ndarray, white_bottom: bool, templates: PieceTemplates, my_color: bool,
                       fp: FeatureParams = DEFAULT_FP):
    """Rebuilds the whole position from the screen alone using the learned piece sets.
    Returns (board, occ_thr, col_thr, turn_was_guessed)."""
    metric, bright = extract_features(img, white_bottom, fp)
    occ_thr, col_thr = auto_thresholds(metric, bright)
    obs = classify_from_features(metric, bright, occ_thr, col_thr)
    feats = {sq: piece_feature(square_image(img, sq, white_bottom)) for sq in chess.SQUARES if obs[sq] != EMPTY}
    if not 2 <= len(feats) <= 32:
        raise PositionError("The number of pieces seen on the board doesn't look right. "
                            "Re-calibrate the corners and make sure nothing covers the board.")
    profile, mean_d = templates.pick_profile(list(feats.values()))
    if profile is None or mean_d > 0.22:
        raise PositionError("These pieces don't look like any piece set the bot has learned (new theme?). "
                            "Press Test Read on a NEW game, or paste a FEN once with this theme - it learns it.")
    ranked = {sq: templates.rank(profile, f) for sq, f in feats.items()}
    assign = {sq: r[0][1] for sq, r in ranked.items()}

    def dist_of(sq, key):
        for d, k in ranked[sq]:
            if k == key:
                return d
        return 9.0

    def alt(sq, colour_white, banned):
        for d, k in ranked[sq]:
            if k.isupper() == colour_white and k not in banned:
                return k
        return None

    for king in ("K", "k"):
        white = king.isupper()
        sqs = [s for s, k in assign.items() if k == king]
        mine = [s for s, k in assign.items() if k.isupper() == white]
        if not mine:
            raise PositionError("Could not find all the pieces. Re-calibrate the board corners.")
        if not sqs:
            assign[min(mine, key=lambda s: dist_of(s, king))] = king
        elif len(sqs) > 1:
            keep = min(sqs, key=lambda s: dist_of(s, king))
            for s in sqs:
                if s != keep:
                    assign[s] = alt(s, white, {king}) or ("P" if white else "p")
    for s, k in list(assign.items()):                       # no pawns on the first / last rank
        if k in "Pp" and chess.square_rank(s) in (0, 7):
            assign[s] = alt(s, k.isupper(), {"P", "p"}) or k

    tmp = chess.Board(None)
    for s, k in assign.items():
        tmp.set_piece_at(s, chess.Piece.from_symbol(k))
    placement = tmp.board_fen()
    cast = ""
    if assign.get(chess.E1) == "K":
        cast += ("K" if assign.get(chess.H1) == "R" else "") + ("Q" if assign.get(chess.A1) == "R" else "")
    if assign.get(chess.E8) == "k":
        cast += ("k" if assign.get(chess.H8) == "r" else "") + ("q" if assign.get(chess.A8) == "r" else "")
    cast = cast or "-"

    mover = detect_last_mover(img, white_bottom, assign)
    order = [not mover] if mover is not None else [my_color, not my_color]
    for turn in order:
        b = chess.Board(f"{placement} {'w' if turn else 'b'} {cast} - 0 1")
        if b.is_valid():
            refined = refine_thresholds(metric, bright, b)
            if refined:
                occ_thr, col_thr = refined
            return b, occ_thr, col_thr, mover is None
    raise PositionError("The recognised pieces don't form a legal position (a piece may have been misread). "
                        "Try again in a moment, or use the FEN / PGN button.")


# ============================================================================
# GAME TEXT: FEN / PGN / move list  ->  chess.Board    (and back, for saving)
# ============================================================================
_FEN_RE = re.compile(
    r"([rnbqkpRNBQKP1-8]+(?:/[rnbqkpRNBQKP1-8]+){7})"
    r"(?:\s+([wb])\s+([KQkq\-]+)\s+([a-h][36]|-)(?:\s+(\d+)\s+(\d+))?)?")


def parse_game_text(text: str, default_turn: bool) -> chess.Board:
    """Accepts a FEN (full or just the piece placement), a PGN, SAN or UCI move lists, or a FEN
    followed by moves. `default_turn` is used only when the FEN has no side-to-move."""
    text = (text or "").strip()
    if not text:
        raise ValueError("Nothing entered.")
    board = None
    m = _FEN_RE.search(text)
    if m:
        placement = m.group(1)
        if m.group(2):
            fen = " ".join([placement, m.group(2), m.group(3), m.group(4),
                            m.group(5) or "0", m.group(6) or "1"])
        else:
            fen = f"{placement} {'w' if default_turn == chess.WHITE else 'b'} KQkq - 0 1"
        try:
            board = chess.Board(fen)
        except ValueError as exc:
            raise ValueError(f"That FEN could not be read: {exc}") from exc
        board.castling_rights = board.clean_castling_rights()
        if not board.is_valid():
            # Self-correct common problems instead of rejecting a pasted FEN: wrong side to move,
            # a bogus en-passant square. Only a missing / duplicated king is fatal.
            fixed = False
            for turn in (board.turn, not board.turn):
                board.turn = turn
                board.ep_square = None
                if board.is_valid():
                    fixed = True
                    break
            if not fixed:
                fatal = chess.STATUS_NO_WHITE_KING | chess.STATUS_NO_BLACK_KING | \
                    chess.STATUS_TOO_MANY_KINGS | chess.STATUS_TOO_MANY_WHITE_PIECES | chess.STATUS_TOO_MANY_BLACK_PIECES
                if board.status() & fatal:
                    raise ValueError("That position is not legal: each side needs exactly one king and at most "
                                     "16 pieces.")
                board.turn = chess.WHITE if default_turn == chess.WHITE else chess.BLACK
        text = text[:m.start()] + " " + text[m.end():]
    else:
        board = chess.Board()

    text = re.sub(r"\{[^}]*\}", " ", text)            # {comments}
    text = re.sub(r";[^\n]*", " ", text)              # ; comments
    text = re.sub(r"\[[^\]]*\]", " ", text)           # [PGN tags]
    while "(" in text:                                # (variations)
        new = re.sub(r"\([^()]*\)", " ", text)
        if new == text:
            break
        text = new
    text = re.sub(r"\$\d+", " ", text)                # $1 annotations
    text = re.sub(r"(?<![A-Za-z0-9])\d+\s*\.(?:\.\.)?", " ", text)    # 12.  /  12...
    tokens = [t for t in text.split() if t not in ("1-0", "0-1", "1/2-1/2", "*")]

    for n, tok in enumerate(tokens, 1):
        tok = tok.rstrip("!?")
        try:
            board.push_san(tok)
            continue
        except ValueError:
            pass
        try:
            mv = chess.Move.from_uci(tok.lower())
            if mv in board.legal_moves:
                board.push(mv)
                continue
        except ValueError:
            pass
        raise ValueError(f"Could not apply move #{n}: '{tok}' is not legal in that position.")
    return board


def board_to_text(board: chess.Board) -> str:
    """Readable text (optional FEN line + numbered SAN moves) that parse_game_text understands."""
    root = board.root()
    tmp = root.copy(stack=False)
    parts = []
    for i, mv in enumerate(board.move_stack):
        san = tmp.san(mv)
        if tmp.turn == chess.WHITE:
            parts.append(f"{tmp.fullmove_number}. {san}")
        elif i == 0:
            parts.append(f"{tmp.fullmove_number}... {san}")
        else:
            parts.append(san)
        tmp.push(mv)
    moves = " ".join(parts)
    if root.fen() == chess.STARTING_FEN:
        return moves
    return root.fen() + ("\n" + moves if moves else "")


# ============================================================================
# THEME  (colours / glass recipe taken from the Nemesis control panel)
# ============================================================================
C_BG, C_PANEL, C_PANEL2, C_LINE = "#05051a", "#0a0a26", "#11113a", "#2a2a70"
C_TEXT, C_CREAM, C_DIM = "#e9e7ff", "#f3efcf", "#8e92c0"
C_ACCENT, C_PURPLE, C_MAGENTA = "#5ff3ea", "#a855f7", "#d77cf0"
C_GOOD, C_WARN, C_BAD = "#5ff3a0", "#ffd166", "#ff6b8b"
C_FIELD = (10, 8, 32)


def hex_rgb(h: str):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def rgb_hex(c) -> str:
    return "#%02x%02x%02x" % tuple(int(max(0, min(255, round(v)))) for v in c)


def mix_hex(a: str, b: str, t: float) -> str:
    ca, cb = hex_rgb(a), hex_rgb(b)
    return rgb_hex([ca[i] + (cb[i] - ca[i]) * t for i in range(3)])


def _smooth(x):
    return x * x * (3 - 2 * x)


def _sdf_xy(px, py, w, h, r):
    """Signed distance to a rounded rectangle centred on the origin."""
    qx = np.abs(px) - (w / 2.0 - r)
    qy = np.abs(py) - (h / 2.0 - r)
    return np.hypot(np.maximum(qx, 0), np.maximum(qy, 0)) + np.minimum(np.maximum(qx, qy), 0) - r


def _blur_axis(a, r, axis):
    a = np.moveaxis(a, axis, 0)
    pad = np.concatenate([np.repeat(a[:1], r + 1, 0), a, np.repeat(a[-1:], r, 0)], 0)
    c = np.cumsum(pad, axis=0, dtype=np.float64)
    out = (c[2 * r + 1:] - c[:-(2 * r + 1)]) / (2 * r + 1)
    return np.moveaxis(out.astype(np.float32), 0, axis)


def box_blur(img, r, passes=3):
    out = img
    for _ in range(passes):
        out = _blur_axis(_blur_axis(out, r, 0), r, 1)
    return out


def to_photo(arr) -> tk.PhotoImage:
    a = np.clip(arr, 0, 255).astype(np.uint8)
    h, w = a.shape[:2]
    return tk.PhotoImage(data=b"P6 %d %d 255\n" % (w, h) + a.tobytes(), format="PPM")


def make_background(w, h, S, seed=7):
    """Dark cosmic backdrop: nebula, glow with light rays, stars, vignette."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    img = np.empty((h, w, 3), np.float32)
    img[:] = (5, 5, 26)

    def screen(col, a):
        for c in range(3):
            s = col[c] / 255.0
            base = img[..., c] / 255.0
            img[..., c] = 255.0 * (1 - (1 - base) * (1 - s * a))

    def radial(cx, cy, rx, ry, stop=1.0):
        d = np.sqrt(((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2) / stop
        return np.clip(1 - d, 0, 1)

    screen((130, 70, 255), 0.55 * radial(w * .20, h * .25, w * .62, h * .30, .7))
    screen((60, 220, 255), 0.38 * radial(w * .86, h * .18, w * .55, h * .26, .7))
    screen((255, 80, 160), 0.34 * radial(w * .60, h * .86, w * .70, h * .32, .7))
    screen((190, 170, 255), 0.30 * radial(w * .5, h * .07, w * .80, h * .22))
    screen((255, 247, 215), 0.38 * radial(w * .5, h * .07, w * .22, h * .07))

    cx, cy = w * .5, h * .07                      # light rays fanning out of the glow
    deg = np.degrees(np.arctan2(yy - cy, xx - cx)) % 11.0
    ray = np.clip(1 - np.abs(deg - 0.7) / 0.9, 0, 1)
    fade = np.clip(1 - np.hypot(xx - cx, yy - cy) / (0.9 * h), 0, 1) ** 1.6
    screen((255, 246, 214), 0.20 * ray * fade)

    palette = [(207, 210, 255), (255, 255, 255), (153, 255, 255), (217, 184, 255)]
    for _ in range(int(w * h / 5200)):
        sx, sy = rng.uniform(5, w - 6), rng.uniform(5, h - 6)
        col = palette[int(rng.integers(0, len(palette)))]
        sigma = (0.9 if rng.random() < .18 else 0.6) * max(1.0, S * .8)
        bright = float(rng.uniform(.45, 1.0))
        x0, x1, y0, y1 = int(sx) - 3, int(sx) + 4, int(sy) - 3, int(sy) + 4
        gx, gy = np.meshgrid(np.arange(x0, x1) + .5 - sx, np.arange(y0, y1) + .5 - sy)
        g = bright * np.exp(-(gx ** 2 + gy ** 2) / (2 * sigma ** 2))
        for c in range(3):
            patch = img[y0:y1, x0:x1, c] / 255.0
            img[y0:y1, x0:x1, c] = 255.0 * (1 - (1 - patch) * (1 - col[c] / 255.0 * g))

    t = yy / max(1, h - 1)                         # soft darkening top -> bottom
    a = np.interp(t, [0, .55, 1], [.12, .34, .60])
    img = img * (1 - a[..., None]) + np.array([5, 3, 14], np.float32) * a[..., None]
    d = np.sqrt(((xx - w * .5) / (w * .85)) ** 2 + ((yy - h * .4) / (h * .8)) ** 2)
    a = .55 * np.clip((d - .45) / .55, 0, 1)
    img = img * (1 - a[..., None]) + np.array([3, 2, 10], np.float32) * a[..., None]
    return img.astype(np.float32)


class Scene:
    """Pre-rendered window picture: backdrop + frosted-glass cards (real blur behind them)."""

    def __init__(self, w, h, S, seed=7):
        self.w, self.h, self.S = w, h, S
        self.img = make_background(w, h, S, seed)
        self.blur = box_blur(self.img, max(3, int(6 * S)), 3)

    def shadow(self, x, y, w, h, r, off=12, blur=26, a=.38):
        S = self.S
        reach = int((blur + off) * S) + 2
        X0, X1 = max(0, int(x - reach)), min(self.w, int(x + w + reach))
        Y0, Y1 = max(0, int(y - reach)), min(self.h, int(y + h + reach))
        yy, xx = np.mgrid[Y0:Y1, X0:X1].astype(np.float32)
        px, py = xx + .5 - (x + w / 2.0), yy + .5 - (y + h / 2.0) - off * S
        sd = _sdf_xy(px, py, w, h, r)
        sh = a * (1 - _smooth(np.clip(sd / (blur * S), 0, 1)))
        self.img[Y0:Y1, X0:X1] *= (1 - sh)[..., None]

    def glass(self, x, y, w, h, r, tint=(9, 7, 28), tint_a=.46, g0=.09, g1=.02, line_a=.16, hi_a=.22):
        S = self.S
        X0, X1 = max(0, int(x - 2)), min(self.w, int(x + w + 2))
        Y0, Y1 = max(0, int(y - 2)), min(self.h, int(y + h + 2))
        yy, xx = np.mgrid[Y0:Y1, X0:X1].astype(np.float32)
        px, py = xx + .5 - (x + w / 2.0), yy + .5 - (y + h / 2.0)
        sd = _sdf_xy(px, py, w, h, r)
        bd = self.blur[Y0:Y1, X0:X1]
        lum = bd.mean(axis=2, keepdims=True)
        bd = lum + (bd - lum) * 1.5                                   # saturate(1.6)
        card = bd * (1 - tint_a) + np.array(tint, np.float32) * tint_a
        ang = math.radians(145)                                       # linear-gradient(145deg, ...)
        dx, dy = math.sin(ang), -math.cos(ang)
        t = np.clip((px * dx + py * dy) / (abs(w * dx) + abs(h * dy)) + .5, 0, 1)
        wa = (g0 + (g1 - g0) * t)[..., None]
        card = card * (1 - wa) + 255.0 * wa
        bw = max(1.0, S)
        ring = np.clip(bw / 2 + .5 - np.abs(sd + bw / 2), 0, 1) * line_a            # 1px border
        card = card * (1 - ring[..., None]) + 255.0 * ring[..., None]
        top = np.clip(1 - (py + h / 2.0) / (r * 1.3 + 2 * S), 0, 1)                 # inset top sheen
        hi = np.clip(bw / 2 + .5 - np.abs(sd + bw * 1.5), 0, 1) * hi_a * top
        card = card * (1 - hi[..., None]) + 255.0 * hi[..., None]
        a = np.clip(.5 - sd, 0, 1)[..., None]
        region = self.img[Y0:Y1, X0:X1]
        region[:] = region * (1 - a) + card * a

    def inset(self, x, y, w, h, r, flat=None, fill=(8, 6, 28), fill_a=.42, line_a=.14):
        """A recessed field (like the inputs / status box)."""
        S = self.S
        X0, X1 = max(0, int(x - 2)), min(self.w, int(x + w + 2))
        Y0, Y1 = max(0, int(y - 2)), min(self.h, int(y + h + 2))
        yy, xx = np.mgrid[Y0:Y1, X0:X1].astype(np.float32)
        px, py = xx + .5 - (x + w / 2.0), yy + .5 - (y + h / 2.0)
        sd = _sdf_xy(px, py, w, h, r)
        region = self.img[Y0:Y1, X0:X1]
        if flat is not None:
            body = np.broadcast_to(np.array(flat, np.float32), region.shape)
            fa = np.clip(.5 - sd, 0, 1)[..., None]
        else:
            body = np.broadcast_to(np.array(fill, np.float32), region.shape)
            fa = (np.clip(.5 - sd, 0, 1) * fill_a)[..., None]
        region[:] = region * (1 - fa) + body * fa
        bw = max(1.0, S)
        ring = (np.clip(bw / 2 + .5 - np.abs(sd + bw / 2), 0, 1) * line_a)[..., None]
        region[:] = region * (1 - ring) + 255.0 * ring

    def hline(self, x, y, w, a=.10):
        t = max(1, int(self.S))
        seg = self.img[int(y):int(y) + t, int(x):int(x + w)]
        seg[:] = seg * (1 - a) + 255.0 * a

    def glow(self, cx, cy, rx, ry, color, a):
        yy, xx = np.mgrid[0:self.h, 0:self.w].astype(np.float32)
        g = np.clip(1 - np.sqrt(((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2), 0, 1) ** 1.5 * a
        for c in range(3):
            base = self.img[..., c] / 255.0
            self.img[..., c] = 255.0 * (1 - (1 - base) * (1 - color[c] / 255.0 * g))

    def crop(self, x, y, w, h):
        return self.img[int(y):int(y + h), int(x):int(x + w)].copy()


def _over(region, color, alpha):
    a = alpha[..., None]
    region *= (1 - a)
    region += np.asarray(color, np.float32) * a


def render_button(base, w, h, pad, S, kind, state="normal", on=False, enabled=True, tone=None):
    """Draws one glass button / chip / switch on top of a crop of the scene."""
    out = base.copy()
    H2, W2 = out.shape[:2]
    yy, xx = np.mgrid[0:H2, 0:W2].astype(np.float32)
    px, py = xx + .5 - W2 / 2.0, yy + .5 - H2 / 2.0
    hover = state in ("hover", "press")
    bw = max(1.0, S)
    accent, magenta, bad = np.array(hex_rgb(C_ACCENT), np.float32), np.array(hex_rgb(C_MAGENTA), np.float32), \
        np.array(hex_rgb(C_BAD), np.float32)

    def glow(sd, col, strength):
        g = strength * (1 - _smooth(np.clip(np.maximum(sd, 0) / pad, 0, 1)))
        _over(out, col, g)

    if kind == "switch":
        r = h / 2.0
        sd = _sdf_xy(px, py, w, h, r)
        a = np.clip(.5 - sd, 0, 1)
        if on:
            glow(sd, accent, .35)
        _over(out, accent if on else np.array([255, 255, 255], np.float32), a * (.26 if on else (.13 if hover else .08)))
        ring = np.clip(bw / 2 + .5 - np.abs(sd + bw / 2), 0, 1)
        _over(out, accent if on else np.array([255, 255, 255], np.float32), ring * (.95 if on else .22))
        kr = r - 3 * S
        kx = (W2 / 2.0 + w / 2.0 - r) if on else (W2 / 2.0 - w / 2.0 + r)
        kd = np.hypot(xx + .5 - kx, py) - kr
        _over(out, accent if on else np.array([205, 205, 235], np.float32), np.clip(.5 - kd, 0, 1) * (1.0 if on else .85))
        return out

    r = h / 2.0 if kind in ("chip", "pill") else (14 * S if kind == "primary" else 12 * S)
    sd = _sdf_xy(px, py, w, h, r)
    a = np.clip(.5 - sd, 0, 1)
    ring = np.clip(bw / 2 + .5 - np.abs(sd + bw / 2), 0, 1)

    if kind == "primary":
        t = np.clip((px + w / 2.0) / w, 0, 1)
        stops = np.array([hex_rgb("#f6f1cd"), hex_rgb(C_ACCENT), hex_rgb(C_MAGENTA)], np.float32)
        t1 = np.clip(t / .55, 0, 1)[..., None]
        t2 = np.clip((t - .55) / .45, 0, 1)[..., None]
        fill = np.where(t[..., None] < .55, stops[0] * (1 - t1) + stops[1] * t1, stops[1] * (1 - t2) + stops[2] * t2)
        if hover:
            fill = np.minimum(fill * 1.08, 255)
        if state == "press":
            fill = fill * .94
        if not enabled:
            gray = fill.mean(axis=2, keepdims=True) * .6
            fill = gray + (fill - gray) * .3
        else:
            glow(sd, magenta if hover else accent, .40 if hover else .25)
        _over(out, fill, a)
        return out

    fill_col = np.array([255, 255, 255], np.float32)
    fill_a = .08
    border, border_a = np.array([255, 255, 255], np.float32), .16
    if kind == "accent":
        fill_col, fill_a, border, border_a = accent, .07, accent, .55
    if kind == "pill":
        border, border_a = {"good": (np.array(hex_rgb(C_GOOD), np.float32), .5),
                            "bad": (bad, .5), "warn": (np.array(hex_rgb(C_WARN), np.float32), .4),
                            "accent": (accent, .5)}.get(tone, (border, border_a))
    if hover:
        fill_a = {"accent": .17}.get(kind, .14)
        border, border_a = (bad if kind == "danger" else accent), .85
        if kind == "chip":
            border, border_a = np.array([120, 120, 210], np.float32), .8
    if state == "press":
        fill_a += .06
    if on:
        fill_col, fill_a, border, border_a = accent, .16, accent, .9
        glow(sd, accent, .35)
    elif hover and kind in ("accent", "ghost", "danger"):
        glow(sd, bad if kind == "danger" else accent, .18)
    k = 1.0 if enabled else .45
    _over(out, fill_col, a * fill_a * k)
    _over(out, border, ring * border_a * k)
    return out


# ============================================================================
# CANVAS WIDGETS (everything is drawn over the pre-rendered glass scene)
# ============================================================================
class CButton:
    """Glass button / chip / switch drawn on the canvas, with hover / pressed / on states."""

    def __init__(self, ui, x, y, w, h, text, command, kind="ghost", font=None, on=False, toggle=False, tone=None):
        self.ui, self.x, self.y, self.w, self.h = ui, x, y, w, h
        self.kind, self.command, self.on, self.toggle, self.tone = kind, command, on, toggle, tone
        self.state, self.enabled = "normal", True
        self.pad = ui.u(8)
        self.base = ui.scene.crop(x - self.pad, y - self.pad, w + 2 * self.pad, h + 2 * self.pad)
        self._cache = {}
        c = ui.canvas
        self.item = c.create_image(x - self.pad, y - self.pad, image=self._img(), anchor="nw")
        self.txt = None
        if text:
            self.txt = c.create_text(x + w / 2.0, y + h / 2.0, text=text, font=font or ui.f_btn,
                                     fill=self._text_color(), anchor="center", state="disabled")
        for ev, fn in (("<Enter>", self._motion), ("<Motion>", self._motion), ("<Leave>", self._leave),
                       ("<ButtonPress-1>", self._press), ("<ButtonRelease-1>", self._release)):
            c.tag_bind(self.item, ev, fn)

    # -- rendering
    def _img(self):
        key = (self.state, self.on, self.enabled, self.tone)
        if key not in self._cache:
            self._cache[key] = to_photo(render_button(self.base, self.w, self.h, self.pad, self.ui.S, self.kind,
                                                      self.state, self.on, self.enabled, self.tone))
        return self._cache[key]

    def _text_color(self):
        k = self.kind
        if not self.enabled:
            return mix_hex(C_DIM, C_BG, .35)
        if k == "primary":
            return "#0a0a26"
        if k == "pill":
            return {"good": C_GOOD, "bad": C_BAD, "warn": C_WARN, "accent": C_ACCENT}.get(self.tone, C_DIM)
        if self.on or k == "accent":
            return C_ACCENT
        if self.state in ("hover", "press"):
            return C_BAD if k == "danger" else (C_TEXT if k == "chip" else C_ACCENT)
        return C_DIM if k == "chip" else C_TEXT

    def refresh(self):
        self.ui.canvas.itemconfigure(self.item, image=self._img())
        if self.txt is not None:
            self.ui.canvas.itemconfigure(self.txt, fill=self._text_color())

    # -- API
    def set_on(self, on):
        self.on = on
        self.refresh()

    def set_enabled(self, enabled):
        self.enabled = enabled
        self.state = "normal"
        self.refresh()

    def set_text(self, text):
        if self.txt is not None:
            self.ui.canvas.itemconfigure(self.txt, text=text)

    def set_tone(self, tone):
        self.tone = tone
        self.refresh()

    # -- events (only the real button rectangle reacts, not the glow padding)
    def _inside(self, ev):
        return self.x <= ev.x <= self.x + self.w and self.y <= ev.y <= self.y + self.h

    def _motion(self, ev):
        if not self.enabled or self.kind == "pill":
            return
        inside = self._inside(ev)
        want = "hover" if inside else "normal"
        if self.state == "press" and inside:
            want = "press"
        if want != self.state:
            self.state = want
            self.refresh()
        self.ui.canvas.configure(cursor="hand2" if inside else "")

    def _leave(self, _ev):
        if self.state != "normal":
            self.state = "normal"
            self.refresh()
        self.ui.canvas.configure(cursor="")

    def _press(self, ev):
        if self.enabled and self.kind != "pill" and self._inside(ev):
            self.state = "press"
            self.refresh()

    def _release(self, ev):
        if self.state != "press":
            return
        self.state = "hover" if self._inside(ev) else "normal"
        if self._inside(ev):
            if self.toggle:
                self.on = not self.on
            self.refresh()
            if self.command:
                self.command(self.on) if self.toggle else self.command()
        else:
            self.refresh()


class CSlider:
    """Glass slider: gradient fill, glowing knob."""

    def __init__(self, ui, x, y, w, lo, hi, step, var, on_change=None):
        self.ui, self.x, self.y, self.w = ui, x, y, w
        self.lo, self.hi, self.step, self.var, self.on_change = lo, hi, step, var, on_change
        self.drag = False
        S, c = ui.S, ui.canvas
        self.track = c.create_line(x, y, x + w, y, width=max(3, int(5 * S)), capstyle="round", fill="#24234f")
        self.segs = []
        n = 28
        for i in range(n):
            x0, x1 = x + w * i / n, x + w * (i + 1) / n
            col = mix_hex(C_ACCENT, C_MAGENTA, i / (n - 1))
            self.segs.append((x1, c.create_line(x0, y, x1 + 1, y, width=max(3, int(5 * S)), fill=col)))
        self.halo = c.create_oval(0, 0, 1, 1, fill=mix_hex(C_ACCENT, "#14123a", .78), outline="")
        self.knob = c.create_oval(0, 0, 1, 1, fill=C_CREAM, outline=C_ACCENT, width=max(2, int(2 * S)))
        self._place()
        c.bind("<ButtonPress-1>", self._press, add="+")
        c.bind("<B1-Motion>", self._move, add="+")
        c.bind("<ButtonRelease-1>", self._release, add="+")

    def _frac(self):
        return (float(self.var.get()) - self.lo) / (self.hi - self.lo)

    def _place(self):
        S, c = self.ui.S, self.ui.canvas
        kx = self.x + self.w * max(0.0, min(1.0, self._frac()))
        r, rh = 8 * S, 14 * S
        c.coords(self.knob, kx - r, self.y - r, kx + r, self.y + r)
        c.coords(self.halo, kx - rh, self.y - rh, kx + rh, self.y + rh)
        for x1, item in self.segs:
            c.itemconfigure(item, state="hidden" if x1 > kx + 1 else "normal")

    def _hit(self, ev):
        S = self.ui.S
        return (self.x - 14 * S <= ev.x <= self.x + self.w + 14 * S) and abs(ev.y - self.y) <= 16 * S

    def _set_from(self, ev):
        f = max(0.0, min(1.0, (ev.x - self.x) / float(self.w)))
        v = self.lo + (self.hi - self.lo) * f
        v = round(round(v / self.step) * self.step, 2)
        self.var.set(max(self.lo, min(self.hi, v)))
        self._place()
        if self.on_change:
            self.on_change(self.var.get())

    def _press(self, ev):
        if self._hit(ev):
            self.drag = True
            self._set_from(ev)

    def _move(self, ev):
        if self.drag:
            self._set_from(ev)

    def _release(self, _ev):
        self.drag = False


class GlassUI:
    """Canvas + fonts + helpers shared by the main window and the dialog."""

    def __init__(self, parent, scene, S):
        self.S, self.scene = S, scene
        self.canvas = tk.Canvas(parent, width=scene.w, height=scene.h, bg=C_BG, highlightthickness=0, bd=0)
        self.canvas.pack()
        self._bg = to_photo(scene.img)
        self.canvas.create_image(0, 0, image=self._bg, anchor="nw")
        fams = set(tkfont.families(parent))

        def pick(cands, fallback):
            for c in cands:
                if c in fams:
                    return c
            return fallback

        pix = pick(("Pixelify Sans", "Silkscreen", "Press Start 2P", "Consolas", "Menlo", "Courier New"), "Courier")
        mono = pick(("Space Mono", "Cascadia Mono", "Consolas", "Menlo", "DejaVu Sans Mono", "Courier New"), "Courier")
        body = pick(("Space Grotesk", "Segoe UI", "SF Pro Text", "Helvetica Neue", "Ubuntu", "DejaVu Sans"), "Helvetica")
        mk = lambda fam, px, weight="normal": tkfont.Font(root=parent, family=fam, size=-max(8, int(px * S)), weight=weight)
        self.f_title = mk(pix, 21, "bold")
        self.f_primary = mk(pix, 17, "bold")
        self.f_head = mk(mono, 11, "bold")
        self.f_label = mk(mono, 10.5)
        self.f_btn = mk(mono, 11, "bold")
        self.f_body = mk(body, 13)
        self.f_bodyb = mk(body, 13, "bold")
        self.f_hint = mk(body, 11.5)
        self.f_mono = mk(mono, 12)
        self.keep = []

    def u(self, v):
        return int(round(v * self.S))

    def tracked(self, x, y, text, font, fill, tracking=0, anchor="w", colors=None):
        """Text with letter-spacing (and optionally a per-letter colour gradient)."""
        t = tracking * self.S
        widths = [font.measure(ch) for ch in text]
        total = sum(widths) + t * (len(text) - 1)
        cx = x - total if anchor == "e" else (x - total / 2.0 if anchor == "c" else x)
        ids = []
        for i, ch in enumerate(text):
            ids.append(self.canvas.create_text(cx, y, text=ch, font=font, fill=colors[i] if colors else fill, anchor="w"))
            cx += widths[i] + t
        return ids

    def card_title(self, x, y, w, title, right=None):
        self.tracked(x + self.u(16), y + self.u(17), "\u2726", self.f_head, C_ACCENT)
        self.tracked(x + self.u(32), y + self.u(17), title, self.f_head, C_ACCENT, tracking=3)
        if right is not None:
            return self.canvas.create_text(x + w - self.u(16), y + self.u(17), text=right, font=self.f_label,
                                           fill=C_DIM, anchor="e")


# ============================================================================
# "CONTINUE GAME" DIALOG
# ============================================================================
class ContinueDialog:
    def __init__(self, app, prefill: str, info: str):
        self.result: Optional[str] = None
        self.prefill = prefill
        S = app.S
        top = self.top = tk.Toplevel(app.root)
        top.title("Continue game")
        top.configure(bg=C_BG)
        top.resizable(False, False)
        top.transient(app.root)
        try:
            top.attributes("-topmost", True)
        except Exception:
            pass
        u = lambda v: int(round(v * S))
        W, H = u(472), u(432)
        sc = Scene(W, H, S, seed=11)
        cx, cy, cw, ch, r = u(14), u(14), u(444), u(404), u(18)
        sc.shadow(cx, cy, cw, ch, r)
        sc.glass(cx, cy, cw, ch, r)
        sc.hline(cx + u(1), cy + u(34), cw - u(2))
        fx, fy, fw, fh = cx + u(14), cy + u(112), cw - u(28), u(196)
        sc.inset(fx, fy, fw, fh, u(12), flat=C_FIELD)
        ui = self.ui = GlassUI(top, sc, S)
        ui.card_title(cx, cy, cw, "CONTINUE GAME")
        ui.canvas.create_text(cx + u(16), cy + u(46),
                              text="Paste a FEN, a PGN, or just the moves (e4 e5 Nf3 ...). The position on "
                                   "screen must match - the bot checks it and can catch up a few moves "
                                   "played since.",
                              font=ui.f_hint, fill=C_DIM, anchor="nw", width=cw - u(32))
        self.text = tk.Text(top, bg=rgb_hex(C_FIELD), fg=C_TEXT, insertbackground=C_ACCENT, relief="flat", bd=0,
                            highlightthickness=0, font=ui.f_mono, wrap="word", padx=u(6), pady=u(4),
                            selectbackground="#3b3b95", selectforeground="#ffffff", undo=True)
        ui.canvas.create_window(fx + u(8), fy + u(8), window=self.text, width=fw - u(16), height=fh - u(16), anchor="nw")
        self.text.insert("1.0", prefill)
        ui.canvas.create_text(cx + u(16), fy + fh + u(10), text=info, font=ui.f_hint, fill=C_DIM, anchor="nw",
                              width=cw - u(32))
        by, bh = cy + ch - u(14) - u(40), u(40)
        self.b_last = CButton(ui, cx + u(14), by, u(132), bh, "USE LAST GAME", self._use_last, kind="ghost")
        self.b_last.set_enabled(bool(prefill))
        CButton(ui, cx + u(14) + u(132) + u(8), by, u(92), bh, "CANCEL", self._cancel, kind="ghost")
        CButton(ui, cx + u(14) + u(132) + u(8) + u(92) + u(8), by, cw - u(28) - u(132) - u(92) - u(16), bh,
                "CONTINUE", self._ok, kind="primary", font=ui.f_primary)
        top.protocol("WM_DELETE_WINDOW", self._cancel)
        top.bind("<Escape>", lambda e: self._cancel())
        top.bind("<Control-Return>", lambda e: self._ok())
        app.root.update_idletasks()
        top.geometry("+%d+%d" % (app.root.winfo_rootx() + u(24), app.root.winfo_rooty() + u(24)))
        self.text.focus_set()
        try:
            top.wait_visibility()
            top.grab_set()
        except tk.TclError:
            pass
        app.root.wait_window(top)

    def _use_last(self):
        self.text.delete("1.0", "end")
        self.text.insert("1.0", self.prefill)

    def _ok(self):
        self.result = self.text.get("1.0", "end").strip()
        self.top.destroy()

    def _cancel(self):
        self.result = None
        self.top.destroy()


# ============================================================================
# APPLICATION (Tkinter UI + worker thread; all UI updates go through a queue)
# ============================================================================
ICON_PNG_B64 = (
    "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"
)


class Aegis:
    W, H = 404, 658

    def __init__(self, root: tk.Tk):
        self.root = root
        root.title("Aegis")
        try:
            self._icon = tk.PhotoImage(data=ICON_PNG_B64)
            root.iconphoto(True, self._icon)
        except Exception:
            pass
        root.configure(bg=C_BG)
        root.resizable(False, False)
        root.attributes("-topmost", True)
        try:
            self.S = max(1.0, round(root.winfo_fpixels("1i") / 96.0, 2))
        except Exception:
            self.S = 1.0
        self._dark_titlebar()

        self.ui_queue: queue.Queue = queue.Queue()
        self.stop_event = threading.Event()
        self.worker: Optional[threading.Thread] = None

        self.region = None          # (x1, y1, x2, y2) in pyautogui coordinates
        self.scale = 1.0            # screenshot pixels per pyautogui unit
        self.mon_left = 0
        self.mon_top = 0
        self.stockfish_path = self._find_stockfish()
        # True only while the engine in use is the bundled exe found automatically (not chosen by hand)
        self.sf_auto = bool(self.stockfish_path) and os.path.basename(self.stockfish_path) == self.SF_EXE \
            and self.stockfish_path == self._find_bundled_stockfish()
        self.game_board: Optional[chess.Board] = None   # kept after Stop so the game can be continued
        self.templates = PieceTemplates.load()          # learned look of your piece sets
        self.win = (0.0, 0.0, 1.0, 1.0)                 # board-area fine alignment (found automatically)
        self.fp = DEFAULT_FP                            # detection settings (found automatically)
        self.eff_region = None                          # region actually used for reading + clicking
        self._turn_uncertain = False                    # True after resuming: verify whose move it is first
        self._turn_guessed = False
        self.run_white_bottom = True

        self.side_var = tk.StringVar(value="white")
        self.think_var = tk.DoubleVar(value=1.5)
        self.promo_click_var = tk.BooleanVar(value=True)
        self._think = 1.5
        self._promo = True
        self.think_var.trace_add("write", lambda *a: self._cache_vars())
        self.promo_click_var.trace_add("write", lambda *a: self._cache_vars())

        self._pill_state = "setup"
        self._build_ui()
        root.protocol("WM_DELETE_WINDOW", self._on_close)
        self._poll_queue()
        self._animate()

    def _cache_vars(self):
        try:
            self._think = float(self.think_var.get())
            self._promo = bool(self.promo_click_var.get())
        except Exception:
            pass

    def _dark_titlebar(self):
        if not sys.platform.startswith("win"):
            return
        try:
            self.root.update()
            hwnd = ctypes.windll.user32.GetParent(self.root.winfo_id())
            val = ctypes.c_int(1)
            for attr in (20, 19):
                ctypes.windll.dwmapi.DwmSetWindowAttribute(hwnd, attr, ctypes.byref(val), ctypes.sizeof(val))
        except Exception:
            pass

    # ---------------------------------------------------------------- UI
    def _build_ui(self):
        S = self.S
        u = lambda v: int(round(v * S))
        W, H = u(self.W), u(self.H)
        sc = Scene(W, H, S, seed=7)

        cards = {                                     # (x, y, w, h) in design units
            "status": (14, 74, 376, 110),
            "settings": (14, 194, 376, 202),
            "controls": (14, 406, 376, 238),
        }
        sc.shadow(u(-24), u(-30), W + u(48), u(92), u(22), off=8, blur=22, a=.35)
        for x, y, w, h in cards.values():
            sc.shadow(u(x), u(y), u(w), u(h), u(18))
        sc.glow(u(34), u(31), u(34), u(34), (190, 170, 255), .55)
        sc.glow(u(96), u(26), u(70), u(16), (95, 243, 234), .16)
        sc.glass(u(-24), u(-30), W + u(48), u(92), u(22), tint=(12, 10, 36), tint_a=.38)
        sc.hline(0, u(61), W, .14)
        for x, y, w, h in cards.values():
            sc.glass(u(x), u(y), u(w), u(h), u(18))
            sc.hline(u(x) + u(1), u(y) + u(34), u(w) - u(2))
        sx, sy, sw, sh = cards["status"]
        sc.inset(u(sx + 14), u(sy + 44), u(sw - 28), u(54), u(12))

        ui = self.ui = GlassUI(self.root, sc, S)
        c = ui.canvas

        # ---- header
        self._star = c.create_text(u(34), u(31), text="\u2726", font=ui.f_title, fill="#ffffff", anchor="center")
        n = len("AEGIS")
        cols = [mix_hex("#f3efcf", C_ACCENT, i / (n - 1) * 2) if i / (n - 1) <= .5
                else mix_hex(C_ACCENT, C_MAGENTA, (i / (n - 1) - .5) * 2) for i in range(n)]
        ui.tracked(u(60), u(26), "AEGIS", ui.f_title, C_ACCENT, tracking=4, colors=cols)
        ui.tracked(u(61), u(46), "CHESS ENGINE \u00b7 AUTO PLAY", ui.f_label, C_DIM, tracking=1.2)
        pw, ph = u(104), u(26)
        self.pill = CButton(ui, u(self.W - 14) - pw, u(18), pw, ph, "SETUP", None, kind="pill", tone="warn",
                            font=ui.f_label)
        c.itemconfigure(self.pill.txt, anchor="w")
        c.coords(self.pill.txt, u(self.W - 14) - pw + u(28), u(18) + ph / 2.0)
        self._dot = c.create_oval(0, 0, 1, 1, fill=C_WARN, outline="")
        self._dot_xy = (u(self.W - 14) - pw + u(15), u(18) + ph / 2.0)
        self._place_dot(1.0)

        # ---- status card
        x, y, w, h = cards["status"]
        ui.card_title(u(x), u(y), u(w), "STATUS")
        self.status_item = c.create_text(u(x + 28), u(y + 52), text="Step 1: calibrate the board.", font=ui.f_bodyb,
                                         fill=C_WARN, anchor="nw", width=u(w - 56))

        # ---- settings card
        x, y, w, h = cards["settings"]
        self.engine_item = ui.card_title(u(x), u(y), u(w), "SETTINGS", right="")
        c.create_text(u(x + 16), u(y + 44 + 15), text="I PLAY (BOTTOM)", font=ui.f_label, fill=C_DIM, anchor="w")
        cw_, chh = u(86), u(30)
        self.chips = {}
        for i, side in enumerate(("white", "black")):
            cx = u(x + w - 14) - cw_ * (2 - i) - u(8) * (1 - i)
            self.chips[side] = CButton(ui, cx, u(y + 44), cw_, chh, side.upper(),
                                       lambda s=side: self._set_side(s), kind="chip",
                                       on=(self.side_var.get() == side))
        c.create_text(u(x + 16), u(y + 88 + 15), text="Click again to pick Queen on promotion", font=ui.f_body,
                      fill=C_TEXT, anchor="w")
        self.switch = CButton(ui, u(x + w - 14) - u(46), u(y + 88 + 3), u(46), u(24), "",
                              lambda on: self.promo_click_var.set(bool(on)), kind="switch",
                              on=self.promo_click_var.get(), toggle=True)
        c.create_text(u(x + 16), u(y + 126 + 8), text="THINK TIME", font=ui.f_label, fill=C_DIM, anchor="w")
        self.think_item = c.create_text(u(x + w - 16), u(y + 126 + 8), text="1.5 s", font=ui.f_head, fill=C_ACCENT,
                                        anchor="e")
        self.slider = CSlider(ui, u(x + 24), u(y + 126 + 34), u(w - 48), 0.3, 5.0, 0.1, self.think_var,
                              on_change=lambda v: c.itemconfigure(self.think_item, text="%.1f s" % float(v)))
        self.hint_item = c.create_text(u(x + 16), u(y + 172), text="", font=ui.f_hint, fill=C_DIM, anchor="nw",
                                       width=u(w - 32))
        self._update_engine_label()

        # ---- controls card
        x, y, w, h = cards["controls"]
        ui.card_title(u(x), u(y), u(w), "CONTROLS")
        half = u((w - 28 - 8) / 2.0)
        x0, x1 = u(x + 14), u(x + 14) + half + u(8)
        CButton(ui, x0, u(y + 44), half, u(36), "1. CALIBRATE BOARD", self.calibrate, kind="ghost")
        CButton(ui, x1, u(y + 44), half, u(36), "TEST READ", self.test_read, kind="ghost")
        CButton(ui, x0, u(y + 88), u(w - 28), u(46), "2. START / RESUME", self.start, kind="primary",
                font=ui.f_primary)
        CButton(ui, x0, u(y + 142), u(w - 28 - 8 - 104), u(38), "CONTINUE GAME", self.continue_game, kind="accent")
        CButton(ui, x0 + u(w - 28 - 104), u(y + 142), u(104), u(38), "FEN / PGN", self.paste_game, kind="ghost")
        CButton(ui, x0, u(y + 188), half, u(36), "STOP", self.stop, kind="danger")
        CButton(ui, x1, u(y + 188), half, u(36), "LOAD STOCKFISH", self._browse_stockfish, kind="ghost")

    def _set_side(self, side):
        self.side_var.set(side)
        for k, b in self.chips.items():
            b.set_on(k == side)

    def _place_dot(self, scale):
        r = 4 * self.S * scale
        cx, cy = self._dot_xy
        self.ui.canvas.coords(self._dot, cx - r, cy - r, cx + r, cy + r)

    def _set_pill(self, state: str):
        self._pill_state = state
        text, tone, col = {
            "setup": ("SETUP", "warn", C_WARN), "ready": ("READY", "accent", C_ACCENT),
            "running": ("RUNNING", "good", C_GOOD), "stopped": ("STOPPED", "bad", C_BAD),
            "done": ("GAME OVER", "accent", C_ACCENT), "error": ("ERROR", "bad", C_BAD),
        }[state]
        self.pill.set_text(text)
        self.pill.set_tone(tone)
        self.ui.canvas.itemconfigure(self._dot, fill=col)

    def _animate(self):
        t = time.time()
        k = (math.sin(t * 2.0) + 1) / 2
        self.ui.canvas.itemconfigure(self._star, fill=mix_hex("#cfc8ff", "#ffffff", k))
        if self._pill_state == "running":
            self._place_dot(.6 + .6 * ((math.sin(t * 6) + 1) / 2))
        self.root.after(110, self._animate)

    def _post(self, fn):
        self.ui_queue.put(fn)

    def _say(self, text: str):
        low = text.lower()
        if low.startswith("error") or "did not" in low or "failed" in low or "doesn't match" in low \
                or "couldn't" in low or "aborted" in low:
            col = C_BAD
        elif low.startswith(("running", "played", "opponent", "board calibrated", "continuing", "game over")):
            col = C_GOOD
        elif low.startswith(("thinking", "playing", "reading")):
            col = C_ACCENT
        else:
            col = C_WARN
        self._post(lambda: self.ui.canvas.itemconfigure(self.status_item, text=text, fill=col))

    def _poll_queue(self):
        try:
            while True:
                self.ui_queue.get_nowait()()
        except queue.Empty:
            pass
        self.root.after(80, self._poll_queue)

    def _on_close(self):
        self.stop_event.set()
        self.root.destroy()

    # ---------------------------------------------------------------- engines
    SF_EXE = "stockfish-windows-x86-64-universal.exe"

    @staticmethod
    def _app_dirs():
        """Folders searched for the bundled engine: next to the program / .exe, the PyInstaller
        bundle folder, and the current folder."""
        dirs = []
        for d in (getattr(sys, "_MEIPASS", None),
                  os.path.dirname(os.path.abspath(sys.executable)) if getattr(sys, "frozen", False) else None,
                  os.path.dirname(os.path.abspath(__file__)),
                  os.path.dirname(os.path.abspath(sys.argv[0])) if sys.argv and sys.argv[0] else None,
                  os.getcwd()):
            if d and d not in dirs:
                dirs.append(d)
        return dirs

    @classmethod
    def _find_bundled_stockfish(cls) -> Optional[str]:
        for d in cls._app_dirs():
            p = os.path.join(d, cls.SF_EXE)
            if os.path.isfile(p):
                return p
        return None

    @classmethod
    def _find_stockfish(cls) -> Optional[str]:
        bundled = cls._find_bundled_stockfish()
        if bundled:
            return bundled
        found = shutil.which("stockfish")
        if found:
            return found
        here = os.path.dirname(os.path.abspath(__file__))
        for p in ("/usr/games/stockfish", "/usr/local/bin/stockfish", "/opt/homebrew/bin/stockfish",
                  "C:\\stockfish\\stockfish.exe", os.path.join(here, "stockfish.exe"),
                  os.path.join(here, "stockfish")):
            if os.path.isfile(p):
                return p
        return None

    def _update_engine_label(self):
        c = self.ui.canvas
        if self.stockfish_path:
            c.itemconfigure(self.engine_item, text="STOCKFISH")
            if self.sf_auto:
                c.itemconfigure(self.hint_item, text="Powered by Stockfish 19")
            else:
                c.itemconfigure(self.hint_item, text=f"Engine: Stockfish ({os.path.basename(self.stockfish_path)})")
        else:
            c.itemconfigure(self.engine_item, text="BUILT-IN")
            c.itemconfigure(self.hint_item,
                            text="Engine: built-in alpha-beta (load Stockfish for much stronger play)")

    def _browse_stockfish(self):
        path = filedialog.askopenfilename(title="Select Stockfish executable")
        if path:
            self.stockfish_path = path
            self.sf_auto = False          # chosen by hand: the "Powered by" line goes away
            self._update_engine_label()

    # ---------------------------------------------------------------- calibration
    def calibrate(self):
        if self.worker and self.worker.is_alive():
            return
        threading.Thread(target=self._calibrate_worker, daemon=True).start()

    def _calibrate_worker(self):
        pts = []
        for label in ("TOP-LEFT corner of the 8x8 squares", "BOTTOM-RIGHT corner of the 8x8 squares"):
            for s in range(5, 0, -1):
                self._say(f"Hover the mouse over the {label} ... {s}")
                time.sleep(1)
            pts.append(pyautogui.position())
        (x1, y1), (x2, y2) = (pts[0].x, pts[0].y), (pts[1].x, pts[1].y)
        if x2 - x1 < 200 or y2 - y1 < 200:
            self._say("Calibration failed: board too small / corners in wrong order. Try again.")
            return
        with mss.mss() as sct:
            mon = sct.monitors[1]
            self.scale = mon["width"] / pyautogui.size().width
            self.mon_left, self.mon_top = mon["left"], mon["top"]
        self.region = (x1, y1, x2, y2)
        self.win, self.fp, self.eff_region = (0.0, 0.0, 1.0, 1.0), DEFAULT_FP, self.region
        self._say("Board calibrated. Press Start / Resume - it works for a new game or one already in progress.")
        self._post(lambda: self._set_pill("ready"))

    def _grab(self, sct) -> np.ndarray:
        """The board image (uses the automatically refined alignment)."""
        x1, y1, x2, y2 = self.eff_region or self.region
        s = self.scale
        mon = {"left": int(self.mon_left + x1 * s), "top": int(self.mon_top + y1 * s),
               "width": int((x2 - x1) * s), "height": int((y2 - y1) * s)}
        return np.asarray(sct.grab(mon))[:, :, :3]

    def _grab_raw(self, sct, pad=0.08):
        """Board plus a little margin, so the alignment can be searched. -> (img, ox, oy, board_w, board_h)."""
        x1, y1, x2, y2 = self.region
        bw, bh = x2 - x1, y2 - y1
        sw, sh = pyautogui.size()
        gx1, gy1 = max(0.0, x1 - pad * bw), max(0.0, y1 - pad * bh)
        gx2, gy2 = min(float(sw), x2 + pad * bw), min(float(sh), y2 + pad * bh)
        s = self.scale
        mon = {"left": int(self.mon_left + gx1 * s), "top": int(self.mon_top + gy1 * s),
               "width": int((gx2 - gx1) * s), "height": int((gy2 - gy1) * s)}
        img = np.asarray(sct.grab(mon))[:, :, :3]
        return img, (x1 - gx1) * s, (y1 - gy1) * s, bw * s, bh * s

    def _classify(self, sct, white_bottom, occ_thr, col_thr):
        return classify(self._grab(sct), white_bottom, occ_thr, col_thr, self.fp)

    def test_read(self):
        """Shows what the bot sees. A new game (or your last game) on screen is matched, the alignment and
        detection settings are tuned automatically, and your piece set is learned."""
        if not self.region:
            messagebox.showinfo("Calibrate first", "Click '1. Calibrate board' first.")
            return
        white_bottom = self.side_var.get() == "white"
        my_color = chess.WHITE if white_bottom else chess.BLACK
        try:
            with mss.mss() as sct:
                raw = self._grab_raw(sct)
            m, name, reports = self._match_known(raw, white_bottom, self._known_candidates(None))
            lines, note = [], ""
            if m:
                img = self._adopt(m, raw)
                wb = m.white_bottom
                self._learn_now(img, m.board, wb)
                for row in range(8):
                    lines.append(" ".join((m.board.piece_at(rc_to_square(row, col, wb)) or chess.Piece(1, True)).symbol()
                                          if m.board.piece_at(rc_to_square(row, col, wb)) else "."
                                          for col in range(8)))
                note = f"\n\nMatched: {name}. Piece set learned / updated."
                if m.flipped:
                    note += f"\nBoard was the other way round - switched to playing {'White' if wb else 'Black'}."
                if m.win != (0.0, 0.0, 1.0, 1.0) or m.fp != DEFAULT_FP:
                    note += "\nBoard alignment / detection settings tuned automatically."
            else:
                img = crop_window(raw, (0.0, 0.0, 1.0, 1.0))
                metric, bright = extract_features(img, white_bottom, self.fp)
                occ_thr, col_thr = auto_thresholds(metric, bright)
                obs = classify_from_features(metric, bright, occ_thr, col_thr)
                sym = {EMPTY: ".", WHITE_P: "W", BLACK_P: "B"}
                for row in range(8):
                    lines.append(" ".join(sym[obs[rc_to_square(row, col, white_bottom)]] for col in range(8)))
                note = "\n\nNot a new game / saved game - showing raw detection (W/B/.)."
                if self.templates.ready():
                    try:
                        b, _o, _c, _g = recognise_position(img, white_bottom, self.templates, my_color, self.fp)
                        lines = []
                        for row in range(8):
                            lines.append(" ".join(b.piece_at(rc_to_square(row, col, white_bottom)).symbol()
                                                  if b.piece_at(rc_to_square(row, col, white_bottom)) else "."
                                                  for col in range(8)))
                        note = "\n\nRecognised from the screen with a learned piece set.\nFEN: " + b.fen()
                    except PositionError as exc:
                        note += "\n" + str(exc)
                else:
                    note += "\nPiece set not learned yet - do this once on a NEW game."
        except Exception as exc:
            messagebox.showerror("Test Read", str(exc))
            return
        messagebox.showinfo("Detected board (top row = top of screen)", "\n".join(lines) + note)

    # ---------------------------------------------------------------- game save / resume
    def _save_game(self, board: chess.Board):
        try:
            data = {"root": board.root().fen(), "moves": [m.uci() for m in board.move_stack],
                    "side": "white" if self.run_white_bottom else "black"}
            with open(GAME_FILE, "w", encoding="utf-8") as f:
                json.dump(data, f)
        except Exception:
            pass

    @staticmethod
    def _load_saved_game():
        try:
            with open(GAME_FILE, "r", encoding="utf-8") as f:
                d = json.load(f)
            b = chess.Board(d["root"])
            for uci in d["moves"]:
                b.push_uci(uci)
            return b, d.get("side")
        except Exception:
            return None, None

    def _sync(self, board: chess.Board):
        self.game_board = board
        self._save_game(board)

    def continue_game(self):
        """Resume whatever is on screen - no FEN / PGN needed."""
        self._start_worker(None)

    def paste_game(self, note: str = ""):
        """Manual route: paste a FEN, PGN or move list."""
        if self.worker and self.worker.is_alive():
            return
        if not self.region:
            messagebox.showinfo("Calibrate first", "Click '1. Calibrate board' first.")
            return
        if self.game_board is not None:
            board, side = self.game_board, ("white" if self.run_white_bottom else "black")
        else:
            board, side = self._load_saved_game()
        prefill, info = "", "No saved game yet - paste a FEN, PGN or move list."
        if board is not None:
            prefill = board_to_text(board)
            n = len(board.move_stack)
            info = (f"Last game: {n} move{'s' if n != 1 else ''} played" +
                    (f", you were {side.capitalize()}." if side else ".") + " It is saved next to the script.")
        dlg = ContinueDialog(self, prefill, (note + "\n" + info).strip())
        if dlg.result is None:
            return
        if board is not None and side in ("white", "black") and dlg.result.strip() == prefill.strip():
            self._set_side(side)
        my_color = chess.WHITE if self.side_var.get() == "white" else chess.BLACK
        try:
            resume = parse_game_text(dlg.result, my_color)
        except ValueError as exc:
            messagebox.showerror("FEN / PGN", str(exc))
            return
        self._start_worker(resume)

    # ---------------------------------------------------------------- run / stop
    def start(self):
        self._start_worker(None)

    def _start_worker(self, resume_board: Optional[chess.Board]):
        if self.worker and self.worker.is_alive():
            if self.stop_event.is_set():
                self._say("Still stopping the previous run - try again in a second.")
            return
        if not self.region:
            messagebox.showinfo("Calibrate first", "Click '1. Calibrate board' first.")
            return
        white_bottom = self.side_var.get() == "white"
        self.run_white_bottom = white_bottom
        self._cache_vars()
        self.stop_event.clear()
        self.worker = threading.Thread(target=self._run_worker, args=(resume_board, white_bottom), daemon=True)
        self.worker.start()

    def stop(self):
        self.stop_event.set()
        self._say("Stopped. Press Continue game to pick the game up again - no moves needed.")
        self._post(lambda: self._set_pill("stopped"))

    def _sleep(self, secs: float):
        end = time.time() + secs
        while time.time() < end and not self.stop_event.is_set():
            time.sleep(0.03)

    def _sq_center(self, sq: int, white_bottom: bool):
        x1, y1, x2, y2 = self.eff_region or self.region
        row, col = square_to_rc(sq, white_bottom)
        return (x1 + (col + 0.5) * (x2 - x1) / 8, y1 + (row + 0.5) * (y2 - y1) / 8)

    def _park_mouse(self):
        """Move the cursor off the board so hover highlights don't confuse detection."""
        x1, y1, x2, y2 = self.eff_region or self.region
        sw, sh = pyautogui.size()
        pyautogui.moveTo(min(x2 + 40, sw - 5), min(max(y1 - 10, 5), sh - 5), duration=0.1)

    def _click_move(self, move: chess.Move, white_bottom: bool):
        fx, fy = self._sq_center(move.from_square, white_bottom)
        tx, ty = self._sq_center(move.to_square, white_bottom)
        pyautogui.moveTo(fx, fy, duration=0.12)
        pyautogui.click()
        self._sleep(0.12)
        pyautogui.moveTo(tx, ty, duration=0.12)
        pyautogui.click()
        if move.promotion and self._promo:
            self._sleep(0.35)
            pyautogui.click(tx, ty)  # the queen is offered on the promotion square itself
        self._sleep(0.15)
        self._park_mouse()

    def _run_worker(self, resume_board: Optional[chess.Board], white_bottom: bool):
        engine = None
        try:
            my_color = chess.WHITE if white_bottom else chess.BLACK
            builtin = BuiltinAlphaBetaEngine()
            self._post(lambda: self._set_pill("running"))

            with mss.mss() as sct:
                self._park_mouse()
                self._sleep(0.3)
                self._say("Reading the board...")
                try:
                    board, occ_thr, col_thr, how, white_bottom = self._find_game(sct, white_bottom, my_color, resume_board)
                    my_color = chess.WHITE if white_bottom else chess.BLACK
                except NeedGameText as exc:
                    self._say("Couldn't identify the pieces on my own yet - paste the FEN / moves once.")
                    self._post(lambda: self.root.after(600, lambda: self.paste_game(
                        "I can't identify the pieces yet - paste the FEN or moves once; I'll learn them.")))
                    self._post(lambda: self._set_pill("ready"))
                    return
                except PositionError as exc:
                    self._say("Couldn't recognise the position - see the message.")
                    self._post(lambda m=str(exc): messagebox.showerror("Aegis", m))
                    self._post(lambda: self._set_pill("error"))
                    return
                self._sync(board)

                if self.stockfish_path and os.path.isfile(self.stockfish_path):
                    try:
                        engine = chess.engine.SimpleEngine.popen_uci(self.stockfish_path)
                    except Exception as exc:
                        self._say(f"Stockfish failed to start ({exc}); using built-in engine.")

                n = len(board.move_stack)
                if how == "new game" and n == 0 and board.turn != my_color:
                    self._say("New game - waiting for White to move...")
                elif how == "new game" and n == 0:
                    self._say("Running. Watching the screen...")
                else:
                    self._say(f"Continuing ({how}). "
                              + ("Your move." if board.turn == my_color else "Waiting for opponent..."))
                self._last_learn_ply = len(board.move_stack)

                while not self.stop_event.is_set():
                    if board.is_game_over():
                        self._say(f"Game over: {board.result()}")
                        self._post(lambda: self._set_pill("done"))
                        break

                    if board.turn == my_color:
                        self._play_my_move(sct, board, white_bottom, occ_thr, col_thr, engine, builtin)
                    else:
                        self._wait_for_opponent(sct, board, white_bottom, occ_thr, col_thr)
                    self._maybe_learn(sct, board, white_bottom, occ_thr, col_thr)
        except pyautogui.FailSafeException:
            self._say("Aborted (mouse hit the screen corner).")
            self._post(lambda: self._set_pill("stopped"))
        except Exception as exc:
            self._say(f"Error: {exc}")
            self._post(lambda: self._set_pill("error"))
        finally:
            if engine is not None:
                try:
                    engine.quit()
                except Exception:
                    pass

    # ---------------------------------------------------------------- finding / learning the game
    def _known_candidates(self, manual_board):
        """Games the screen might show, in order: your pasted game / a new game (a move or two in,
        e.g. Black after White's first move) then the last saved game (plus moves since)."""
        cands = []
        if manual_board is not None:
            cands.append(("your game", manual_board, 3))
        else:
            cands.append(("new game", chess.Board(), 2))
            live = self.game_board
            if live is None:
                live, _side = self._load_saved_game()
            if live is not None:
                cands.append(("saved game", live, 3))
        return cands

    def _match_known(self, raw, white_bottom, cands):
        reports = []
        for name, root, max_plies in cands:
            m, rep = smart_match(raw, white_bottom, root, max_plies)
            if m is not None:
                return m, name, reports
            reports.append(f"- {name}: {rep}")
        return None, None, reports

    def _adopt(self, m, raw):
        """Applies what the automatic troubleshooting found (alignment, detection settings, side)."""
        self.win, self.fp = m.win, m.fp
        x1, y1, x2, y2 = self.region
        bw, bh = x2 - x1, y2 - y1
        L, T, W, H = m.win
        self.eff_region = (x1 + L * bw, y1 + T * bh, x1 + (L + W) * bw, y1 + (T + H) * bh)
        if m.white_bottom != self.run_white_bottom:
            self.run_white_bottom = m.white_bottom
            side = "white" if m.white_bottom else "black"
            self._post(lambda: self._set_side(side))
        return crop_window(raw, m.win)

    def _find_game(self, sct, white_bottom, my_color, manual_board):
        """Works out the game from the screen, troubleshooting automatically (board alignment, flipped side,
        detection settings). Order: your pasted game / a new game -> the last saved game (+ moves since)
        -> recognising the pieces from the screen alone with the learned piece sets.
        Returns (board, occ_thr, col_thr, description, white_bottom)."""
        raw = self._grab_raw(sct)
        m, name, reports = self._match_known(raw, white_bottom, self._known_candidates(manual_board))
        if m is not None:
            img = self._adopt(m, raw)
            self._learn_now(img, m.board, m.white_bottom)
            self._turn_guessed = False
            self._can_learn = True
            # the turn is certain only when it follows from a known move history (new game / saved game)
            self._turn_uncertain = name not in ("new game", "saved game")
            return m.board, m.occ_thr, m.col_thr, name, m.white_bottom
        if manual_board is None:
            if not self.templates.ready():
                raise NeedGameText(
                    "I can't identify the pieces on my own yet, because this isn't a new game or the last game "
                    "I played, and I haven't learned your piece set.\n\nPaste the FEN (or moves) of this game "
                    "once. I'll learn how your pieces look, and from then on any game can be resumed with "
                    "nothing typed.")
            img = crop_window(raw, self.win) or crop_window(raw, (0.0, 0.0, 1.0, 1.0))
            try:
                board, occ, col, guessed = recognise_position(img, white_bottom, self.templates, my_color, self.fp)
                self._turn_guessed = guessed
                self._turn_uncertain = True
                self._can_learn = False      # piece types came from the screen, not from a known game
                return board, occ, col, "recognised from screen", white_bottom
            except PositionError as exc:
                reports.append(f"- screen only: {exc}")
        dbg = write_debug_images(raw, white_bottom, os.path.dirname(os.path.abspath(__file__)))
        msg = "Couldn't match the screen to the game.\n\nWhat I tried (alignment, flipped side, detection settings):\n"
        msg += "\n".join(reports) if reports else "- nothing"
        if manual_board is not None:
            msg += ("\n\nThe FEN / moves you entered must be the position that is on the board right now "
                    "(up to 3 moves before it).")
        if dbg:
            msg += ("\n\nA picture of what I see was saved to:\n" + dbg +
                    "\n(grid + a white/black dot on every square I think holds a piece - if it looks wrong, "
                    "re-calibrate the two board corners.)")
        raise PositionError(msg)

    def _learn_now(self, img, board, white_bottom):
        try:
            pieces = {sq: board.piece_at(sq).symbol() for sq in chess.SQUARES if board.piece_at(sq)}
            if self.templates.learn(img, white_bottom, pieces):
                self.templates.save()
        except Exception:
            pass

    def _maybe_learn(self, sct, board, white_bottom, occ_thr, col_thr):
        """Every few moves, re-learn the piece look from a settled frame (also picks up promotions)."""
        try:
            if not self._can_learn or len(board.move_stack) - self._last_learn_ply < 8 or self.stop_event.is_set():
                return
            self._sleep(0.3)
            img = self._grab(sct)
            if occ_equal(classify(img, white_bottom, occ_thr, col_thr, self.fp), board_signature(board)):
                self._learn_now(img, board, white_bottom)
                self._last_learn_ply = len(board.move_stack)
        except Exception:
            pass

    def _check_turn(self, sct, board, white_bottom):
        """Is it really my move? Uses the last-move highlight: the piece standing on it belongs to the side
        that just moved. Returns True (my move), False (opponent's move) or None (can't tell)."""
        try:
            img = self._grab(sct)
            assign = {sq: board.piece_at(sq).symbol() for sq in chess.SQUARES if board.piece_at(sq)}
            mover = detect_last_mover(img, white_bottom, assign)
        except Exception:
            return None
        if mover is None:
            return None
        return mover != board.turn          # the side to move did NOT just move

    def _play_my_move(self, sct, board, white_bottom, occ_thr, col_thr, engine, builtin):
        if self._turn_uncertain:
            # After resuming a game the side to move was not observed. Verify before clicking anything.
            self._turn_uncertain = False
            self._say("Checking whose move it is...")
            first = self._classify(sct, white_bottom, occ_thr, col_thr)
            self._sleep(0.5)
            if not occ_equal(self._classify(sct, white_bottom, occ_thr, col_thr), first):
                self._turn_uncertain = True              # the board is changing: look again next round
                return
            verdict = self._check_turn(sct, board, white_bottom)
            if verdict is False:
                board.turn = not board.turn              # it is the opponent's move
                self._turn_guessed = False
                self._say("It's the opponent's move - waiting...")
                return
            self._turn_guessed = verdict is None         # unknown: if the click does nothing, assume not my turn
        self._say("Thinking...")
        think = float(self._think)
        snapshot = board.copy()
        move = None
        if engine is not None:
            try:
                move = engine.play(snapshot, chess.engine.Limit(time=think)).move
            except Exception:
                move = None
        if move is None:
            move = builtin.find_best_move(snapshot, time_limit=think, max_depth=4 if think >= 1.2 else 3)
        if move is None or move not in board.legal_moves:      # strict legality guarantee
            legal = list(board.legal_moves)
            if not legal:
                self.stop_event.set()
                return
            move = legal[0]
        if self.stop_event.is_set():
            return

        san = board.san(move)
        pre_sig = board_signature(board)
        expected_board = board.copy()
        expected_board.push(move)
        expected = board_signature(expected_board)

        for attempt in (1, 2):
            if self.stop_event.is_set():
                return
            self._say(f"Playing {san}" + (" (retry)" if attempt == 2 else ""))
            self._click_move(move, white_bottom)
            deadline = time.time() + 4.0
            pending, count = None, 0
            while time.time() < deadline and not self.stop_event.is_set():
                obs = self._classify(sct, white_bottom, occ_thr, col_thr)
                if occ_equal(obs, expected):
                    board.push(move)
                    self._sync(board)
                    self._say(f"Played {san}. Waiting for opponent...")
                    return
                # A fast opponent may already have replied: accept "my move + their reply" too.
                reply = find_move_from_observation(expected_board, obs)
                if reply is not None:
                    count = count + 1 if reply == pending else 1
                    pending = reply
                    if count >= 2:
                        reply_san = expected_board.san(reply)
                        board.push(move)
                        board.push(reply)
                        self._sync(board)
                        self._say(f"Played {san}. Opponent replied {reply_san}.")
                        return
                else:
                    pending, count = None, 0
                self._sleep(0.15)
            # Only click again if the screen still shows the position before my move.
            if self.stop_event.is_set() or not occ_equal(self._classify(sct, white_bottom, occ_thr, col_thr), pre_sig):
                break

        if self._turn_guessed and not self.stop_event.is_set():
            # The turn was only a guess after recognising the screen: probably it is the opponent's move.
            self._turn_guessed = False
            board.turn = not board.turn
            self._say("Not my turn yet - waiting for the opponent...")
            return
        self._say(f"Move {san} did not register on screen - stopped. Check calibration / side, then press Continue game.")
        self._post(lambda: self._set_pill("error"))
        self.stop_event.set()

    def _wait_for_opponent(self, sct, board, white_bottom, occ_thr, col_thr):
        current = board_signature(board)
        pending, count = None, 0
        mismatch_since = None
        while not self.stop_event.is_set():
            obs = self._classify(sct, white_bottom, occ_thr, col_thr)
            if occ_equal(obs, current):
                pending, count, mismatch_since = None, 0, None
                self._sleep(0.2)
                continue
            mv = find_move_from_observation(board, obs)
            if mv is None:                       # mid-animation, or the screen is obstructed
                pending, count = None, 0
                mismatch_since = mismatch_since or time.time()
                if time.time() - mismatch_since > 20:
                    self._say("Screen doesn't match the expected position - is the board covered/moved? "
                              "(Stop, then Continue game to re-sync)")
                self._sleep(0.15)
                continue
            count = count + 1 if mv == pending else 1
            pending = mv
            if count >= 2:                       # stable for two frames
                san = board.san(mv)
                board.push(mv)
                self._sync(board)
                self._say(f"Opponent played {san}.")
                return
            self._sleep(0.12)


def main():
    root = tk.Tk()
    Aegis(root)
    root.mainloop()


if __name__ == "__main__":
    main()
